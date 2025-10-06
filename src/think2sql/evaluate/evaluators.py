import multiprocessing as mp
import sqlite3
import sys
from typing import Protocol

from func_timeout import FunctionTimedOut, func_timeout

from think2sql.logger import get_logger
from think2sql.utils.sql import check_crud_sql, get_sql_from_generation

logger = get_logger(__name__)


class Evaluator(Protocol):
    def evaluate(self,
                 target_queries: list[str],
                 llm_predictions: list[str],
                 db_files: list[str],
                 *args, **kwargs) -> list[float]:
        ...


def _worker_compare_sqls(db_file, ground_truth, llm_prediction, timeout, **kwargs) -> float:
    def _internal():
        ex = 0
        try:
            pred_sql = check_crud_sql(get_sql_from_generation(llm_prediction))
            # Read-only best-effort: query_only; also disable triggers if you can (not standard)
            conn = sqlite3.connect(db_file)
            try:
                conn.execute("PRAGMA foreign_keys=ON;")  # Enforce foreign keys
                conn.execute("PRAGMA query_only=ON;")  # Read-only mode
                cur = conn.cursor()

                conn.execute("BEGIN TRANSACTION;")
                cur.execute(pred_sql)
                pred_rows = cur.fetchall()

                cur.execute(ground_truth)
                gt_rows = cur.fetchall()

                # Multiset (order-insensitive, multiplicity-sensitive)
                # rec["EX"] = int(Counter(pred_rows) == Counter(gt_rows))
                if set(pred_rows) == set(gt_rows):
                    ex = 1

            finally:
                # Rollback to be extra safe
                try:
                    conn.rollback()
                finally:
                    conn.close()

        except Exception as e:
            ...
        return ex

    try:
        result = func_timeout(timeout, _internal)
    except KeyboardInterrupt:
        sys.exit(1)
    except (FunctionTimedOut, Exception):
        result = 0
    return result


class SqliteEvaluatorEX:
    def __init__(self):
        self.execution_results = []

    def evaluate(self,
                 target_queries: list[str],
                 llm_predictions: list[str],
                 db_files: list[str],
                 *args, **kwargs) -> list[float]:
        num_cpus = kwargs.get('num_cpus', mp.cpu_count())
        timeout = kwargs.get('timeout', 10)
        self._evaluate_sql_in_parallel(
            db_files,
            target_queries,
            llm_predictions,
            num_cpus,
            timeout,
            *args,
            **kwargs,
        )
        return self.execution_results

    def _evaluate_sql_in_parallel(self,
                                  db_files,
                                  target_queries,
                                  llm_predictions,
                                  num_cpus,
                                  timeout=10,
                                  *args,
                                  **kwargs):
        pool = mp.Pool(processes=num_cpus)
        for db_file, llm_prediction, ground_truth in zip(
                db_files,
                llm_predictions,
                target_queries
        ):
            pool.apply_async(
                _worker_compare_sqls,
                args=(db_file, ground_truth, llm_prediction, timeout),
                callback=self.execution_results.append
            )
        pool.close()
        pool.join()


class Nl2SQLEvaluator:
    def __init__(self):
        ...

    def evaluate(self,
                 target_queries: list[str],
                 llm_predictions: list[str],
                 db_files: list[str],
                 *args, **kwargs) -> list[float]:
        ...
