import multiprocessing as mp
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Protocol

from func_timeout import FunctionTimedOut, func_timeout

from think2sql.logger import get_logger
from think2sql.utils.sql import check_crud_sql, get_sql_from_generation

logger = get_logger(__name__)


class Evaluator(Protocol):
    def evaluate(self,
                 target_queries: list[str],
                 llm_predictions: list[list[str]],
                 db_files: list[str],
                 *args, **kwargs) -> list[float]:
        ...


class SqliteEvaluatorEX:
    def __init__(self):
        self.execution_results: list[int] = []

    def evaluate(
            self,
            target_queries: list[str],
            llm_predictions: list[list[str] | str],
            db_files: list[str],
            *args, **kwargs
    ) -> list[int]:
        """
        Returns a list[int] of EX scores (0/1) per (db_file, target_query, predictions) triple.
        """
        # reset per run to avoid leaking across calls
        self.execution_results = []

        llm_predictions = llm_predictions if isinstance(llm_predictions[0], list) else [[pred] for pred in
                                                                                        llm_predictions]
        num_cpus = kwargs.get(
            'num_cpus',
            min(max(1, mp.cpu_count() // 2), max(1, len(llm_predictions)))
        )
        timeout = kwargs.get('timeout', 10)

        # Build one flat list of tasks (kind, job_id, pred_idx, db_file, sql, timeout)
        # kind: 'gt' for ground truth, 'pred' for model predictions
        tasks: list[tuple[str, int, int, str, str, int]] = []
        for job_id, (db_file, preds, gt) in enumerate(zip(db_files, llm_predictions, target_queries)):
            tasks.append(('gt', job_id, -1, db_file, gt, timeout))
            for i, sql in enumerate(preds):
                tasks.append(('pred', job_id, i, db_file, sql, timeout))

        # Single pool for all executions
        with mp.Pool(processes=num_cpus) as pool:
            # Each result: (job_id, kind, pred_idx, rows)
            results = pool.starmap(
                SqliteEvaluatorEX._worker_execute_sql_tagged,
                tasks
            )

        # Group by job_id
        grouped = defaultdict(lambda: {'gt': None, 'preds': []})
        for job_id, kind, pred_idx, rows in results:
            if kind == 'gt':
                grouped[job_id]['gt'] = rows
            else:
                grouped[job_id]['preds'].append((pred_idx, rows))

        # For each job_id (in order), compute majority vote vs GT
        num_jobs = len(db_files)
        ex_per_job: list[int] = []
        for job_id in range(num_jobs):
            gt_rows = grouped[job_id]['gt'] or []
            preds = grouped[job_id]['preds']

            # sort by pred_idx to be deterministic before counting
            preds.sort(key=lambda x: x[0])
            # majority over *row sets* (order-insensitive)
            counter = Counter(tuple(p_rows) for _, p_rows in preds)
            if counter:
                majority_pred, _ = counter.most_common(1)[0]
                ex = int(set(majority_pred) == set(gt_rows))
            else:
                ex = 0  # no valid predictions -> fail

            ex_per_job.append(ex)

        self.execution_results = ex_per_job
        return self.execution_results

    # ---------- workers ----------

    @staticmethod
    def _worker_execute_sql_tagged(
            kind: str,
            job_id: int,
            pred_idx: int,
            db_file: str,
            sql_query: str,
            timeout: int
    ):
        """
        Executes SQL and returns rows, tagged with (job_id, kind, pred_idx).
        - For predictions, we sanitize with get_sql_from_generation + check_crud_sql.
        - For gt, we run as given (it should be a SELECT).
        """
        try:
            if kind == 'pred':
                sql_query = check_crud_sql(get_sql_from_generation(sql_query))
            # else: ground truth assumed to be a valid, safe SELECT

            rows = SqliteEvaluatorEX._execute_sql_with_timeout(
                db_file, sql_query, timeout
            )
        except KeyboardInterrupt:
            # Ensure proper termination behavior in pool
            sys.exit(1)
        except Exception:
            rows = []

        return (job_id, kind, pred_idx, rows)

    @staticmethod
    def _execute_sql_with_timeout(db_file: str, sql_query: str, timeout: int):
        def _internal():
            conn = None
            output = []
            try:
                conn = sqlite3.connect(db_file)
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA query_only=ON;")
                cur = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                cur.execute(sql_query)
                output = cur.fetchall()
            finally:
                try:
                    if conn is not None:
                        conn.rollback()
                        conn.close()
                finally:
                    pass
            return output

        try:
            return func_timeout(timeout, _internal)
        except (FunctionTimedOut, Exception):
            return []
