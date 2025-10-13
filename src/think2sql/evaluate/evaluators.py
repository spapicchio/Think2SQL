import multiprocessing as mp
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Literal, Sequence

from func_timeout import FunctionTimedOut, func_timeout

from think2sql.logger import get_logger
from think2sql.utils.sql import get_sql_from_generation, check_crud_sql

logger = get_logger(__name__)


class Evaluator(Protocol):
    def evaluate(self,
                 target_queries: list[str],
                 llm_predictions: list[list[str]],
                 db_files: list[str],
                 *args, **kwargs) -> list[float]:
        ...


Kind = Literal["gt", "pred"]


@dataclass(frozen=True)
class Task:
    """A single SQL execution task."""
    kind: Kind  # 'gt' or 'pred'
    job_id: int  # index of the (db_file, target_query, predictions) triple
    pred_idx: int  # -1 for gt; 0..N-1 for predictions
    db_file: str
    sql: str
    timeout_s: int


@dataclass(frozen=True)
class Result:
    """Result of executing a Task."""
    job_id: int
    kind: Kind
    pred_idx: int
    rows: Sequence[tuple] | None


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
        self.execution_results = []  # avoid leaking across calls
        # Validate inputs early
        self._validate_inputs(target_queries, llm_predictions, db_files)

        # Normalize predictions to List[List[str]]
        preds_per_item = [
            p if isinstance(p, list) else [p]  # wrap single string into a list
            for p in llm_predictions
        ]

        num_cpus = kwargs.get("num_cpus", min(max(1, mp.cpu_count()), max(1, len(preds_per_item))))
        timeout_s = kwargs.get("timeout", 10)

        # Build one flat list of tasks (kind, job_id, pred_idx, db_file, sql, timeout)
        # kind: 'gt' for ground truth, 'pred' for model predictions
        tasks = self._build_tasks(db_files, target_queries, preds_per_item, timeout_s)

        # Execute all tasks with a single pool; starmap preserves order of 'tasks'
        with mp.Pool(processes=num_cpus) as pool:
            raw_results = pool.starmap(
                SqliteEvaluatorEX._worker_execute_sql_tagged,
                [(t.kind, t.job_id, t.pred_idx, t.db_file, t.sql, t.timeout_s) for t in tasks]
            )

        results = [Result(*r) for r in raw_results]
        ex_per_job = self._compute_ex_per_job(results, num_jobs=len(db_files))
        self.execution_results = ex_per_job
        return self.execution_results

    def _compute_ex_per_job(self, results: list[Result], num_jobs: int) -> list[int]:
        """
        Compute EX per job_id based on grouped execution results and majority voting.
        """
        grouped = self._group_results_by_job(results)
        ex_per_job: list[int] = []

        for job_id in range(num_jobs):
            item = grouped.get(job_id, {"gt": None, "preds": []})
            gt_rows = item["gt"]  # type: ignore[assignment]
            preds: list[tuple[int, Sequence[tuple]]] = item["preds"]  # type: ignore[assignment]

            if gt_rows is None:
                # If GT failed to execute, we conservatively mark as incorrect (0).
                ex_per_job.append(0)
                continue

            majority_rows = self._majority_vote_rows(preds)
            if majority_rows is None:
                # No valid predictions -> EX=0
                ex_per_job.append(0)
                continue

            # Compare as sets (order-insensitive)
            ex = int(set(majority_rows) == set(gt_rows))  # type: ignore[arg-type]
            ex_per_job.append(ex)

        return ex_per_job

    # ---------- workers ----------
    @staticmethod
    def _validate_inputs(
            target_queries: list[str],
            llm_predictions: list[list[str] | str],
            db_files: list[str]
    ) -> None:
        if not (len(target_queries) == len(llm_predictions) == len(db_files)):
            raise ValueError(
                "Lengths must match: target_queries, llm_predictions, db_files."
            )
        if len(target_queries) == 0:
            raise ValueError("Empty inputs: nothing to evaluate.")

    @staticmethod
    def _build_tasks(
            db_files: list[str],
            target_queries: list[str],
            preds_per_item: list[list[str]],
            timeout_s: int
    ) -> list[Task]:
        """
        Create a flat list of execution tasks. One GT + N preds per item.
        """
        tasks: list[Task] = []
        for job_id, (db, gt, preds) in enumerate(zip(db_files, target_queries, preds_per_item)):
            tasks.append(Task(kind="gt", job_id=job_id, pred_idx=-1, db_file=db, sql=gt, timeout_s=timeout_s))
            for i, sql in enumerate(preds):
                normalized_sql = check_crud_sql(get_sql_from_generation(sql))
                tasks.append(
                    Task(kind="pred", job_id=job_id, pred_idx=i, db_file=db, sql=normalized_sql, timeout_s=timeout_s)
                )
        return tasks

    @staticmethod
    def _group_results_by_job(results: list[Result]) -> dict[int, dict[str, object]]:
        """
        Group results by job_id.

        Returns a dict:
          job_id -> {'gt': rows_gt (Sequence[tuple] | None),
                     'preds': list[(pred_idx, rows)] with rows != None}
        """
        grouped: dict[int, dict[str, object]] = defaultdict(lambda: {"gt": None, "preds": []})
        for res in results:
            if res.kind == "gt":
                grouped[res.job_id]["gt"] = res.rows
            else:  # pred
                if res.rows is not None:
                    grouped[res.job_id]["preds"].append((res.pred_idx, res.rows))
        return grouped

    @staticmethod
    def _majority_vote_rows(pred_rows_list: list[tuple[int, Sequence[tuple]]]) -> frozenset[tuple] | None:
        """
        Majority vote over row-sets (order-insensitive).
        Input: list of (pred_idx, rows).
        Returns the rows (as a tuple-of-tuples representative) that won the vote, or None if empty.
        """
        if not pred_rows_list:
            return None

        # Sort by pred_idx for determinism before counting
        pred_rows_list = sorted(pred_rows_list, key=lambda x: x[0])
        # Convert each rows (Sequence[tuple]) to an order-insensitive canonical form.
        # We use a frozenset of rows for equality;
        counter = Counter()
        for _, rows in pred_rows_list:
            if rows is None:
                # case when the SQL query is invalid
                continue
            counter[frozenset(rows)] += 1

        if not counter:
            return None

        best_canonical, _ = counter.most_common(1)[0]
        return best_canonical

    # ---------- Worker adapter (kept static to be picklable) ----------

    @staticmethod
    def _worker_execute_sql_tagged(
            kind: Kind,
            job_id: int,
            pred_idx: int,
            db_file: str,
            sql: str,
            timeout_s: int
    ) -> tuple[int, Kind, int, Sequence[tuple] | None]:
        """
        Executes the SQL with a timeout and returns (job_id, kind, pred_idx, rows_or_none).
        Return None in case of any error or timeout.
        """

        def _internal():
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA query_only=ON;")
                cur = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                cur.execute(sql)
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
            rows = func_timeout(timeout_s, _internal)
        except (FunctionTimedOut, Exception):
            rows = None
        return job_id, kind, pred_idx, rows
