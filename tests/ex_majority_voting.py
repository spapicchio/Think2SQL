import random
import sqlite3
from pathlib import Path

import pytest

from think2sql.evaluate.evaluators import SqliteEvaluatorEX

# --- Helpers -----------------------------------------------------------------

def create_sqlite_db(tmp_path: Path, name: str, rows):
    """
    Create a tiny SQLite DB with a single table t(id, val) and insert `rows`
    (iterable of tuples). Returns the db file path.
    """
    db = tmp_path / f"{name}.sqlite"
    conn = sqlite3.connect(db.as_posix())
    try:
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, val TEXT);")
        conn.executemany("INSERT INTO t(id, val) VALUES(?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return db.as_posix()


@pytest.fixture(autouse=True)
def seed_random():
    random.seed(123)


@pytest.fixture
def monkeypatch_sql_sanitizers(monkeypatch):
    """
    Make check_crud_sql/get_sql_from_generation no-ops if they exist in the module.
    This keeps tests independent of external project code.
    """
    import think2sql.evaluate.evaluators as mod
    monkeypatch.setattr(mod, "check_crud_sql", lambda s: s, raising=False)
    monkeypatch.setattr(mod, "get_sql_from_generation", lambda s: s, raising=False)


# --- Tests -------------------------------------------------------------------

def test_majority_matches_gt_returns_1(tmp_path, monkeypatch_sql_sanitizers):
    """
    Majority of predictions produce the same rows as GT -> EX == 1.
    Also validates order-insensitive matching (results can be returned in any order).
    """
    db = create_sqlite_db(tmp_path, "db_ok", rows=[(1, "a"), (2, "b"), (3, "c")])

    # Ground truth returns rows in ID ASC
    gt = "SELECT id, val FROM t ORDER BY id ASC;"

    # Preds: 2 equal to GT (one unordered), 1 wrong
    preds = [
        "SELECT id, val FROM t ORDER BY id ASC;",  # exact
        "SELECT id, val FROM t ORDER BY id DESC;",  # same set, reversed order
        "SELECT id, val FROM t WHERE id < 3;"  # wrong (subset)
    ]

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=2,  # small but > 0
        timeout=5
    )

    assert res == [1], f"Expected EX==1, got {res}"


def test_majority_differs_from_gt_returns_0(tmp_path, monkeypatch_sql_sanitizers):
    """
    Majority of predictions return a different set from GT -> EX == 0.
    """
    db = create_sqlite_db(tmp_path, "db_bad", rows=[(1, "a"), (2, "b"), (3, "c")])

    gt = "SELECT id, val FROM t WHERE id IN (1,2) ORDER BY id;"
    preds = [
        "SELECT id, val FROM t WHERE id IN (2,3) ORDER BY id;",
        "SELECT id, val FROM t WHERE id IN (2,3) ORDER BY id;",
        "SELECT id, val FROM t WHERE id IN (1,2) ORDER BY id;"  # minority correct
    ]

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=2,
        timeout=5
    )
    assert res == [0]

def test_execution_acc_single_pred(tmp_path, monkeypatch_sql_sanitizers):
    db = create_sqlite_db(tmp_path, "db_bad", rows=[(1, "a"), (2, "b"), (3, "c")])

    gt = "SELECT id, val FROM t WHERE id IN (1,2) ORDER BY id;"
    pred = "SELECT id, val FROM t WHERE id IN (2,3) ORDER BY id;"
    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[pred],
        db_files=[db],
        num_cpus=2,
        timeout=5
    )
    assert res == [0]

def test_execution_acc_single_pred(tmp_path, monkeypatch_sql_sanitizers):
    db = create_sqlite_db(tmp_path, "db_bad", rows=[(1, "a"), (2, "b"), (3, "c")])

    gt = "SELECT id, val FROM t WHERE id IN (1,2) ORDER BY id;"
    pred = "SELECT id, val FROM t WHERE id IN (1,2) ORDER BY id;"
    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[pred],
        db_files=[db],
        num_cpus=2,
        timeout=5
    )
    assert res == [1]


def test_empty_predictions_yield_0(tmp_path, monkeypatch_sql_sanitizers):
    """
    No predictions -> counter is empty -> EX == 0.
    """
    db = create_sqlite_db(tmp_path, "db_empty_preds", rows=[(1, "a")])
    gt = "SELECT id, val FROM t ORDER BY id;"

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[[]],  # no predictions
        db_files=[db],
        num_cpus=1,
        timeout=5
    )
    assert res == [0]


def test_invalid_prediction_sql_is_ignored_as_empty(tmp_path, monkeypatch, monkeypatch_sql_sanitizers):
    """
    If a prediction raises during execution, worker returns empty [].
    Majority still computed over all predictions.
    We simulate invalid SQL by monkeypatching _execute_sql_with_timeout to raise.
    """
    db = create_sqlite_db(tmp_path, "db_invalid", rows=[(1, "a"), (2, "b")])

    gt = "SELECT id, val FROM t ORDER BY id;"
    preds = [
        "SELECT id, val FROM t ORDER BY id;",  # valid
        "THIS IS NOT SQL",  # will be forced to raise by monkeypatch
        "SELECT id, val FROM t ORDER BY id;"  # valid
    ]

    def fake_exec(db_file, sql_query, timeout):
        if "NOT SQL" in sql_query:
            raise RuntimeError("boom")
        # execute real for others
        conn = sqlite3.connect(db_file)
        try:
            cur = conn.execute(sql_query)
            return cur.fetchall()
        finally:
            conn.close()

    import think2sql.evaluate.evaluators as mod
    monkeypatch.setattr(mod.SqliteEvaluatorEX, "_execute_sql_with_timeout", staticmethod(fake_exec))

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=2,
        timeout=5
    )
    # 2 valid predictions match GT => majority equals GT => EX == 1
    assert res == [1]


def test_timeout_predictions_count_as_empty(tmp_path, monkeypatch, monkeypatch_sql_sanitizers):
    """
    Simulate a timeout on specific predictions: they should return [] and not crash.
    Majority still computed correctly.
    """
    db = create_sqlite_db(tmp_path, "db_timeout", rows=[(1, "x"), (2, "y"), (3, "z")])

    gt = "SELECT id, val FROM t ORDER BY id;"

    preds = [
        "SELECT id, val FROM t ORDER BY id;",  # ok
        "TIMEOUT_TAG",  # simulate timeout
        "SELECT id, val FROM t ORDER BY id;"  # ok
    ]

    def fake_exec(db_file, sql_query, timeout):
        if sql_query == "TIMEOUT_TAG":
            # Mimic timeout behavior by returning [] as in the class
            return []
        conn = sqlite3.connect(db_file)
        try:
            cur = conn.execute(sql_query)
            return cur.fetchall()
        finally:
            conn.close()

    import think2sql.evaluate.evaluators as mod
    monkeypatch.setattr(mod.SqliteEvaluatorEX, "_execute_sql_with_timeout", staticmethod(fake_exec))

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=2,
        timeout=1
    )
    assert res == [1]


def test_order_insensitive_set_equality(tmp_path, monkeypatch_sql_sanitizers):
    """
    Even if prediction returns same rows in a different order, it should match GT.
    """
    db = create_sqlite_db(tmp_path, "db_order", rows=[(1, "a"), (2, "b"), (3, "c")])

    gt = "SELECT id, val FROM t ORDER BY id ASC;"
    preds = [
        "SELECT id, val FROM t ORDER BY id DESC;",
        "SELECT id, val FROM t ORDER BY id DESC;",
        "SELECT id, val FROM t WHERE id > 1 ORDER BY id;"  # minority mismatch
    ]

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=2,
        timeout=5
    )
    assert res == [1]


def test_multiple_jobs_processed_and_state_resets(tmp_path, monkeypatch_sql_sanitizers):
    """
    Two independent jobs (two DBs) are evaluated; ensure:
    - we get two EX scores in order,
    - calling evaluate twice resets internal state (no leakage).
    """
    db1 = create_sqlite_db(tmp_path, "db1", rows=[(1, "a"), (2, "b")])
    db2 = create_sqlite_db(tmp_path, "db2", rows=[(10, "x"), (20, "y")])

    gt1 = "SELECT id, val FROM t ORDER BY id;"
    gt2 = "SELECT id, val FROM t WHERE id = 10;"  # only one row

    preds1 = [
        "SELECT id, val FROM t ORDER BY id;",
        "SELECT id, val FROM t ORDER BY id DESC;",
        "SELECT id, val FROM t WHERE id = 1;"
    ]  # majority == GT -> EX1 = 1

    preds2 = [
        "SELECT id, val FROM t WHERE id = 20;",  # wrong
        "SELECT id, val FROM t WHERE id = 20;",  # wrong (majority)
        "SELECT id, val FROM t WHERE id = 10;"  # correct
    ]  # majority != GT -> EX2 = 0

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt1, gt2],
        llm_predictions=[preds1, preds2],
        db_files=[db1, db2],
        num_cpus=2,
        timeout=5
    )
    assert res == [1, 0]

    # call again with only job1 and ensure no leakage from previous call
    res2 = evaluator.evaluate(
        target_queries=[gt1],
        llm_predictions=[preds1],
        db_files=[db1],
        num_cpus=2,
        timeout=5
    )
    assert res2 == [1]


def test_gt_empty_rows_and_majority_empty_is_success(tmp_path, monkeypatch_sql_sanitizers):
    """
    If GT returns empty [] and the majority prediction also returns [],
    EX should be 1.
    """
    db = create_sqlite_db(tmp_path, "db_empty_sets", rows=[(1, "a")])

    gt = "SELECT id, val FROM t WHERE id < 0;"  # empty
    preds = [
        "SELECT id, val FROM t WHERE id < 0;",  # empty
        "SELECT id, val FROM t WHERE id < 0;",  # empty -> majority empty
        "SELECT id, val FROM t;"  # non-empty (minority)
    ]

    evaluator = SqliteEvaluatorEX()
    res = evaluator.evaluate(
        target_queries=[gt],
        llm_predictions=[preds],
        db_files=[db],
        num_cpus=1,
        timeout=5
    )
    assert res == [1]
