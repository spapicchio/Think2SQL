import os
import pathlib
import re
import threading
from functools import partial
from statistics import mean
from typing import Callable

from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteCacheDB
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import OrchestratorInput
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import (
    evaluator_orchestrator,
)
from NL2SQLEvaluator.orchestrator_state import AvailableMetrics, AvailableDialect

from think2sql.configs import GRPOScriptArguments
from think2sql.logger import get_logger
from think2sql.utils.data import utils_get_engine
from think2sql.utils.sql import get_sql_from_generation, check_crud_sql

# This is needed to have a different cache for each process
# when using multiprocessing training
REGISTRY_REWARD_CACHE = {}
_lock = threading.Lock()


def _get_cache_pred_db_based_on_pid(
        pid,
        cache_db_connections_path,
        target_sql_cache_db_path,
        pred_sql_cache_db_path=None,
) -> SqliteCacheDB | None:
    # the registry is needed for the multiprocessing training
    if cache_db_connections_path is not None:
        # first check if SQL in target cache_db, then in pred
        pred_cache_db = SqliteCacheDB.from_uri(
            relative_base_path=pred_sql_cache_db_path
        )
        target_cache_db = SqliteCacheDB.from_uri(
            relative_base_path=target_sql_cache_db_path,
            cache_db=pred_cache_db
        )

        cache_db_connections_path = pathlib.Path(cache_db_connections_path)
        new_db_path = cache_db_connections_path / str(pid) / f"cache_pred_{pid}.sqlite"
        new_db_path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        # create file
        if not new_db_path.exists():
            new_db_path.touch(mode=0o777)
        path = str(new_db_path)

        with _lock:
            if pid not in REGISTRY_REWARD_CACHE:
                REGISTRY_REWARD_CACHE[pid] = SqliteCacheDB.from_uri(
                    relative_base_path=path,
                    cache_db=target_cache_db if target_sql_cache_db_path else None,
                )
        return REGISTRY_REWARD_CACHE[pid]
    return None


def _parse_model_response(completion):
    if isinstance(completion, str):
        return completion
    elif "content" in completion[0]:
        return completion[0]["content"]
    else:
        return completion


def ex_reward(
        completions: list[dict | str | list],
        target_sql: list[str],
        db_id: list[str],
        relative_db_base_path: str,
        cache_db_connections_path: str,
        target_sql_cache_db_path: str,
        pred_sql_cache_db_path: str,
        save_in_cache: bool,
        *args,
        **kwargs,
) -> list[float]:
    # completion is STR when used in CorpusLevelMetric from lighteval
    assert len(target_sql) == len(completions)
    model_prediction = [_parse_model_response(completion) for completion in completions]
    predicted_sqls = [
        check_crud_sql(get_sql_from_generation(pred)) for pred in model_prediction
    ]
    cache = _get_cache_pred_db_based_on_pid(
        os.getpid(),
        cache_db_connections_path,
        target_sql_cache_db_path,
        pred_sql_cache_db_path,
    )
    orchestrator_input = OrchestratorInput(
        target_queries=target_sql,
        predicted_queries=predicted_sqls,
        executor=[
            utils_get_engine(
                relative_db_base_path, AvailableDialect("sqlite"), id_, cache
            )
            for id_ in db_id
        ],
        metrics_to_calculate=[
            AvailableMetrics.EXECUTION_ACCURACY.value,
        ],
        save_executed_query_in_cache=save_in_cache,
    )
    results = evaluator_orchestrator.invoke(orchestrator_input)
    results = [result[AvailableMetrics.EXECUTION_ACCURACY.value] for result in results]
    return results


def qatch_reward(
        completions: list[dict],
        target_sql: list[str],
        db_id: list[str],
        relative_db_base_path: str,
        cache_db_connections_path: str,
        target_sql_cache_db_path: str,
        pred_sql_cache_db_path: str,
        save_in_cache: bool,
        *args,
        **kwargs,
):
    assert len(target_sql) == len(completions)
    model_prediction = [_parse_model_response(completion) for completion in completions]
    predicted_sqls = [
        check_crud_sql(get_sql_from_generation(pred)) for pred in model_prediction
    ]
    cache = _get_cache_pred_db_based_on_pid(
        os.getpid(),
        cache_db_connections_path,
        target_sql_cache_db_path,
        pred_sql_cache_db_path,
    )
    orchestrator_input = OrchestratorInput(
        target_queries=target_sql,
        predicted_queries=predicted_sqls,
        executor=[
            utils_get_engine(
                relative_db_base_path, AvailableDialect("sqlite"), id_, cache
            )
            for id_ in db_id
        ],
        metrics_to_calculate=[
            AvailableMetrics.CELL_PRECISION.value,
            AvailableMetrics.CELL_RECALL.value,
            AvailableMetrics.TUPLE_CARDINALITY.value,
        ],
        save_executed_query_in_cache=save_in_cache,
    )
    results = evaluator_orchestrator.invoke(orchestrator_input)
    results = [mean(list(result.values())) for result in results]
    return results


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def get_reward_funcs(script_args: GRPOScriptArguments, save_in_cache=True) -> list[Callable]:
    logger = get_logger(__name__)
    execution_accuracy_fn = partial(
        ex_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        cache_db_connections_path=script_args.cache_db_connections_path,
        target_sql_cache_db_path=script_args.target_sql_cache_db_path,
        pred_sql_cache_db_path=script_args.pred_sql_cache_db_path,
        save_in_cache=save_in_cache,
    )
    # This is to make sure the function name is correct in the logs and can be used with lighteval
    execution_accuracy_fn.__name__ = ex_reward.__name__
    qatch_reward_fn = partial(
        qatch_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        cache_db_connections_path=script_args.cache_db_connections_path,
        target_sql_cache_db_path=script_args.target_sql_cache_db_path,
        pred_sql_cache_db_path=script_args.pred_sql_cache_db_path,
        save_in_cache=save_in_cache,
    )
    qatch_reward_fn.__name__ = qatch_reward.__name__
    # This is to make sure the function name is correct in the logs and can be used with lighteval

    REWARD_FUNCS_REGISTRY = {
        "EX": execution_accuracy_fn,
        "QATCH": qatch_reward_fn,
        "format": format_reward,
        "tag_count": tag_count_reward,
    }

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    logger.info(f"loaded reward functions: {script_args.reward_funcs}")
    return reward_funcs


if __name__ == "__main__":
    cache_db = _get_cache_pred_db_based_on_pid(
        pid=183742,
        cache_db_connections_path=".think2sql_cache",
        target_sql_cache_db_path=".think2sql_cache/target_cached_query.sqlite",
        pred_sql_cache_db_path=".think2sql_cache/pred_cached_query.sqlite",
    )
    cache_db.insert_in_cache(
        db_id="table_a",
        query="SELECT * FROM table_a WHERE column_b = 'value1'",
        result=[('value1', 'value2'), ('value3', 'value4')],
    )
