import os
import re
import threading
from functools import partial
from typing import Callable

from NL2SQLEvaluator.db_executor_nodes import SQLiteDBExecutor, SqliteCache
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecuteTask, extract_sql_or_same
from NL2SQLEvaluator.evaluator_nodes import BirdEXEvaluator
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask, EvaluatorProtocol
from NL2SQLEvaluator.evaluator_nodes.qatch_metrics import QATCHEvaluator

from think2sql.configs import GRPOScriptArguments
from think2sql.logger import get_logger

# This is needed to have a different cache for each process
# when using multiprocessing training
REGISTRY_REWARD_CACHE = {}
_lock = threading.Lock()


def _parse_model_response(completion) -> str | list[str]:
    if isinstance(completion, str):
        return completion
    elif "content" in completion[0]:
        pred = [val['content'] for val in completion]
        return pred if len(pred) > 1 else pred[0]
    else:
        return completion


def nl2sql_reward(
        completions: list[dict],
        target_sql: list[str],
        db_id: list[str],
        evaluator: EvaluatorProtocol,
        relative_db_base_path: str,
        *args,
        **kwargs,
) -> list[float]:
    # run in parallel predictions and targets
    model_predictions = [_parse_model_response(completion) for completion in completions]
    target_sql_results, model_predictions_results = _execute_target_and_pred_sql(
        db_id,
        target_sql,
        model_predictions,
        relative_db_base_path
    )
    # make evaluation
    tasks = [
        EvaluateTask(
            predictions=[pred for pred in preds if isinstance(pred, OutputTable)],
            target=[tar for tar in tars if isinstance(tar, OutputTable)]  # assuming all targets are correct
        )
        for tars, preds in zip(target_sql_results, model_predictions_results)
    ]

    scores = evaluator.execute_metric(
        tasks=tasks,
        metric='cp_cr_tc',
        *args,
        **kwargs
    )

    return scores


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


def _execute_target_and_pred_sql(db_ids: list[str],
                                 target_sqls: list[str],
                                 pred_sqls: list[str],
                                 relative_db_base_path: str):
    db_files = [os.path.join(relative_db_base_path, id_, f"{id_}.sqlite") for id_ in db_ids] * 2
    query_to_execute = target_sqls + pred_sqls

    tasks = [
        ExecuteTask(db_files=db_file, queries=[extract_sql_or_same(query)])
        for db_file, query in zip(db_files, query_to_execute)
    ]
    executed_queries = SQLiteDBExecutor().execute_queries(
        tasks=tasks,
        timeout=500,
        cache_db=SqliteCache(),
        cache_db_file='.nl2sql_cache/train_omnisql_cache.sqlite'
    )

    target_sql_results = executed_queries[: len(target_sqls)]
    model_predictions_results = executed_queries[len(target_sqls):]
    return target_sql_results, model_predictions_results


def get_reward_funcs(script_args: GRPOScriptArguments, save_in_cache=True) -> list[Callable]:
    logger = get_logger(__name__)
    execution_accuracy_fn = partial(
        nl2sql_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        evaluator=BirdEXEvaluator()
    )
    # This is to make sure the function name is correct in the logs and can be used with lighteval
    execution_accuracy_fn.__name__ = 'execution_accuracy'
    qatch_reward_fn = partial(
        nl2sql_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        evaluator=QATCHEvaluator(),
        metric='cp_cr_tc',
    )
    qatch_reward_fn.__name__ = 'qatch_reward'
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
