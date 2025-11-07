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


def reward_selected_tables(
        completions: list[dict],
        tbls_in_query: list[list[str]],
        db_id: list[str],
        *args,
        **kwargs,
) -> list[float]:
    """Calculate the recall of the tables in the completions"""
    model_predictions = [_parse_model_response(completion) for completion in completions]
    rewards = []
    for model_pred, tbls in zip(model_predictions, tbls_in_query):
        predicted_tbls = extract_from_completion_with('tables', model_pred)
        if predicted_tbls is None:
            rewards.append(0.0)
            continue
        predicted_tbls_set = {t.strip().lower() for t in predicted_tbls.split(',')}
        true_tbls_set = {t.strip().lower() for t in tbls}
        intersection = predicted_tbls_set.intersection(true_tbls_set)
        reward = len(intersection) / len(true_tbls_set) if true_tbls_set else 0.0
        rewards.append(reward)
    return rewards


def reward_selected_columns(
        completions: list[dict],
        cols_in_query: list[list[str]],
        db_id: list[str],
        *args,
        **kwargs,
) -> list[float]:
    """Calculate the recall of the cols in the completions"""
    model_predictions = [_parse_model_response(completion) for completion in completions]
    rewards = []
    for model_pred, cols in zip(model_predictions, cols_in_query):
        predicted_cols = extract_from_completion_with('columns', model_pred)
        if predicted_cols is None:
            rewards.append(0.0)
            continue
        predicted_cols_set = {t.strip().lower() for t in predicted_cols.split(',')}
        true_cols_set = {t.strip().lower() for t in cols}
        intersection = predicted_cols_set.intersection(true_cols_set)
        reward = len(intersection) / len(true_cols_set) if true_cols_set else 0.0
        rewards.append(reward)
    return rewards


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


def multi_tag_format_reward(completions, **kwargs):
    """
    Sparse format reward for:
      <reasoning>...</reasoning>
      <tables>...</tables>
      <columns>...</columns>
      <checks>...</checks>
      <answer>...</answer>

    Returns a list of floats in [0,1] with partial credit by component.
    You can tweak weights to emphasize specific parts.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<tables>\n.*?\n</tables>\n<columns>\n.*?\n</columns>\n<checks>\n.*?\n</checks>\n<answer>\n.*?\n</answer>$"
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


def extract_from_completion_with(tag: str, completion: str) -> str | None:
    pattern = rf"<{tag}>\n(.*?)\n</{tag}>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


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
        "multi_tag_format": multi_tag_format_reward,
        "table_recall": reward_selected_tables,
        "column_recall": reward_selected_columns,
    }

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    logger.info(f"loaded reward functions: {script_args.reward_funcs}")
    return reward_funcs
