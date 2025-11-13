import multiprocessing as mp
import os
import re
import time
from functools import partial
from typing import Callable, Any

import torch
from NL2SQLEvaluator.db_executor_nodes import SQLiteDBExecutor, SqliteCache
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import (
    ExecuteTask,
    extract_sql_or_same, ExecutorError,
)
from NL2SQLEvaluator.evaluator_nodes import BirdEXEvaluator
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import (
    EvaluateTask,
    EvaluatorProtocol,
)
from NL2SQLEvaluator.evaluator_nodes.qatch_metrics import QATCHEvaluator

from think2sql.configs import GRPOScriptArguments
from think2sql.logger import get_logger

TRAINING_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = max(1, mp.cpu_count())
AVAIL_CPU_PER_GPU = AVAILABLE_CPUS // max(1, TRAINING_GPUS)


def _read_counter(name: str) -> Any:
    return globals().get(name, 0)


def _parse_model_response(completion: list[dict]) -> list[str] | str:
    # completion may be composed of different decoded sequences
    # most probably during training we will have only one decoded sequence
    pred: list[str] = [val["content"] for val in completion]
    return pred if len(pred) > 1 else pred[0]


def reward_selected_tables(
        completions: list[list[dict]],
        tbls_in_query: list[list[str]],
        *args,
        **kwargs,
) -> list[float]:
    """Calculate the recall of the tables in the completions"""
    logger = get_logger("REWARD_TABLES")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger.info(
        f"[REWARD_TABLES][START][{hash_id}] Calculating reward_selected_tables for {len(completions)} completions"
    )
    model_predictions = [_parse_model_response(val) for val in completions]

    assert len(model_predictions) == len(tbls_in_query), (
        "Length mismatch between completions and tables in query"
    )

    rewards = []
    for model_pred, tbls in zip(model_predictions, tbls_in_query):
        predicted_tbls = extract_from_completion_with("tables", model_pred)
        if predicted_tbls is None:
            rewards.append(0.0)
            continue
        predicted_tbls_set = {
            t.strip().lower().replace("`", "").replace('"', "").replace("'", "")
            for t in predicted_tbls.split(",")
        }
        true_tbls_set = {t.strip().lower() for t in tbls}
        intersection = predicted_tbls_set.intersection(true_tbls_set)
        reward = len(intersection) / len(true_tbls_set) if true_tbls_set else 0.0
        rewards.append(reward)
    logger.info(
        f"[REWARD_TABLES][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return rewards


def reward_selected_columns(
        completions: list[list[dict]],
        cols_in_query: list[list[str]],
        *args,
        **kwargs,
) -> list[float]:
    """Calculate the recall of the cols in the completions"""
    logger = get_logger("REWARD_TABLES")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger.info(
        f"[REWARD_COLS][START][{hash_id}] Calculating reward_selected_columns for {len(completions)} completions"
    )
    model_predictions = [_parse_model_response(val) for val in completions]
    assert len(model_predictions) == len(cols_in_query), (
        "Length mismatch between completions and columns in query"
    )
    rewards = []
    for model_pred, cols in zip(model_predictions, cols_in_query):
        predicted_cols = extract_from_completion_with("columns", model_pred)
        if predicted_cols is None:
            rewards.append(0.0)
            continue
        predicted_cols_set = {
            t.strip().lower().replace("`", "") for t in predicted_cols.split(",")
        }
        true_cols_set = {t.strip().lower() for t in cols}
        intersection = predicted_cols_set.intersection(true_cols_set)
        reward = len(intersection) / len(true_cols_set) if true_cols_set else 0.0
        rewards.append(reward)
    logger.info(
        f"[REWARD_COLS][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return rewards


def nl2sql_reward(
        completions: list[list[dict]],
        target_sql: list[str],
        db_id: list[str],
        evaluator: EvaluatorProtocol,
        relative_db_base_path: str,
        sql_execution_time: list[float],
        *args,
        **kwargs,
) -> list[float]:
    def add_or_log(values: list[OutputTable | Exception]) -> list[OutputTable]:
        output_ = []
        for val in values:
            if isinstance(val, OutputTable):
                output_.append(val)
                continue
            logger.info(val)
        return output_

    # run in parallel predictions and targets
    logger = get_logger("REWARD_SQLS")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    start_time = time.perf_counter()
    model_predictions = [_parse_model_response(val) for val in completions]

    target_sql_results, model_predictions_results = _execute_target_and_pred_sql(
        db_ids=db_id,
        target_sqls=target_sql,
        pred_sqls=model_predictions,
        timeout=sql_execution_time,
        relative_db_base_path=relative_db_base_path,
    )
    # make evaluation
    # in case of execution errors, the task will have empty predictions or targets
    # leading to 0 score for that example

    tasks = []
    for tars, preds in zip(target_sql_results, model_predictions_results):
        executed_preds, executed_tars = add_or_log(preds), add_or_log(tars)
        task = EvaluateTask(predictions=executed_preds, target=executed_tars)
        tasks.append(task)

    scores = evaluator.execute_metric(tasks=tasks, *args, **kwargs)

    logger.info(
        f"[REWARD_SQLS][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
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
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    pattern = re.compile(
        r"\s*<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*"
        r"<answer>\s*([\s\S]*?)\s*</answer>\s*\Z",  # \Z = end of string (ignores final \n issues)
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [pattern.fullmatch(content.strip()) for content in completion_contents]
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
    pattern = re.compile(
        r"\s*<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*"
        r"<tables>\s*([\s\S]*?)\s*</tables>\s*"
        r"<columns>\s*([\s\S]*?)\s*</columns>\s*"
        r"<checks>\s*([\s\S]*?)\s*</checks>\s*"
        r"<answer>\s*([\s\S]*?)\s*</answer>\s*\Z",  # \Z = end of string (ignores final \n issues)
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    completion_contents = [_parse_model_response(val) for val in completions]

    matches = [pattern.fullmatch(content.strip()) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def _execute_target_and_pred_sql(
        db_ids: list[str],
        target_sqls: list[str],
        pred_sqls: list[str] | list[list[str]],
        timeout: list[float],
        relative_db_base_path: str,
) -> tuple[list[list[OutputTable | ExecutorError]], list[list[OutputTable | ExecutorError]]]:
    if not isinstance(target_sqls, list) or not isinstance(pred_sqls, list) or not isinstance(db_ids, list):
        raise ValueError(
            "target_sqls and pred_sqls must be lists in _execute_target_and_pred_sql"
        )
    if isinstance(pred_sqls[0], str):
        pred_sqls = [[extract_sql_or_same(sql)] for sql in pred_sqls]
    elif isinstance(pred_sqls[0], list):
        pred_sqls = [[extract_sql_or_same(sql) for sql in sqls] for sqls in pred_sqls]
    else:
        raise ValueError(
            f"pred_sqls must be a list of strings or a list of list of strings in _execute_target_and_pred_sql instead of {type(pred_sqls[0])}"
        )

    db_files = [os.path.join(relative_db_base_path, id_, f"{id_}.sqlite") for id_ in db_ids] * 2
    timeout = timeout * 2
    timeout = [min(t + (0.20 * t) + 10, 500) for t in timeout]
    target_sqls = [[sql] for sql in target_sqls]

    query_to_execute = target_sqls + pred_sqls

    assert len(db_files) == len(query_to_execute) == len(timeout), (
        "Mismatched lengths in executing SQL queries. _execute_target_and_pred_sql"
    )

    # logger = get_logger("REWARD_SQLS")
    # for db_file, query, t in zip(db_files, query_to_execute, timeout):
    #     logger.info(f'{db_file}\ntimeout {t:.4f} seconds\nSQL Query:\n{extract_sql_or_same(query)}\n{"-" * 60}')

    num_cpus = min(_read_counter("AVAIL_CPU_PER_GPU"), max(1, len(query_to_execute)))
    tasks = [
        ExecuteTask(db_files=db_file, queries=query, timeout=t)
        for db_file, query, t in zip(db_files, query_to_execute, timeout)
    ]
    executed_queries = SQLiteDBExecutor().execute_queries(
        tasks=tasks,
        cache_db=SqliteCache(),
        cache_db_file=".nl2sql_cache/train_omnisql_cache.sqlite",
        num_cpus=num_cpus,
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


def get_reward_funcs(
        script_args: GRPOScriptArguments, save_in_cache=True
) -> list[Callable]:
    execution_accuracy_fn = partial(
        nl2sql_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        evaluator=BirdEXEvaluator(),
    )
    # This is to make sure the function name is correct in the logs and can be used with lighteval
    execution_accuracy_fn.__name__ = "execution_accuracy"
    qatch_reward_fn = partial(
        nl2sql_reward,
        relative_db_base_path=script_args.relative_db_base_path,
        evaluator=QATCHEvaluator(),
        metric="cp_cr_tc",
    )
    qatch_reward_fn.__name__ = "qatch_reward"
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
    return reward_funcs
