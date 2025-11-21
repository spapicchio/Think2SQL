import multiprocessing as mp
import os
import re
from typing import Any, Optional

import torch
from NL2SQLEvaluator.db_executor_nodes import SQLiteDBExecutor, SqliteCache
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, extract_sql_or_same, ExecuteTask

TRAINING_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = max(1, mp.cpu_count())
AVAIL_CPU_PER_GPU = AVAILABLE_CPUS // max(1, TRAINING_GPUS)


def utils_read_counter(name: str) -> Any:
    return globals().get(name, 0)


def utils_parse_model_response(completion: list[dict]) -> list[str] | str:
    # completion may be composed of different decoded sequences
    # most probably during training we will have only one decoded sequence
    pred: list[str] = [val["content"] for val in completion]
    return pred if len(pred) > 1 else pred[0]


def utils_execute_target_and_pred_sql(
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

    num_cpus = min(utils_read_counter("AVAIL_CPU_PER_GPU"), max(1, len(query_to_execute)))
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


def utils_extract_from_completion_with(tag: str, completion: str) -> str | None:
    pattern = rf"<{tag}>\n(.*?)\n</{tag}>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def utils_add_or_log(values: list[OutputTable | Exception], logger=None) -> list[OutputTable]:
    output_ = []
    for val in values:
        if isinstance(val, OutputTable):
            output_.append(val)
            continue
        if logger is not None:
            logger.info(val)
    return output_


def validate_response_structure_sql_r1(completion: str, think_tag, answer_tag) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        completion: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': (f'<{think_tag}>', 1),
        'think_end': (f'</{think_tag}>', 1),
        'answer_start': (f'<{answer_tag}>', 1),
        'answer_end': (f'</{answer_tag}>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = completion.count(tag_str)
        positions[tag_name] = completion.find(tag_str)

        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        validation_passed = False

    return validation_passed


def parse_sql_from_answer_sql_r1(answer_text: str) -> Optional[str]:
    """Parses SQL from the model's answer text.

    Args:
        answer_text: Text extracted from model's <answer> tags

    Returns:
        SQL string, or None if no SQL is found
    """
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))

    if not matches:
        return None
    return matches[-1].group(1).strip()
