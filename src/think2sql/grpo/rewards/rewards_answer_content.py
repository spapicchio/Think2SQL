import time

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import extract_sql_or_same, ExecutorError
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluatorProtocol, EvaluateTask

from think2sql.grpo.rewards.utils import (utils_parse_model_response,
                                          utils_execute_target_and_pred_sql,
                                          utils_extract_from_completion_with,
                                          utils_add_or_log,
                                          validate_response_structure_sql_r1)
from think2sql.logger import get_logger


def reward_sql_r1(completions: list[list[dict]],
                  target_sql: list[str],
                  db_id: list[str],
                  sql_execution_time: list[float],
                  *,
                  evaluator: EvaluatorProtocol,
                  relative_db_base_path: str,
                  **kwargs,
                  ) -> list[float]:
    # The reward goes from 0 to 6
    # The code is taken from https://github.com/DataArcTech/SQL-R1/blob/d3645cc72820b27ed09fdda65f1bb6494c80b37e/verl/utils/reward_score/synsql.py#L118
    logger = get_logger("REWARD_SQL-R1")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    start_time = time.perf_counter()
    model_predictions = [utils_parse_model_response(val) for val in completions]

    FORMAT_REWARD = 1
    EXEC_REWARD = 2
    RESULT_REWARD = 3
    LIMIT_LENGTH = 2048

    target_sql_results, model_predictions_results = utils_execute_target_and_pred_sql(
        db_ids=db_id,
        target_sqls=target_sql,
        pred_sqls=model_predictions,
        timeout=sql_execution_time,
        relative_db_base_path=relative_db_base_path,
    )

    scores = []
    for completion, exec_pred, exec_target in zip(model_predictions, model_predictions_results, target_sql_results):
        think_in_completion = utils_extract_from_completion_with('reasoning', completion)
        answer_in_completion = utils_extract_from_completion_with('answer', completion)

        is_format_correct = validate_response_structure_sql_r1(
            completion,
            think_tag='reasoning',
            answer_tag='answer'
        )
        pred_sql = extract_sql_or_same(completion)
        format_score = FORMAT_REWARD if is_format_correct else -abs(FORMAT_REWARD)
        exec_score = 0
        result_score = 0
        if is_format_correct:
            if isinstance(exec_pred[0], ExecutorError) or isinstance(exec_target[0], ExecutorError):
                if isinstance(exec_target[0], ExecutorError):
                    logger.warning(f"Target SQL execution error: {exec_target}")
                exec_score = -abs(EXEC_REWARD)
                result_score = 0
            else:
                # now we need to calculate the EX metric
                executed_preds, executed_tars = utils_add_or_log(exec_pred, logger), utils_add_or_log(exec_target,
                                                                                                      logger)
                task = EvaluateTask(predictions=executed_preds, target=executed_tars)
                ex_accuracy = evaluator.execute_metric(tasks=[task], **kwargs)[0]
                exec_score = EXEC_REWARD
                result_score = RESULT_REWARD if ex_accuracy > 0 else -abs(RESULT_REWARD)

        length_score = 0
        if result_score > 0:
            pos_length = len(think_in_completion) + len(answer_in_completion)
            if pos_length <= LIMIT_LENGTH:
                sql_in_answer_sub_score = len(pred_sql) / len(answer_in_completion)
                length_sub_score = pos_length / LIMIT_LENGTH * 0.5
                length_score = length_sub_score + sql_in_answer_sub_score
            else:
                sql_in_answer_sub_score = len(pred_sql) / len(answer_in_completion)
                length_score = 0.5 + sql_in_answer_sub_score
        final_score = format_score + exec_score + result_score + length_score
        scores.append(final_score)

    logger.info(
        f"[REWARD_SQL-R1][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


def reward_arctic_sql(completions: list[list[dict]],
                      target_sql: list[str],
                      db_id: list[str],
                      sql_execution_time: list[float],
                      *
                      evaluator: EvaluatorProtocol,
                      relative_db_base_path: str,
                      **kwargs,
                      ) -> list[float]:
    logger = get_logger("REWARD-Arctic-sql")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    start_time = time.perf_counter()
    model_predictions = [utils_parse_model_response(val) for val in completions]
    target_sql_results, model_predictions_results = utils_execute_target_and_pred_sql(
        db_ids=db_id,
        target_sqls=target_sql,
        pred_sqls=model_predictions,
        timeout=sql_execution_time,
        relative_db_base_path=relative_db_base_path,
    )
    scores = []
    for completion, exec_pred, exec_target in zip(model_predictions, model_predictions_results, target_sql_results):
        if isinstance(exec_pred, Exception):
            scores.append(0)
            continue
        task = EvaluateTask(predictions=exec_pred, target=exec_target)
        ex_accuracy = evaluator.execute_metric(tasks=[task], **kwargs)[0]
        scores.append(1 if ex_accuracy > 0 else 0.1)
    logger.info(
        f"[REWARD-Arctic-sql][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


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
    # run in parallel predictions and targets
    logger = get_logger("REWARD_SQLS")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    start_time = time.perf_counter()
    model_predictions = [utils_parse_model_response(val) for val in completions]

    target_sql_results, model_predictions_results = utils_execute_target_and_pred_sql(
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
        executed_preds, executed_tars = utils_add_or_log(preds, logger), utils_add_or_log(tars, logger)
        task = EvaluateTask(predictions=executed_preds, target=executed_tars)
        tasks.append(task)

    scores = evaluator.execute_metric(tasks=tasks, *args, **kwargs)

    logger.info(
        f"[REWARD_SQLS][END][{hash_id}] Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores
