import time
from statistics import mean, stdev

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import extract_sql_or_same, ExecutorError
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluatorProtocol, EvaluateTask

from think2sql.grpo.rewards.rewards_reasoning_content import format_reward
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
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger = get_logger(f"REWARD_SQL-R1-{hash_id}")
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
        think_in_completion = utils_extract_from_completion_with('reasoning', completion) or completion
        answer_in_completion = utils_extract_from_completion_with('answer', completion) or completion

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
        f"Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


def reward_arctic_sql(completions: list[list[dict]],
                      target_sql: list[str],
                      db_id: list[str],
                      sql_execution_time: list[float],
                      *,
                      evaluator: EvaluatorProtocol,
                      relative_db_base_path: str,
                      **kwargs,
                      ) -> list[float]:
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger = get_logger(f"REWARD-Arctic-sql-{hash_id}")
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
        think_in_completion = utils_extract_from_completion_with('reasoning', completion)
        answer_in_completion = utils_extract_from_completion_with('answer', completion)
        if isinstance(exec_pred[0], ExecutorError) or isinstance(exec_target[0], ExecutorError):
            # The prediction contains an error
            scores.append(0)
            continue
        task = EvaluateTask(predictions=exec_pred, target=exec_target)
        ex_accuracy = evaluator.execute_metric(tasks=[task], **kwargs)[0]
        if ex_accuracy > 0:
            # If target and prediction matches then it is 1
            scores.append(1)
        elif think_in_completion is not None and answer_in_completion is not None:
            # If both reasoning and answer are present and the pred is executable then give small reward
            scores.append(0.1)
        else:
            scores.append(0)
    logger.info(
        f"Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


def reward_small_update(completions: list[list[dict]],
                        target_sql: list[str],
                        db_id: list[str],
                        sql_execution_time: list[float],
                        *,
                        evaluator: EvaluatorProtocol,
                        relative_db_base_path: str,
                        include_format_reward: bool = False,
                        is_discrete_buckets: bool = False,
                        **kwargs,
                        ) -> list[float]:
    """
    The reward is 0 only when the execution fails, 0 is given to strongly discourage
    invalid SQL queries. If the execution is successful, the reward is the metric chosen to evaluate the SQL.

    In case of Execution accuracy, the reward is 0.1 for valid execution and 1.0 for correct execution.
    In case of QATCH or other metrics, the reward is the metric value if greater than 0.1, otherwise 0.1 for valid execution.
    """
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger = get_logger(f"REWARD-SU-{hash_id}")
    model_predictions = [utils_parse_model_response(val) for val in completions]
    target_sql_results, model_predictions_results = utils_execute_target_and_pred_sql(
        db_ids=db_id,
        target_sqls=target_sql,
        pred_sqls=model_predictions,
        timeout=sql_execution_time,
        relative_db_base_path=relative_db_base_path,
    )
    scores = []
    matches = [1] * len(completions) if not include_format_reward else format_reward(completions, reasoning_tag="think", **kwargs)
    for completion, exec_pred, exec_target, match in zip(model_predictions, model_predictions_results,
                                                         target_sql_results, matches):
        if isinstance(exec_pred[0], ExecutorError) or isinstance(exec_target[0], ExecutorError):
            # The prediction contains an error
            scores.append(0)
            continue
        task = EvaluateTask(predictions=exec_pred, target=exec_target)
        score = evaluator.execute_metric(tasks=[task], **kwargs)[0]
        # if the score is greater than 0.1, keep it, otherwise give a small reward of 0.1 for having a valid execution
        if score == 1:
            scores.append(score)
        elif match:
            if is_discrete_buckets:
                # Discrete buckets: 0.1, 0.3, 0.5, 0.7, 0.9
                if 0.0 <= score < 0.1:
                    scores.append(0.1)
                if 0.1 <= score < 0.5:
                    scores.append(0.5)
                if 0.5 <= score < 0.8:
                    scores.append(0.8)
                if 0.8 <= score < 1.0:
                    scores.append(0.9)
            else:
                scores.append(score if score > 0.1 else 0.1)
        else:
            scores.append(0.0)

    logger.info(
        f"Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


def complex_reward(completions: list[list[dict]],
                   target_sql: list[str],
                   db_id: list[str],
                   sql_execution_time: list[float],
                   *,
                   evaluator: EvaluatorProtocol,
                   relative_db_base_path: str,
                   num_of_generations: int,
                   **kwargs,
                   ) -> list[float]:
    """Each reward gets B * NUM_GENERATIONS completions, where B is the batch for each GPU without gradient accumulation:"""
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger = get_logger(f"REWARD-Len-Penalty-{hash_id}")
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
        if isinstance(exec_pred[0], ExecutorError) or isinstance(exec_target[0], ExecutorError):
            # The prediction contains an error
            scores.append(0)
            continue
        task = EvaluateTask(predictions=exec_pred, target=exec_target)
        score = evaluator.execute_metric(tasks=[task], **kwargs)[0]
        # if the score is greater than 0.1, keep it, otherwise give a small reward of 0.1 for having a valid execution
        # scores.append(score if score > 0.1 else 0.1)
        scores.append(score)

    scores = apply_dynamic_len_penalty(model_predictions, scores, num_of_generations, penalty_strength=0.1)

    logger.info(
        f"Completed in {time.perf_counter() - start_time:.2f} seconds"
    )
    return scores


def apply_dynamic_len_penalty(
        completions: list[str],
        scores: list[float],
        num_generations: int,
        penalty_strength: float = 0.1,
        correctness_threshold: float = 0.5,
        min_std_floor: float = 0.2
) -> list[float]:
    """Applies a length penalty based on the Z-score of the generation length."""
    if len(completions) != len(scores):
        raise ValueError("Completions and scores must have the same length")

    num_prompts = len(completions) // num_generations
    new_scores = scores.copy()

    for i in range(num_prompts):
        start_idx = i * num_generations
        end_idx = start_idx + num_generations

        batch_completions = completions[start_idx:end_idx]
        batch_scores = scores[start_idx:end_idx]

        # 1. Filter for correct generations
        correct_indices = [
            idx for idx, s in enumerate(batch_scores)
            if s > correctness_threshold
        ]
        correct_lengths = [len(batch_completions[idx]) for idx in correct_indices]

        # 2. Compute Baseline Statistics
        if len(correct_lengths) < 2:
            # No baseline can be established; skip penalty
            continue

        avg_length = mean(correct_lengths)

        current_std = stdev(correct_lengths)
        # Use a floor for std_dev to allow for natural whitespace variance
        # without triggering massive Z-scores.
        # we allow a standard deviation of at least `min_std_floor` of the average length to not strongly penalize small variations in language
        std_dev = max(current_std, avg_length * min_std_floor)

        # 3. Apply Penalty
        applied_penaties = []
        for j in range(num_generations):
            global_idx = start_idx + j

            # Only penalize positive scores
            if batch_scores[j] > 0:
                current_len = len(batch_completions[j])
                z_score = (current_len - avg_length) / std_dev

                # Only punish positive Z-scores (longer than avg)
                if z_score > 0:
                    length_penalty = z_score * penalty_strength

                    # Safety clamp: Ensure we don't negative-score a good answer
                    # and limit max penalty to 0.9
                    length_penalty = min(0.9, length_penalty)
                    applied_penaties.append(length_penalty)
                    new_val = new_scores[global_idx] - length_penalty
                    # Ensure score doesn't drop below 0
                    new_scores[global_idx] = max(0.0, new_val)

        logger = get_logger("REWARD-Len-Penalty")
        logger.info(
            f"[REWARD-Len-Penalty][Prompt {i}] Avg Length: {avg_length:.2f}, Std Dev: {std_dev:.2f}, mean penalties applied: {mean(applied_penaties) if applied_penaties else 0:.4f}"
        )

    return new_scores


def nl2sql_reward(
        completions: list[list[dict]],
        target_sql: list[str],
        db_id: list[str],
        evaluator: EvaluatorProtocol,
        relative_db_base_path: str,
        sql_execution_time: list[float],
        log: bool = True,
        *args,
        **kwargs,
) -> list[float]:
    # run in parallel predictions and targets
    logger = None
    if log:
        start_time = time.perf_counter()
        hash_id = hash(start_time)
        logger = get_logger(f"REWARD_SQLS-{hash_id}")

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
    if log:
        logger.info(
            f"Completed in {time.perf_counter() - start_time:.2f} seconds"
        )
    return scores
