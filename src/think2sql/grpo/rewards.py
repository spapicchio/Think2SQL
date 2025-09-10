import re
from typing import Callable

import NL2SQLEvaluator.main_run_evaluation as sql_evaluator
from NL2SQLEvaluator.orchestrator_state import AvailableMetrics

from think2sql.configs import GRPOScriptArguments
from think2sql.logger import get_logger
from think2sql.utils.sql import get_sql_from_generation


def get_execution_acc_reward(relative_db_base_path):
    def ex_reward(completions: list[dict], target_sql: list[str], db_id: list[str], *args, **kwargs):
        model_prediction = [completion[0]["content"] for completion in completions]
        predicted_sqls = [get_sql_from_generation(pred) for pred in model_prediction]
        data = [
            sql_evaluator.DatasetRow(target_query=sql, predicted_query=pred, db_id=db_id_)
            for sql, pred, db_id_ in zip(target_sql, predicted_sqls, db_id)
        ]
        script_args_evaluator = sql_evaluator.ScriptArgs(
            metrics=[AvailableMetrics.EXECUTION_ACCURACY],
            relative_db_base_path=relative_db_base_path,
            dataset=data,
            column_name_target='target_query',
            column_name_predicted='predicted_query',
            column_name_db_id='db_id',
        )

        _, df_completed_tasks = sql_evaluator.run_evaluation(script_args_evaluator)
        return df_completed_tasks['execution_accuracy'].tolist()

    return ex_reward


def get_qatch_reward(relative_db_base_path):
    def qatch_reward(completions: list[dict], target_sql: list[str], db_id: list[str], *args, **kwargs):
        model_prediction = [completion[0]["content"] for completion in completions]
        predicted_sqls = [get_sql_from_generation(pred) for pred in model_prediction]
        data = [
            sql_evaluator.DatasetRow(target_query=sql, predicted_query=pred, db_id=db_id_)
            for sql, pred, db_id_ in zip(target_sql, predicted_sqls, db_id)
        ]
        script_args_evaluator = sql_evaluator.ScriptArgs(
            metrics=[AvailableMetrics.CELL_PRECISION,
                     AvailableMetrics.CELL_RECALL,
                     AvailableMetrics.TUPLE_CARDINALITY],
            relative_db_base_path=relative_db_base_path,
            dataset=data,
            column_name_target='target_query',
            column_name_predicted='predicted_query',
            column_name_db_id='db_id',
        )

        _, df_completed_tasks = sql_evaluator.run_evaluation(script_args_evaluator)
        # make a third column with the mean of the three metrics above
        df_completed_tasks['qatch_metric'] = df_completed_tasks[
            ['cell_precision', 'cell_recall', 'tuple_cardinality']
        ].mean(axis=1)

        return df_completed_tasks['qatch_metric'].tolist()

    return qatch_reward


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
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def get_reward_funcs(script_args: GRPOScriptArguments) -> list[Callable]:
    logger = get_logger(__name__)

    REWARD_FUNCS_REGISTRY = {
        "EX": get_execution_acc_reward(script_args.relative_db_base_path),
        "QATCH": get_qatch_reward(script_args.relative_db_base_path),
        "format": format_reward,
        "tag_count": tag_count_reward,
    }

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    logger.info(f'loaded reward functions: {script_args.reward_funcs}')
    return reward_funcs
