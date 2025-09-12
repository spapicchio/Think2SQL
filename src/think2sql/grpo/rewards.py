import re
from statistics import mean
from typing import Callable

from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import OrchestratorInput
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import (
    evaluator_orchestrator,
)
from NL2SQLEvaluator.orchestrator_state import AvailableMetrics, AvailableDialect

from think2sql.configs import GRPOScriptArguments
from think2sql.logger import get_logger
from think2sql.utils.data import utils_get_engine
from think2sql.utils.sql import get_sql_from_generation


def _check_crud_sql_(query):
    if (
            "INSERT" in str(query).upper()
            or "UPDATE" in str(query).upper()
            or "DELETE" in str(query).upper()
            or "CREATE" in str(query).upper()
            or "DROP" in str(query).upper()
            or "ALTER" in str(query).upper()
    ):
        return "WRONG_QUERY"
    return query


def get_execution_acc_reward(relative_db_base_path):
    def ex_reward(
            completions: list[dict],
            target_sql: list[str],
            db_id: list[str],
            *args,
            **kwargs,
    ):
        model_prediction = [completion[0]["content"] for completion in completions]
        predicted_sqls = [
            _check_crud_sql_(get_sql_from_generation(pred)) for pred in model_prediction
        ]

        orchestrator_input = OrchestratorInput(
            target_queries=target_sql,
            predicted_queries=predicted_sqls,
            executor=[
                utils_get_engine(relative_db_base_path, AvailableDialect("sqlite"), id_)
                for id_ in db_id
            ],
            metrics_to_calculate=[
                AvailableMetrics.EXECUTION_ACCURACY,
            ],
        )
        results = evaluator_orchestrator.invoke(orchestrator_input)
        results = [
            result[AvailableMetrics.EXECUTION_ACCURACY.value] for result in results
        ]
        return results

    return ex_reward


def get_qatch_reward(relative_db_base_path):
    def qatch_reward(
            completions: list[dict],
            target_sql: list[str],
            db_id: list[str],
            *args,
            **kwargs,
    ):
        model_prediction = [completion[0]["content"] for completion in completions]
        predicted_sqls = [
            _check_crud_sql_(get_sql_from_generation(pred)) for pred in model_prediction
        ]
        orchestrator_input = OrchestratorInput(
            target_queries=target_sql,
            predicted_queries=predicted_sqls,
            executor=[
                utils_get_engine(relative_db_base_path, AvailableDialect("sqlite"), id_)
                for id_ in db_id
            ],
            metrics_to_calculate=[
                AvailableMetrics.CELL_PRECISION,
                AvailableMetrics.CELL_RECALL,
                AvailableMetrics.TUPLE_CARDINALITY,
            ],
        )
        results = evaluator_orchestrator.invoke(orchestrator_input)
        results = [
            mean(list(result.values())) for result in results
        ]
        return results

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
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
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
    logger.info(f"loaded reward functions: {script_args.reward_funcs}")
    return reward_funcs
