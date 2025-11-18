from functools import partial
from typing import Callable

from NL2SQLEvaluator.evaluator_nodes import BirdEXEvaluator
from NL2SQLEvaluator.evaluator_nodes.qatch_metrics import QATCHEvaluator

from think2sql.configs import GRPOScriptArguments
from think2sql.grpo.rewards.rewards_answer_content import nl2sql_reward, reward_sql_r1
from think2sql.grpo.rewards.rewards_reasoning_content import format_reward, tag_count_reward, multi_tag_format_reward, \
    reward_selected_tables, reward_selected_columns


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
        "SQL-R1": reward_sql_r1,
    }

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    return reward_funcs
