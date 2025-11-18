import re
import time

from think2sql.grpo.rewards.utils import (
    utils_parse_model_response,
    utils_extract_from_completion_with
)
from think2sql.logger import get_logger


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
    model_predictions = [utils_parse_model_response(val) for val in completions]

    assert len(model_predictions) == len(tbls_in_query), (
        "Length mismatch between completions and tables in query"
    )

    rewards = []
    for model_pred, tbls in zip(model_predictions, tbls_in_query):
        predicted_tbls = utils_extract_from_completion_with("tables", model_pred)
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
    model_predictions = [utils_parse_model_response(val) for val in completions]
    assert len(model_predictions) == len(cols_in_query), (
        "Length mismatch between completions and columns in query"
    )
    rewards = []
    for model_pred, cols in zip(model_predictions, cols_in_query):
        predicted_cols = utils_extract_from_completion_with("columns", model_pred)
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
    completion_contents = [utils_parse_model_response(val) for val in completions]

    matches = [pattern.fullmatch(content.strip()) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
