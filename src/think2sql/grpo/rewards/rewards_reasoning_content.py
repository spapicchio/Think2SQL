import re
import time
from collections import Counter

from transformers import AutoTokenizer

from think2sql.grpo.rewards.utils import (
    utils_parse_model_response,
    utils_extract_from_completion_with,
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


def penalty_not_english(completions, *, penalty=0.25, threshold=0.90, **kwargs):
    from langdetect import detect_langs, LangDetectException

    logger = get_logger("PENALTY-LANG")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger.info(f"[PENALTY-LANG][{hash_id}] Starting language penalty evaluation.")

    completions = [utils_parse_model_response(val) for val in completions]
    scores = []
    for text in completions:
        if not text or len(text.strip()) < 5:
            scores.append(0.0)  # Ignore very short segments or assume neutral
            continue

        # Pre-cleaning: Remove SQL code blocks for language detection
        # We only want to check the reasoning/text, not the SQL keywords
        text_without_code = re.sub(r"```sql.*?```", "", text, flags=re.DOTALL)
        try:
            # Detected language
            probs = detect_langs(text_without_code)
            # Check the top result
            top_lang = probs[0]
            if top_lang.lang == "en" and top_lang.prob >= threshold:
                scores.append(0.0)  # Good
            else:
                scores.append(-penalty)
        except LangDetectException:
            # If detection fails (e.g. text is just numbers), usually safe to ignore
            scores.append(0.0)
            logger.warning(f"Language detection failed for text: {text_without_code}")
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"[PENALTY-LANG][{hash_id}] Completed language penalty evaluation in {elapsed_time:.2f} seconds."
    )
    return scores


def penalty_repetitions(
    completion_ids, *, max_penalty=0.1, n_gram_size=5, threshold=0.5, **kwargs
):
    scores = []
    logger = get_logger("PENALTY-REPETITIONS")
    start_time = time.perf_counter()
    hash_id = hash(start_time)
    logger.info(
        f"[PENALTY-REPETITION][{hash_id}] Starting repetition penalty evaluation."
    )
    for seq in completion_ids:
        # 1. Convert tensor to list if necessary
        if hasattr(seq, "tolist"):
            seq = seq.tolist()

        # 2. Filter out padding (usually -100 or pad_token_id) if present
        # Assuming 0 is not a valid logic token or handled by tokenizer
        seq = [t for t in seq if t != -100]

        # 3. Guard clause for short sequences
        if len(seq) < n_gram_size:
            scores.append(0.0)
            continue

        # 4. Generate N-grams (tuples of integers)
        # e.g., [(101, 204, 305), (204, 305, 999)...]
        ngrams = [
            tuple(seq[i : i + n_gram_size]) for i in range(len(seq) - n_gram_size + 1)
        ]

        # 5. Calculate Ratio
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        # 6. Apply Penalty
        # If ratio is 0.4 (very repetitive), penalty is -0.6
        if ratio < threshold:
            scores.append(max(-max_penalty, ratio - 1.0))
        else:
            scores.append(0.0)
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"[PENALTY-REPETITION][{hash_id}] Completed repetition penalty evaluation in {elapsed_time:.2f} seconds."
    )
    return scores
