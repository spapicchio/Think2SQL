#!/bin/bash
#SBATCH --job-name=eval-Qwen25-coder-0-5

# --- robust shell settings ---
set -Eeuo pipefail

# --- GPU selection (override per call if needed) ---
# MODEL_NAME='Qwen/Qwen2.5-Coder-0.5B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-1.5B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-3B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-7B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-14B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-32B-Instruct'

# OmniSQL
#MODEL_NAME='seeklhy/OmniSQL-7B'
MODEL_NAME='seeklhy/OmniSQL-14B'
#MODEL_NAME='seeklhy/OmniSQL-32B'

MAX_NEW_TOKENS=2048

USER_PROMPT_NAME="omnisql_user_prompt.jinja"
SYSTEM_PROMPT_NAME=''


CUDA_VISIBLE_DEVICES='4,6' \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
GREEDY_TEMP="0.0" \
GREEDY_TOP_P="1" \
GREEDY_TOP_K="0" \
MV_TEMP="1.0" \
MV_TOP_P="0.95" \
MV_TOP_K="20" \
${BASE_WORK}/scripts/evaluate.sh