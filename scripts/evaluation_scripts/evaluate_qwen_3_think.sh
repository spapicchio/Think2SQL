#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

set -Eeuo pipefail

# --- env & utils ---

export JOB_ID=${SLURM_JOB_ID:-MY_SLURM_JOB_ID}


if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Running inside SLURM job ${SLURM_JOB_ID}"
  export BASE_WORK="${SCRATCH}/Think2SQL"
  cd $BASE_WORK
  export HF_HOME="${SCRATCH}/hf_cache"
  source "${BASE_WORK}/scripts/utils/slurm_job_requeue.sh"
  uv sync --frozen
  source "${BASE_WORK}/.venv/bin/activate"
fi

source "${BASE_WORK}/.env"
source "${BASE_WORK}/scripts/utils/utils.sh"


# MODEL_NAME='Qwen/Qwen3-0.6B'
# MODEL_NAME='Qwen/Qwen3-1.7B'
# MODEL_NAME='Qwen/Qwen3-4B-Thinking-2507'
# MODEL_NAME='Qwen/Qwen3-8B'
# MODEL_NAME='Qwen/Qwen3-4B'
MODEL_NAME='Qwen/Qwen3-14B'

MAX_NEW_TOKENS=4096

USER_PROMPT_NAME="omnisql_user_prompt.jinja"
SYSTEM_PROMPT_NAME=''

ENABLE_THINKING_MODE='true'
RUN_ONLY_PREDICTIONS=true

# CUDA_VISIBLE_DEVICES='0,1,2,3' \
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
ENABLE_THINKING_MODE=$ENABLE_THINKING_MODE \
RUN_ONLY_PREDICTIONS=$RUN_ONLY_PREDICTIONS \
GREEDY_TEMP="0.6" \
GREEDY_TOP_P="0.95" \
GREEDY_TOP_K="20" \
MV_TEMP="1.0" \
MV_TOP_P="0.95" \
MV_TOP_K="20" \
${BASE_WORK}/scripts/evaluate.sh
