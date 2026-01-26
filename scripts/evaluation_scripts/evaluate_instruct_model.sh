#!/bin/bash
#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

# --- robust shell settings ---
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

# OmniSQL
# MODEL_NAME='seeklhy/OmniSQL-7B'
# MODEL_NAME='seeklhy/OmniSQL-14B'
# MODEL_NAME='seeklhy/OmniSQL-32B'


# MODEL_NAME='Qwen/Qwen2.5-Coder-3B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-7B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-14B-Instruct'
# MODEL_NAME='Qwen/Qwen2.5-Coder-32B-Instruct'

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"

MAX_NEW_TOKENS=4096

SYSTEM_PROMPT_NAME=''
USER_PROMPT_NAME="omnisql_user_prompt.jinja"

ENABLE_THINKING_MODE=''
RUN_ONLY_PREDICTIONS=true

# CUDA_VISIBLE_DEVICES='4,6' \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
ENABLE_THINKING_MODE=$ENABLE_THINKING_MODE \
RUN_ONLY_PREDICTIONS=$RUN_ONLY_PREDICTIONS \
GREEDY_TEMP="0.0" \
GREEDY_TOP_P="1" \
GREEDY_TOP_K="-1" \
${BASE_WORK}/scripts/evaluate.sh