#!/bin/bash
#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=03:00:00
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

# MODEL_NAME='DeepRetrieval/DeepRetrieval-SQL-3B'
MODEL_NAME='DeepRetrieval/DeepRetrieval-SQL-7B'


MAX_NEW_TOKENS=2048

SYSTEM_PROMPT_NAME=''
USER_PROMPT_NAME="deepretrieval_user_prompt.jinja"

ENABLE_THINKING_MODE=''
RUN_ONLY_PREDICTIONS=true

# CUDA_VISIBLE_DEVICES='5' \
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
ENABLE_THINKING_MODE=$ENABLE_THINKING_MODE \
RUN_ONLY_PREDICTIONS=$RUN_ONLY_PREDICTIONS \
GREEDY_TEMP="0.6" \
GREEDY_TOP_P="0.8" \
GREEDY_TOP_K="20" \
${BASE_WORK}/scripts/evaluate.sh
