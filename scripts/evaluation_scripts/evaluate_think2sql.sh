#!/bin/bash
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


# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-4B-Instruct-2507/dapo/bs256_ml4096_gen8_qwen-think4-b934d9d7_RL"
MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-b304a262_RL"

MAX_NEW_TOKENS=8096

USER_PROMPT_NAME='base_think_user_prompt.jinja'
SYSTEM_PROMPT_NAME='base_think_system_prompt.jinja'

ENABLE_THINKING_MODE=''

CUDA_VISIBLE_DEVICES='4,6' \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
ENABLE_THINKING_MODE=$ENABLE_THINKING_MODE \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
GREEDY_TEMP="0.6" \
GREEDY_TOP_P="0.95" \
GREEDY_TOP_K="20" \
MV_TEMP="1.0" \
MV_TOP_P="0.95" \
MV_TOP_K="20" \
${BASE_WORK}/scripts/evaluate.sh
