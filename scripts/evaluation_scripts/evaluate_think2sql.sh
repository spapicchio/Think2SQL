#!/bin/bash
#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=01:00:00
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

# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-4B-Instruct-2507/dapo/bs256_ml4096_gen8_qwen-think4-b934d9d7_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-b304a262_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-cf82256c_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-18facd9a_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-c5cdacfe_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/grpo/Qwen3-1_7B/dapo/bs256_ml8092_gen16_qwen-think4-fe7cde7d_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_e49038d5_RL"
# MODEL_NAME="${BASE_WORK}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_644bec77_RL"

# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_212527_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_212596_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_212534_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_236062_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_236063_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_351542_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_351394_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRnone_IStoken_236069_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_360536_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_710504_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_728108_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_710512_RL"
MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_710457_RL"
#MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_845475_RL"
#MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_845599_RL"
#MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_902768_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRnone_IStoken_898655_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_925842_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_927224_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-06B/TMFalse_ml4096_SRbatch_IStoken_956059_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-1_7B/TMFalse_ml4096_SRbatch_IStoken_956053_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_957475_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRnone_IStoken_957501_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_966642_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_972879_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_972879_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRgroup_IStoken_983327_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRgroup_IStoken_984425_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_984625_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRgroup_IStoken_1052560_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRnone_IStoken_1052820_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRbatch_IStoken_1049743_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRnone_IStoken_1049954_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRgroup_IStoken_1049767_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRbatch_IStoken_1052818_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRgroup_IStoken_1052555_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRbatch_IStoken_1052556_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRnone_IStoken_1052559_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRbatch_IStoken_1052551_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRnone_IStoken_1049949_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-14B/TMFalse_ml4096_SRnone_IStoken_1052549_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRgroup_IStoken_1049962_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRgroup_IStoken_1120263_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRbatch_IStoken_1120266_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-8B/TMFalse_ml4096_SRnone_IStoken_1120269_RL"
# MODEL_NAME="${SCRATCH}/Think2SQL/model_trained/SFT/Qwen3-4B/TMFalse_ml8000_1143264_SFT"
# MODEL_NAME="${SCRATCH}/Think2SQL/model_trained/SFT/Qwen3-4B/TMFalse_ml8000_1157005_SFT"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_1155769_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_1155784_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_1155766_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B-SFT-1143264/TMFalse_ml4096_SRgroup_IStoken_1155812_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_1155763_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_957475_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_957475_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_845599_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRgroup_IStoken_728108_RL"
# MODEL_NAME="${SCRATCH}/model_trained/dapo/Qwen3-4B/TMFalse_ml4096_SRbatch_IStoken_710504_RL"


MAX_NEW_TOKENS=4096

SYSTEM_PROMPT_NAME='base_think_system_prompt.jinja'
USER_PROMPT_NAME='base_think_user_prompt.jinja'
# SYSTEM_PROMPT_NAME="no_tag_system_prompt.jinja"
# USER_PROMPT_NAME="no_tag_user_prompt.jinja"

ENABLE_THINKING_MODE=''
RUN_ONLY_PREDICTIONS=true

# CUDA_VISIBLE_DEVICES='6,7' \
MODEL_NAME=$MODEL_NAME \
MAX_NEW_TOKENS=$MAX_NEW_TOKENS \
USER_PROMPT_NAME=$USER_PROMPT_NAME \
SYSTEM_PROMPT_NAME=$SYSTEM_PROMPT_NAME \
ENABLE_THINKING_MODE=$ENABLE_THINKING_MODE \
RUN_ONLY_PREDICTIONS=$RUN_ONLY_PREDICTIONS \
GREEDY_TEMP="0.6" \
GREEDY_TOP_P="0.95" \
GREEDY_TOP_K="20" \
${BASE_WORK}/scripts/evaluate.sh
