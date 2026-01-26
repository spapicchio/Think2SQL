#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --output=./logs/rl/%j.out
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# NUM_GPUS=8
NUM_GPUS=$(($SLURM_NNODES * SLURM_GPUS_PER_NODE))

set -Eeuo pipefail

module purge
module load arch/a100
module load cuda/12.4.1

nvidia-smi

export BASE_WORK="${SCRATCH}/Think2SQL"
export BASE_WORK_MODEL="${SCRATCH}/model_trained"
export BASE_WORK_DATA="${SCRATCH}/data"

cd $BASE_WORK
export HF_HOME="${SCRATCH}/hf_cache"


source "${BASE_WORK}/.env"
source "${BASE_WORK}/scripts/utils/utils.sh"
source "${BASE_WORK}/scripts/utils/utils_clenup_vllm_if_crash.sh"

export GIT_PYTHON_REFRESH=quiet
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline


uv sync --frozen
source "${BASE_WORK}/.venv/bin/activate"

log_section "Environment and modules loaded with BASE_WORK=${BASE_WORK}" "${SLURM_JOB_ID}"

# ----------- JOB ID of the run -----------
# if you are relaunching a job and want to keep the same folders for logging and continue training
#JOB_ID='<your id>'
#export WANDB_RUN_ID='5wxzzpvx'
JOB_ID=$SLURM_JOB_ID

log_section "JOB_ID = ${JOB_ID}" "${JOB_ID}"

# ----------- Configuration -----------
export OMP_NUM_THREADS=50
export WANDB_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export WANDB_ARTIFACT_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export TOKENIZERS_PARALLELISM=True
LOGGING_DIR_TENSORBOARD="${BASE_WORK}/.tensorboard_logging/${JOB_ID}/"

# ----------- Custom  Params -----------
PROMPT_FOLDER="${BASE_WORK}/prompts"
# SYSTEM_PROMPT_NAME="base_think_system_prompt.jinja"
# USER_PROMPT_NAME="base_think_user_prompt.jinja"
SYSTEM_PROMPT_NAME=""
USER_PROMPT_NAME="omnisql_user_prompt.jinja"

# ----------- Dataset Params -----------
DATASET_NAME="${BASE_WORK}/gemini/gemini_collected_dataset/gemini-3-flash-preview/bird_train/sft_df.json"
RESP_COL_NAME='SQL'
# ----------- Training Params -----------
LEARNING_RATE=4e-5

MODEL_BASE='Qwen3-4B'
MODEL_BASE_PATH="Qwen/Qwen3-4B"

ENABLE_THINKING_MODE='False'
MAX_LENGTH=8000
EVAL_BATCH_SIZE=1
TRAIN_BATCH_SIZE=2
ACCUMULATION_STEPS=16
TOTAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))


SFT_MODEL_NAME="TM${ENABLE_THINKING_MODE}_ml${MAX_LENGTH}_${JOB_ID}_SFT"
echo "SFT_MODEL_NAME: ${SFT_MODEL_NAME}"

OUTPUT_DIR="${BASE_WORK}/model_trained/SFT/${MODEL_BASE}/${SFT_MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

LAUNCHER=(
    accelerate launch
    --config_file "${BASE_WORK}/config/accelerate_config_sft.yaml"
    --num_processes $NUM_GPUS
    "${BASE_WORK}/src/think2sql/sft/main_sft.py"
    --config "${BASE_WORK}/config/config_train_sft.yaml"
    --model_name_or_path $MODEL_BASE_PATH
    --output_dir "$OUTPUT_DIR"
    --dataset_name ${DATASET_NAME}
    --assistant_response_col_name $RESP_COL_NAME
    --max_length $MAX_LENGTH
    --learning_rate $LEARNING_RATE
    --logging_dir $LOGGING_DIR_TENSORBOARD
    --save_total_limit 2
    --per_device_eval_batch_size $EVAL_BATCH_SIZE
    --per_device_train_batch_size $TRAIN_BATCH_SIZE
    --gradient_accumulation_steps $ACCUMULATION_STEPS
)

log_section "Script: ${LAUNCHER[*]}" "${JOB_ID}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
NCCL_P2P_LEVEL=NVL \
"${LAUNCHER[@]}"