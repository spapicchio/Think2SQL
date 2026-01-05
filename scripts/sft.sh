#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3'
NUM_GPUS=4

# --- robust shell settings ---
set -Eeuo pipefail

source "${BASE_WORK}/.env"
source "${BASE_WORK}/scripts/utils/utils.sh"
source "${BASE_WORK}/scripts/utils/utils_clenup_vllm_if_crash.sh"

# If one job crash and you want to start from it again,
# set the JOB_ID to the one you want to resume from
# JOB_ID='aba0bebc'
# export WANDB_RUN_ID='mrondefv'

JOB_ID=${MY_SLURM_JOB_ID}

log_section "JOB_ID = ${JOB_ID}" "${JOB_ID}"

# ----------- Configuration -----------
export OMP_NUM_THREADS=50
export WANDB_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export WANDB_ARTIFACT_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export TOKENIZERS_PARALLELISM=True
LOGGING_DIR_TENSORBOARD="${BASE_WORK}/.tensorboard_logging/${JOB_ID}/"

# ----------- Custom  Params -----------
PROMPT_FOLDER="${BASE_WORK}/prompts"
SYSTEM_PROMPT_NAME="base_think_system_prompt.jinja"
USER_PROMPT_NAME="base_think_user_prompt.jinja"

# ----------- Dataset Params -----------
DATASET_NAME="${BASE_WORK}/gemini/gemini_collected_dataset/gemini-3-flash-preview/bird_train/sft_df.json"
RESP_COL_NAME='gemini-3-flash-preview'
# ----------- Training Params -----------
LEARNING_RATE=4e-5

MODEL_BASE='Qwen3-4B'
MODEL_BASE_PATH="Qwen/Qwen3-4B"

ENABLE_THINKING_MODE='False'
MAX_LENGTH=8000

SFT_MODEL_NAME="TM${ENABLE_THINKING_MODE}_ml${MAX_LENGTH}_${JOB_ID}_SFT"
echo "SFT_MODEL_NAME: ${SFT_MODEL_NAME}"

OUTPUT_DIR="${BASE_WORK}/model_trained/SFT/${MODEL_BASE}/${SFT_MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

LAUNCHER=(
    accelerate launch
    --config_file "config/accelerate_config_sft.yaml"
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
    --run_name "${JOB_NAME}"
    --save_total_limit 2

)

log_section "Script: ${LAUNCHER[*]}" "${JOB_ID}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
NCCL_P2P_LEVEL=NVL \
"${LAUNCHER[@]}"