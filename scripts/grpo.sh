#!/bin/bash
#SBATCH --job-name=qwen-think4
DEVICE_TRL='1,2,3,4'
NUM_GPUS=4

DEVICE_VLLM='5,6'
NUM_GPU_RESERVED_VLLM=2

echo "NUM_GPUS: ${NUM_GPUS}"
echo "GPU_VLLM: ${DEVICE_VLLM}"

# --- robust shell settings ---
set -Eeuo pipefail

source "${BASE_WORK}/.env"
source "${BASE_WORK}/scripts/utils/utils.sh"


# If one job crash and you want to start from it again,
# set the JOB_ID to the one you want to resume from
JOB_ID='qwen-think4-6e12dc09'
export WANDB_RUN_ID='0czrmvbj'

# JOB_ID=${MY_SLURM_JOB_ID}

log_section "JOB_ID = ${JOB_ID}" "${JOB_ID}"

# ----------- Configuration -----------
export OMP_NUM_THREADS=50
export WANDB_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export WANDB_ARTIFACT_DIR="${BASE_WORK}/wandb/${JOB_ID}/"
export TOKENIZERS_PARALLELISM=True
LOGGING_DIR_TENSORBOARD="${BASE_WORK}/.tensorboard_logging/${JOB_ID}/"

# ----------- Custom  Params -----------
PROMPT_FOLDER="${BASE_WORK}/prompts"
USER_PROMPT_NAME="base_think_user_prompt.jinja"
SYSTEM_PROMPT_NAME="base_think_system_prompt.jinja"

# ----------- Dataset Params -----------
DATASET_NAME="${BASE_WORK}/data/omnisql/data/train_bird_processed.json"
DB_PATH="${BASE_WORK}/data/omnisql/data/bird/train/train_databases"

# ----------- Training Params -----------
LOSS_TYPE='dapo'
REWARD_FUNCS="EX"
REWARD_WEIGHTS="1.0"
LEARNING_RATE=1e-6
NUM_EPOCHS=2
BS=8
ACCUMULATION_STEPS=8
MAX_PROMPT_LENGTH=6000
MAX_LENGTH=4096
MAX_MODEL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_LENGTH + 1024))

TOTAL_BATCH_SIZE=$((BS * ACCUMULATION_STEPS * NUM_GPUS))
NUM_GENERATIONS=8
NUM_GENERATIONS=$(python scripts/utils/get_num_generations.py --num_gpus "$NUM_GPUS" --bs "$BS" --max_generations "$NUM_GENERATIONS")
echo "NUM_GENERATIONS: ${NUM_GENERATIONS}"

RL_MODEL_NAME="bs${TOTAL_BATCH_SIZE}_ml${MAX_LENGTH}_gen${NUM_GENERATIONS}_${JOB_ID}_RL"
echo "RL_MODEL_NAME: ${RL_MODEL_NAME}"

# MODEL_BASE='Qwen3-4B-Thinking-2507'
MODEL_BASE='Qwen3-4B-Instruct-2507'
MODEL_BASE_PATH="Qwen/${MODEL_BASE}"

OUTPUT_DIR="${BASE_WORK}/model_trained/grpo/${MODEL_BASE}/${LOSS_TYPE}/${RL_MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# ----------- VLLM Server -----------
VLLM_SERVER_HOST=127.0.0.1
VLLM_SERVER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "SERVER_HOST: ${VLLM_SERVER_HOST}"
echo "SERVER_PORT: ${VLLM_SERVER_PORT}"

# https://huggingface.co/docs/trl/main/en/vllm_integration
launch_trl_vllm ${DEVICE_VLLM} $MODEL_BASE_PATH false "$VLLM_SERVER_HOST" "$VLLM_SERVER_PORT" "${NUM_GPU_RESERVED_VLLM}" $MAX_MODEL_LENGTH

LAUNCHER=(
        accelerate launch
        --config_file "${BASE_WORK}/config/accelerate_config_grpo.yaml"
        --num_processes "$NUM_GPUS"
        "${BASE_WORK}/src/think2sql/grpo/main_rl.py"
        --config "${BASE_WORK}/config/config_train_grpo.yaml"
        --prompt_folder "${PROMPT_FOLDER}"
        --user_prompt_name "${USER_PROMPT_NAME}"
        --system_prompt_name "${SYSTEM_PROMPT_NAME}"
        --dataset_name "${DATASET_NAME}"
        --relative_db_base_path "${DB_PATH}"
        --loss_type "${LOSS_TYPE}"
        --reward_funcs $REWARD_FUNCS
        --reward_weights $REWARD_WEIGHTS
        --learning_rate "${LEARNING_RATE}"
        --num_train_epochs "${NUM_EPOCHS}"
        --per_device_train_batch_size "${BS}"
        --gradient_accumulation_steps "${ACCUMULATION_STEPS}"
        --max_prompt_length "${MAX_PROMPT_LENGTH}"
        --max_completion_length "${MAX_LENGTH}"
        --num_generations "${NUM_GENERATIONS}"
        --model_name_or_path "${MODEL_BASE_PATH}"
        --output_dir "${OUTPUT_DIR}"

        --logging_dir "${LOGGING_DIR_TENSORBOARD}"
        --run_name "${JOB_NAME}"
        --log_completions True
        --num_completions_to_print 0
        --vllm_server_host "${VLLM_SERVER_HOST}"
        --vllm_server_port "${VLLM_SERVER_PORT}"
        --save_steps 5
        --save_total_limit 1
        --ddp_timeout=7200  # https://github.com/huggingface/open-r1/issues/160
)

log_section "Script: ${LAUNCHER[*]}" "${JOB_ID}"


#TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
CUDA_VISIBLE_DEVICES="${DEVICE_TRL}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
NCCL_P2P_LEVEL=NVL \
"${LAUNCHER[@]}"