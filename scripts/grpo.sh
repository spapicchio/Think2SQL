#!/bin/bash
#SBATCH --job-name=nips-all_db

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# --- robust shell settings ---
set -Eeuo pipefail

source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"


# If one job crash and you want to start from it again,
# set the JOB_ID to the one you want to resume from
#JOB_ID='ID-want-to-resume'
JOB_ID=${MY_SLURM_JOB_ID}

echo JOB_ID = "$JOB_ID"

# ----------- Configuration -----------
export OMP_NUM_THREADS=10
export WANDB_DIR="${WORK}/wandb/${JOB_ID}/"
export WANDB_ARTIFACT_DIR="${WORK}/wandb/${JOB_ID}/"
export TOKENIZERS_PARALLELISM=True
LOGGING_DIR_TENSORBOARD="${WORK}/.tensorboard_logging/${JOB_ID}/"

# ----------- Custom  Params -----------
PROMPT_FOLDER="${WORK}/prompts"
USER_PROMPT_NAME="base_think_user_prompt.jinja"
SYSTEM_PROMPT_NAME="base_think_system_prompt.jinja"

# ----------- Dataset Params -----------
DATASET_NAME="simone-papicchio/bird"
DB_PATH="${WORK}/data/bird/train_databases"

# ----------- Training Params -----------
LOSS_TYPE='grpo'
REWARD_FUNCS="EX format tag_count"
REWARD_WEIGHTS="0.85 0.10 0.05"
LEARNING_RATE=1e-6
NUM_EPOCHS=1
BS=8
ACCUMULATION_STEPS=8
MAX_PROMPT_LENGTH=4096
MAX_LENGTH=4096
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
NUM_GPU_RESERVED_VLLM=1
GPU_VLLM="4"
NUM_GPUS=$((NUM_GPUS - NUM_GPU_RESERVED_VLLM))
echo "NUM_GPUS: ${NUM_GPUS}"
echo "GPU_VLLM: ${GPU_VLLM}"

TOTAL_BATCH_SIZE=$((BS * ACCUMULATION_STEPS * NUM_GPUS))
NUM_GENERATIONS=16
NUM_GENERATIONS=$(python scripts/utils/get_num_generations.py --num_gpus "$NUM_GPUS" --bs "$BS" --max_generations "$NUM_GENERATIONS")
echo "NUM_GENERATIONS: ${NUM_GENERATIONS}"

RL_MODEL_NAME="bs${TOTAL_BATCH_SIZE}_ml${MAX_LENGTH}_gen${NUM_GENERATIONS}_${JOB_ID}_RL"
echo "RL_MODEL_NAME: ${RL_MODEL_NAME}"

MODEL_BASE='Qwen2.5-Coder-7B-Instruct'
MODEL_BASE_PATH="Qwen/${MODEL_BASE}"

OUTPUT_DIR="${WORK}/model_trained/grpo/${MODEL_BASE}/${LOSS_TYPE}/${RL_MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# ----------- VLLM Server -----------
VLLM_SERVER_HOST=127.0.0.1
VLLM_SERVER_PORT=24879
echo "SERVER_HOST: ${VLLM_SERVER_HOST}"
echo "SERVER_PORT: ${VLLM_SERVER_PORT}"

CUDA_VISIBLE_DEVICES=${GPU_VLLM} \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
NCCL_P2P_LEVEL=NVL \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m trl.scripts.vllm_serve --model "$MODEL_BASE_PATH" \
--host "$VLLM_SERVER_HOST" \
--port "$VLLM_SERVER_PORT" \
--data-parallel-size "${NUM_GPU_RESERVED_VLLM}" \
--gpu-memory-utilization 0.85 &

VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= "$VLLM_PID" | tr -d ' ')
echo "vLLM PID: $VLLM_PID (PGID: $VLLM_PGID)"


LAUNCHER="\
        accelerate launch \
        --config_file ${WORK}/config/accelerate_config_grpo.yaml \
        --num_processes $NUM_GPUS \
        ${WORK}/src/think2sql/grpo/main_rl.py \
        --config ${WORK}/config/config_train_grpo.yaml \
        --prompt_folder ${PROMPT_FOLDER} \
        --user_prompt_name ${USER_PROMPT_NAME} \
        --system_prompt_name ${SYSTEM_PROMPT_NAME} \
        --dataset_name ${DATASET_NAME} \
        --relative_db_base_path ${DB_PATH} \
        --loss_type ${LOSS_TYPE} \
        --reward_funcs ${REWARD_FUNCS} \
        --reward_weights ${REWARD_WEIGHTS} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BS} \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --max_completion_length ${MAX_LENGTH} \
        --num_generations ${NUM_GENERATIONS} \
        --model_name_or_path ${MODEL_BASE_PATH} \
        --output_dir ${OUTPUT_DIR} \

        --logging_dir ${LOGGING_DIR_TENSORBOARD} \
        --run_name ${JOB_NAME} \
        --log_completions True \
        --num_completions_to_print 0 \
        --vllm_server_host ${VLLM_SERVER_HOST} \
        --vllm_server_port ${VLLM_SERVER_PORT} \
        --save_steps 5 \
        --save_total_limit 1 \
        --ddp_timeout=7200"  # https://github.com/huggingface/open-r1/issues/160

log_section "Script: ${LAUNCHER}" "${JOB_NAME}"

CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_LEVEL=NVL \
$LAUNCHER


