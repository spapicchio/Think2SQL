#!/bin/bash
#SBATCH -A vno@a100
#SBATCH -C a100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

set -e

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
source "${BASE_WORK}/scripts/utils/slurm_job_requeue.sh"

setup_idris  # function in utils.sh

uv sync --frozen
source "${BASE_WORK}/.venv/bin/activate"


log_section "Environment and modules loaded with BASE_WORK=${BASE_WORK}" "${SLURM_JOB_ID}"

DEVICE_TRL='0'
NUM_GPUS=1
DEVICE_VLLM='1'
NUM_GPU_RESERVED_VLLM=1

echo "NUM_GPUS: ${NUM_GPUS}"
echo "GPU_VLLM: ${DEVICE_VLLM}"


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
USER_PROMPT_NAME="base_think_user_prompt.jinja"
SYSTEM_PROMPT_NAME="base_think_system_prompt.jinja"
ENABLE_THINKING_MODE='False'

# ----------- Dataset Params -----------
DATASET_NAME="${BASE_WORK_DATA}/data/train_bird_processed_with_plan_cols_time.json"
DB_PATH="${BASE_WORK_DATA}/data/bird/train/train_databases"


# ----------- Training Params -----------
LOSS_TYPE='dapo'
REWARD_FUNCS="QATCH format"
REWARD_WEIGHTS="0.9 0.1"
LEARNING_RATE=1e-6
NUM_EPOCHS=1
BS=8
ACCUMULATION_STEPS=8
MAX_PROMPT_LENGTH=8000
MAX_LENGTH=8092
MAX_MODEL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_LENGTH + 1024))

TOTAL_BATCH_SIZE=$((BS * ACCUMULATION_STEPS * NUM_GPUS))
NUM_GENERATIONS=16
NUM_GENERATIONS=$(python scripts/utils/get_num_generations.py --num_gpus "$NUM_GPUS" --bs "$BS" --max_generations "$NUM_GENERATIONS")
log_section "NUM_GENERATIONS: ${NUM_GENERATIONS}" "${JOB_ID}"

RL_MODEL_NAME="bs${TOTAL_BATCH_SIZE}_ml${MAX_LENGTH}_gen${NUM_GENERATIONS}_${JOB_ID}_RL"
log_section "RL_MODEL_NAME: ${RL_MODEL_NAME}" "${JOB_ID}"

MODEL_BASE='Qwen3-4B-Instruct-2507'
MODEL_BASE_PATH="Qwen/${MODEL_BASE}"
MODEL_BASE_PATH=$(python scripts/utils/get_model_path_hf_cache.py --cache_dir "${HF_HOME}" --model_id "${MODEL_BASE_PATH}")


OUTPUT_DIR="${BASE_WORK_MODEL}/${LOSS_TYPE}/${MODEL_BASE}/${RL_MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"
log_section "OUTPUT_DIR: ${OUTPUT_DIR}" "${JOB_ID}"

# ----------- VLLM Server -----------
VLLM_SERVER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
VLLM_SERVER_HOST=127.0.0.1
echo "SERVER_HOST: ${VLLM_SERVER_HOST}"
echo "SERVER_PORT: ${VLLM_SERVER_PORT}"

# https://huggingface.co/docs/trl/main/en/vllm_integration
launch_trl_vllm ${DEVICE_VLLM} $MODEL_BASE_PATH false "$VLLM_SERVER_HOST" "$VLLM_SERVER_PORT" "$NUM_GPU_RESERVED_VLLM" "${MAX_MODEL_LENGTH}"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Launcher
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        --run_name "${JOB_ID}"
        --enable_thinking_mode "${ENABLE_THINKING_MODE}"
        --scale_rewards 'none'
        --mask_truncated_completions 'True'
        --log_completions True
        --num_completions_to_print 0
        --vllm_server_host "${VLLM_SERVER_HOST}"
        --vllm_server_port "${VLLM_SERVER_PORT}"
        --save_steps 5
        --save_total_limit 1
        --ddp_timeout=7200  # https://github.com/huggingface/open-r1/issues/160
)
log_section "Script: ${LAUNCHER[*]}" "${JOB_ID}"


srun \
--wait=60 \
--kill-on-bad-exit=1 \
--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1,NCCL_P2P_LEVEL=NVL,CUDA_VISIBLE_DEVICES=${DEVICE_TRL}  \
"${LAUNCHER[@]}"

# The chat template needs to be specified only for DeepSeek models
#    --chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
if [[ -n "${WORK}" ]]; then    
    echo "Moving file into WORK: ${WORK}"
    # Strip BASE_WORK prefix to keep relative structure
    # Moving trained model
    REL_PATH="${OUTPUT_DIR#${BASE_WORK_MODEL}/}"
    DEST="${WORK}/${REL_PATH}"
    cp_files "${DEST}" "${OUTPUT_DIR}" "${JOB_ID}"
    # Moving wandb logs
    REL_PATH_WANDB="${WANDB_DIR#${BASE_WORK}/}"
    DEST_WANDB="${WORK}/${REL_PATH_WANDB}"
    cp_files "${DEST_WANDB}" "${WANDB_DIR}" "${JOB_ID}"
    # Moving tmux log specified in submit and log!
    SLURM_LOG="${BASE_WORK}/logs/rl/${SLURM_JOB_NAME}-${JOB_ID}.out"
    REL_PATH_LOGS="${SLURM_LOG#${BASE_WORK}/}"
    DEST_LOGS="${WORK}/${REL_PATH_LOGS}"
    cp_files "${DEST_LOGS}" "${SLURM_LOG}" "${JOB_ID}"
else
    echo "WORK is not set; skipping move"
fi