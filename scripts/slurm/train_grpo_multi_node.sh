#!/bin/bash
#SBATCH -A wjx@h100
#SBATCH -C h100
#SBATCH --job-name=rl-exp-30
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --output=./logs/rl/%x-%j.out
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

set -e

source slurm/scripts/env_setup.sh
source slurm/scripts/utils.sh
source slurm/scripts/training_fun.sh

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Define the JOB ID of the run.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# if you are relaunchin a job and want to keep the same folders for logging and continue training
#MY_SLURM_JOB_ID='<your id>'

# if the job has been just launched
MY_SLURM_JOB_ID=$SLURM_JOB_ID

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Define folders for logging Wandb/Tensorboard
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#export WANDB_DIR="wandb/${MY_SLURM_JOB_ID}/"
#export WANDB_ARTIFACT_DIR="wandb/${MY_SLURM_JOB_ID}/"
LOGGING_DIR_TENSORBOARD="./.tensorboard_logging/${MY_SLURM_JOB_ID}/"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Set UP variables from slurm
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
JOB_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

log_section "Log File in: logs/rl/${JOB_NAME}.out" "${JOB_NAME}"
log_section "Wandb offline in: ${WANDB_DIR}" "${JOB_NAME}"
log_section "Tensorboard in: ${LOGGING_DIR_TENSORBOARD}" "${JOB_NAME}"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]} # First node for main process
MASTER_PORT=6000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         VLLM parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
TRAIN_NODES_STR=$(NODES="${TRAIN_NODES[*]}" python -c 'import os; print(",".join(os.getenv("NODES").split()))')

VLLM_NODE=${NODELIST[-1]} # Last node
WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
NUM_NODES=$((NUM_NODES - 1))

log_section "VLLM NODE: ${VLLM_NODE} - Training NODES: ${TRAIN_NODES_STR} - Total number of GPUs used for VLLM: ${GPUS_PER_NODE} - Total number of GPUs used for training: ${WORLD_SIZE}" "${JOB_NAME}"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Training parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#REWARD_FUNCS="qatch_metrics execution_accuracy format tag_count"
#REWARD_WEIGHTS="0.70 0.20 0.05 0.05"

REWARD_FUNCS="qatch_metrics format tag_count"
REWARD_WEIGHTS="0.85 0.10 0.05"

DATASET_NAME="simone-papicchio/bird"
DB_PATH="${ALL_CCFRWORK}/data/bird_train/train_databases"

COMPLEXITY_BUCKET=""
SHARD_NUMBER=0
FEW_SHOT=0

ACCUMULATION_STEPS=8
BS=8
MAX_LENGTH=4096

TOTAL_BATCH_SIZE=$((BS * ACCUMULATION_STEPS * WORLD_SIZE))
NUM_GENERATIONS=16
NUM_GENERATIONS=$(python slurm/scripts/utils_get_num_generations.py --num_gpus "$WORLD_SIZE" --bs "$BS" --max_generations "$NUM_GENERATIONS")
log_section "Num of Generations: ${NUM_GENERATIONS}" "${JOB_NAME}"
RL_MODEL_NAME="bs_${TOTAL_BATCH_SIZE}_ml_${MAX_LENGTH}_gen_${NUM_GENERATIONS}_${MY_SLURM_JOB_ID}_RL"

# uncomment for finetuned model
#SFT_MODEL='Qwen/size_3B_bs_128_ml_4096'
#SFT_MODEL='Qwen/Qwen2.5-Coder-3B-Instruct_bs_128_ml_4096_sft_qwen_3B-CODER'
#SFT_MODEL='Qwen/Qwen2.5-Coder-7B-Instruct_bs_128_ml_4096_1867319_SFT'

#SFT_PATH="${ALL_CCFRWORK}/finetuned_models/sft/${SFT_MODEL}"
#OUTPUT_DIR="${ALL_CCFRWORK}/finetuned_models/grpo/${SFT_MODEL}/${RL_MODEL_NAME}"

# uncomment for base models
#SFT_MODEL='Qwen/Qwen2.5-Coder-3B-Instruct'
#SFT_MODEL='Qwen/Qwen2.5-Coder-7B-Instruct'
#SFT_MODEL='Qwen/Qwen2.5-Coder-14B-Instruct'
#SFT_MODEL='meta-llama/Llama-3.1-8B-Instruct'
SFT_MODEL='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
SFT_PATH=$SFT_MODEL
OUTPUT_DIR="${ALL_CCFRWORK}/base_models/grpo/${SFT_MODEL}/${RL_MODEL_NAME}"

LEARNING_RATE=1e-6
NUM_EPOCHS=1
PROMPT_NAME="text2sql_model_grpo"
MAX_PROMPT_LENGTH=2048
MAX_GRAD_NORM=0.2
TEMPERATURE=0.7
OPTIMIZER="adamw_8bit"
CACHED_FILE_PATH="${WORK}/deep_thinking/cache_target_sql2execution_BIRD_train.pkl"



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Vllm serve
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VLLM_SERVER_PORT=24879

TP=$(python slurm/scripts/utils_get_tensor_parallel_size.py --model_name "$SFT_PATH" --default_tp "$GPUS_PER_NODE")
log_section "Tensor parallel size : ${TP}" "${JOB_NAME}"

srun \
--nodes=1 \
--ntasks=1 \
--nodelist=$VLLM_NODE \
trl vllm-serve \
--model "$SFT_PATH" \
--data-parallel-size "${GPUS_PER_NODE}" \
--gpu-memory-utilization 0.85 \
--port "$VLLM_SERVER_PORT" \
--host="$VLLM_NODE" &

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         Launcher
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LAUNCHER="\
    srun \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$TRAIN_NODES_STR \
    --export=ALL,ACCELERATE_LOG_LEVEL=info,TRANSFORMERS_VERBOSITY=info,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    accelerate launch \
    --config_file config/accelerate_config_grpo.yaml \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --tee 3 \
    --role $SLURMD_NODENAME \
    src/deep_thinking/train/reinforcement_learning/main_rl.py \
    --config config/config_train_grpo.yaml \
    --model_name_or_path ${SFT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --dataset_name ${DATASET_NAME} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --num_generations ${NUM_GENERATIONS} \
    --prompt_name ${PROMPT_NAME} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_completion_length ${MAX_LENGTH} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --base_db_path ${DB_PATH} \
    --temperature ${TEMPERATURE} \
    --reward_funcs ${REWARD_FUNCS} \
    --reward_weights ${REWARD_WEIGHTS} \
    --optim ${OPTIMIZER} \
    --resume_from_checkpoint True \
    --logging_dir ${LOGGING_DIR_TENSORBOARD} \
    --run_name ${JOB_NAME} \
    --wandb_log_unique_prompts True \
    --log_completions True \
    --num_completions_to_print 1 \
    --vllm_server_host $VLLM_NODE \
    --vllm_server_port ${VLLM_SERVER_PORT} \
    --cached_file_path ${CACHED_FILE_PATH} \
    --save_steps 5 \
    --return_shard_number ${SHARD_NUMBER} \
    --insert_few_shot ${FEW_SHOT} \
    --ddp_timeout 7200"

# The chat template needs to be specified only for DeepSeek models
#    --chat_template "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"

$LAUNCHER &
PYTHON_PID=$!
log_section "Script: ${LAUNCHER}" "${JOB_NAME}"

wait $PYTHON_PID

check_exit_status $? "Training" "${JOB_NAME}"