#!/bin/bash
#SBATCH --job-name=eval-omnisql7b-bird

CUDA_VISIBLE_DEVICES='0'
export CUDA_VISIBLE_DEVICES

# --- robust shell settings ---
set -Eeuo pipefail

source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"

# ----------- Configuration -----------
export OMP_NUM_THREADS=10
export TOKENIZERS_PARALLELISM=True


# ----------- VLLM -----------
MODEL_NAME="seeklhy/OmniSQL-7B"
TP=1
DP=$(python -c "import torch; print(torch.cuda.device_count())")
MAX_MODEL_LENGTH=8192


# ----------- PROMPT -----------
USER_PROMPT_NAME='omnisql_user_prompt.jinja'
SYSTEM_PROMPT_NAME='none'

# ----------- DATASET -----------
DATASET_NAME='simone-papicchio/bird'
TASK_NAME='bird_dev'
DB_BASE_PATH='data/bird_dev/dev_databases'
ADD_SAMPLE_ROWS_STRATEGY='inline'


# ----------- GENERATION PARAMS -----------
TEMP=0.7
TOP_P=0.8
TOP_K=20
MAX_NEW_TOKENS=4048


LAUNCHER="\
        ${WORK}/src/think2sql/evaluate/main_eval.py \
        --config ${WORK}/config/config_evaluate.yaml \
        --user_prompt_name ${USER_PROMPT_NAME} \
        --system_prompt_name ${SYSTEM_PROMPT_NAME} \
        --dataset_name ${DATASET_NAME} \
        --task_name ${TASK_NAME} \
        --relative_db_base_path ${DB_BASE_PATH} \
        --add_sample_rows_strategy ${ADD_SAMPLE_ROWS_STRATEGY} \
        --model_name ${MODEL_NAME} \
        --tensor_parallel_size ${TP} \
        --data_parallel_size ${DP} \
        --max_model_length ${MAX_MODEL_LENGTH} \
        --temp ${TEMP} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        "

log_section "Script: ${LAUNCHER}" "${JOB_NAME}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
VLLM_HOST_IP='localhost' \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python $LAUNCHER