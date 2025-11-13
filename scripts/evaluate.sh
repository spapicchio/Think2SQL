#!/bin/bash
#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%x-%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

# --- robust shell settings ---
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source "${BASE_WORK}/scripts/utils/slurm_job_requeue.sh"
  setup_idris
  # label, dataset, db_path
  datasets=(
    "Bird-dev"          "data/processed/dev_bird_processed_with_plan_cols_time.json"                   "data/omnisql/bird/dev_20240627/dev_databases"
    "SPIDER-test"       "data/processed/test_spider_processed_with_plan_cols_time.json"                "data/omnisql/spider/test_database"
    "SPIDER-DK"         "data/processed/dev_spider_dk_processed_with_plan_cols_time.json"              "data/omnisql/Spider-DK/database"
    "SPIDER-SYN"        "data/processed/dev_spider_syn_processed_with_plan_cols_time.json"             "data/omnisql/spider/database"
    "SPIDER-REALISTIC"  "data/processed/dev_spider_realistic_processed_with_plan_cols_time.json"       "data/omnisql/spider/database"
    "sciencebenchmark"  "data/processed/dev_sciencebenchmark_processed_with_plan_cols_time.json"       "data/omnisql/sciencebenchmark/databases"
    "EHRSQL"            "data/processed/dev_ehrsql_processed_with_plan_cols_time.json"                 "data/omnisql/EHRSQL/database"
  )
else
  source "${BASE_WORK}/scripts/utils/utils_clenup_vllm_if_crash.sh"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
  export CUDA_VISIBLE_DEVICES
    # label, dataset, db_path
  datasets=(
    "Bird-dev"          "data/omnisql/data/processed/dev_bird_processed_with_plan_cols_time.json"                   "data/omnisql/data/bird/dev_20240627/dev_databases"
    "SPIDER-test"       "data/omnisql/data/processed/test_spider_processed_with_plan_cols_time.json"                "data/omnisql/data/spider/test_database"
    "SPIDER-DK"         "data/omnisql/data/processed/dev_spider_dk_processed_with_plan_cols_time.json"              "data/omnisql/data/Spider-DK/database"
    "SPIDER-SYN"        "data/omnisql/data/processed/dev_spider_syn_processed_with_plan_cols_time.json"             "data/omnisql/data/spider/database"
    "SPIDER-REALISTIC"  "data/omnisql/data/processed/dev_spider_realistic_processed_with_plan_cols_time.json"       "data/omnisql/data/spider/database"
    "sciencebenchmark"  "data/omnisql/data/processed/dev_sciencebenchmark_processed_with_plan_cols_time.json"       "data/omnisql/data/sciencebenchmark/databases"
    "EHRSQL"            "data/omnisql/data/processed/dev_ehrsql_processed_with_plan_cols_time.json"                 "data/omnisql/data/EHRSQL/database"
  )
fi


NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo  "Using ${NUM_GPUS} GPUs."

DATA_PARALLEL_SIZE=$NUM_GPUS
TENSOR_PARALLEL_SIZE=1

ENABLE_THINKING_MODE="${ENABLE_THINKING_MODE:-false}"
#MODEL_NAME='Qwen/Qwen3-0.6B'
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
#MODEL_NAME='Qwen/Qwen3-8B'
#MODEL_NAME='model_trained/grpo/Qwen3-4B-Instruct-2507/dapo/bs256_ml4096_gen8_qwen-think4-e1147174_RL'

MAX_PROMPT_LENGTH=8000

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MOD_LENGTH=$((MAX_PROMPT_LENGTH + MAX_NEW_TOKENS + 1024))

USER_PROMPT_NAME="${USER_PROMPT_NAME:-omnisql_user_prompt.jinja}"
SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-}"

#USER_PROMPT_NAME='base_think_user_prompt.jinja'
#SYSTEM_PROMPT_NAME='base_think_system_prompt.jinja'

# ----------- Configuration -----------
export OMP_NUM_THREADS=50

# ----------- VLLM Server -----------
VLLM_SERVER_HOST=127.0.0.1
VLLM_SERVER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "SERVER_HOST: ${VLLM_SERVER_HOST}"
echo "SERVER_PORT: ${VLLM_SERVER_PORT}"

## defined in utils.sh
launch_vllm "$CUDA_VISIBLE_DEVICES" \
            "$MODEL_NAME" \
            "$VLLM_SERVER_HOST" \
            "$VLLM_SERVER_PORT" \
            "$DATA_PARALLEL_SIZE" \
            "$TENSOR_PARALLEL_SIZE" \
            0.9 \
            "$MAX_MOD_LENGTH" \

# run_suite MODEL TEMP TOP_P [TOP_K] [REPETITION_PENALTY]
run_suite() {
  local model_name="${1}"
  local temp="${2}"
  local top_p="${3}"
  local top_k="${4}"
  local repetition_penalty="${5}"
  local n="${6}"

  echo ">>> Running suite with:"
  echo "    MODEL_NAME=${model_name}"
  echo "    TEMP=${temp}  TOP_P=${top_p}  TOP_K=${top_k}  REPETITION_PENALTY=${repetition_penalty} NUM_SAMPLES=${n}"
  echo "    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  local launcher=(
    "${BASE_WORK}/src/think2sql/evaluate/main_eval.py"
    --config "${BASE_WORK}/config/config_evaluate.yaml"
    --temperature "${temp}"
    --model_name "${model_name}"
    --top_p "${top_p}"
    --top_k "${top_k}"
    --number_of_completions "${n}"
    --repetition_penalty "${repetition_penalty}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --max_model_length "${MAX_MOD_LENGTH}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    --vllm_server_host "${VLLM_SERVER_HOST}"
    --vllm_server_port "${VLLM_SERVER_PORT}"
    --litellm_provider "hosted_vllm"
    --user_prompt_name "${USER_PROMPT_NAME}"
    --system_prompt_name "${SYSTEM_PROMPT_NAME}"
    --enable_thinking_mode "${ENABLE_THINKING_MODE}"
  )

  for ((i=0; i<${#datasets[@]}; i+=3)); do
    local label="${datasets[i]}"
    local dataset="${datasets[i+1]}"
    local db_path="${datasets[i+2]}"
    local save_folder_path="${BASE_WORK}/results/${label}/${USER_PROMPT_NAME%.jinja}/${model_name//\//_}/"

    echo "=== Evaluating on ${label} ==="

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    TOKENIZERS_PARALLELISM=true \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python "${launcher[@]}" \
      --dataset_name "${BASE_WORK}/${dataset}" \
      --relative_db_base_path "${BASE_WORK}/${db_path}" \
      --save_folder_path "${save_folder_path}"

    echo "=== Done ${label} saved in ${save_folder_path} ==="
  done
}


###############################################################################
# EXAMPLES
###############################################################################
# run GREEDY
GREEDY_TEMP="${GREEDY_TEMP:-0.0}"
GREEDY_TOP_P="${GREEDY_TOP_P:-1.0}"
GREEDY_TOP_K="${GREEDY_TOP_K:-0}"
GREEDY_REP_PENALTY="${GREEDY_REP_PENALTY:-1.0}"
GREEDY_NUM_SAMPLES="${GREEDY_NUM_SAMPLES:-1}"
run_suite "$MODEL_NAME" "$GREEDY_TEMP" "$GREEDY_TOP_P" "$GREEDY_TOP_K"  "$GREEDY_REP_PENALTY" "$GREEDY_NUM_SAMPLES"

# majority voting config
MV_TEMP="${MV_TEMP:-1.0}"
MV_TOP_P="${MV_TOP_P:-0.8}"
MV_TOP_K="${MV_TOP_K:-20}"
MV_REP_PENALTY="${MV_REP_PENALTY:-1.1}"
MV_NUM_SAMPLES="${MV_NUM_SAMPLES:-8}"
run_suite "$MODEL_NAME" "$MV_TEMP" "$MV_TOP_P" "$MV_TOP_K" "$MV_REP_PENALTY" "$MV_NUM_SAMPLES"


if [[ -n "${WORK}" ]]; then
    echo "Moving file into WORK: ${WORK}"
    # Strip BASE_WORK prefix to keep relative structure
    # Moving trained model
    OUTPUT_DIR="${BASE_WORK}/results/"
    DEST="${WORK}/evaluation_results"
    cp_files "${DEST}" "${OUTPUT_DIR}" "${JOB_ID}"
else
    echo "WORK is not set; skipping move"
fi