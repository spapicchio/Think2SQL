#!/bin/bash
#SBATCH --job-name=eval-mv-Qwen3-thinking-3
CUDA_VISIBLE_DEVICES='3'
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=1

# --- robust shell settings ---
set -Eeuo pipefail

# --- GPU selection (override per call if needed) ---
#MODEL_NAME='Qwen/Qwen3-30B-A3B-Thinking-2507'
MODEL_NAME='Qwen/Qwen3-4B-Thinking-2507'


MAX_NEW_TOKENS=30000
MAX_MOD_LENGTH=32768

# --- env & utils ---
source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"

# ----------- VLLM Server -----------
VLLM_SERVER_HOST=127.0.0.1
VLLM_SERVER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "SERVER_HOST: ${VLLM_SERVER_HOST}"
echo "SERVER_PORT: ${VLLM_SERVER_PORT}"

# defined in utils.sh
launch_vllm "$CUDA_VISIBLE_DEVICES" "$MODEL_NAME" "$VLLM_SERVER_HOST" "$VLLM_SERVER_PORT" "$DATA_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" 0.9 "$MAX_MOD_LENGTH"



# ----------- Configuration -----------
export OMP_NUM_THREADS=10

# label, dataset, db_path, id_json (4 fields per tuple)
datasets=(
  "SPIDER-dev"        "data/omnisql/data/dev_spider.json"                 "data/omnisql/data/spider/database"                         "data/omnisql/data/spider/dev.json"
  "SPIDER-test"       "data/omnisql/data/test_spider.json"                "data/omnisql/data/spider/test_database"                         "data/omnisql/data/spider/test.json"
  "Bird-dev"          "data/omnisql/data/dev_bird.json"                   "data/omnisql/data/bird/dev_20240627/dev_databases"         "data/omnisql/data/bird/dev_20240627/dev.json"
  "SPIDER-DK"         "data/omnisql/data/dev_spider_dk.json"              "data/omnisql/data/Spider-DK/database"                      "data/omnisql/data/Spider-DK/Spider-DK.json"
  "SPIDER-SYN"        "data/omnisql/data/dev_spider_syn.json"             "data/omnisql/data/spider/database"                         "data/omnisql/data/Spider-Syn/dev.json"
  "SPIDER-REALISTIC"  "data/omnisql/data/dev_spider_realistic.json"       "data/omnisql/data/spider/database"                         "data/omnisql/data/spider-realistic/spider-realistic.json"
  "sciencebenchmark"  "data/omnisql/data/dev_sciencebenchmark.json"       "data/omnisql/data/sciencebenchmark/databases"              "data/omnisql/data/sciencebenchmark/dev.json"
  "EHRSQL"            "data/omnisql/data/dev_ehrsql.json"                 "data/omnisql/data/EHRSQL/database"                         "data/omnisql/data/EHRSQL/dev.json"
)

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
    "${WORK}/src/think2sql/evaluate/main_eval.py"
    --config "${WORK}/config/config_evaluate.yaml"
    --temperature "${temp}"
    --model_name "${model_name}"
    --top_p "${top_p}"
    --top_k "${top_k}"
    --number_of_completions "${n}"
    --repetition_penalty "${repetition_penalty}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --max_model_length "${MAX_MOD_LENGTH}"
    --tensor_parallel_size "${NUM_GPUS}"
    --vllm_server_host "${VLLM_SERVER_HOST}"
    --vllm_server_port "${VLLM_SERVER_PORT}"
    --litellm_provider "hosted_vllm"
  )

  for ((i=0; i<${#datasets[@]}; i+=4)); do
    local label="${datasets[i]}"
    local dataset="${datasets[i+1]}"
    local db_path="${datasets[i+2]}"
    local id_json="${datasets[i+3]}"

    echo "=== Evaluating on ${label} ==="

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    TOKENIZERS_PARALLELISM=true \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python "${launcher[@]}" \
      --dataset_name "${dataset}" \
      --relative_db_base_path "${db_path}" \
      --omnisql_file_db_id_json_path "${id_json}"

    echo "=== Done ${label} ==="
  done
}

###############################################################################
# EXAMPLES
###############################################################################
# Single run with defaults:
# run_suite

# run with best parameters from HF https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507
TEMP=0.6
TOP_P=0.95
TOP_K=20
REP_PENALTY=1.05
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

# majority Voting
TEMP=1.0
TOP_P=0.95
TOP_K=20
REP_PENALTY=1.05
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

