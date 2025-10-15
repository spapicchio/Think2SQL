#!/bin/bash
#SBATCH --job-name=eval-Qwen25-coder

# --- robust shell settings ---
set -Eeuo pipefail

# --- GPU selection (override per call if needed) ---
#MODEL_NAME='Qwen/Qwen2.5-Coder-0.5B-Instruct'
#MODEL_NAME='Qwen/Qwen2.5-Coder-1.5B-Instruct'
#MODEL_NAME='Qwen/Qwen2.5-Coder-3B-Instruct'
#MODEL_NAME='Qwen/Qwen2.5-Coder-7B-Instruct'
#MODEL_NAME='Qwen/Qwen2.5-Coder-14B-Instruct'
MODEL_NAME='Qwen/Qwen2.5-Coder-32B-Instruct'

CUDA_VISIBLE_DEVICES='4'

# --- env & utils ---
source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"

# ----------- Configuration -----------
export OMP_NUM_THREADS=10

# label, dataset, db_path, id_json (4 fields per tuple)
datasets=(
  "SPIDER-dev"        "data/omnisql/data/dev_spider.json"                 "data/omnisql/data/spider/database"                         "data/omnisql/data/spider/dev.json"
  "SPIDER-test"       "data/omnisql/data/test_spider.json"                "data/omnisql/data/spider/database"                         "data/omnisql/data/spider/test.json"
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

# run GREEDY (temp=0.0, top_p=1.0)
TEMP=0.0
TOP_P=1.0
TOP_K=0
REP_PENALTY=1.0
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

# majority Voting
TEMP=0.8
TOP_P=0.8
TOP_K=20
REP_PENALTY=1.1
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

