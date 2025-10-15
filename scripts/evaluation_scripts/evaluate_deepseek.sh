#!/bin/bash
#SBATCH --job-name=eval-Llama405

# --- robust shell settings ---
set -Eeuo pipefail

# --- env & utils ---
source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"


#MODEL_NAME='deepseek-ai/DeepSeek-R1'
MODEL_NAME='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'

# deepseek
#MAX_NEW_TOKENS=30000
#MAX_MOD_LENGTH=32768

# Llama 405
MAX_NEW_TOKENS=2048
MAX_MOD_LENGTH=8192

# ----------- Configuration -----------
export OMP_NUM_THREADS=10

# label, dataset, db_path, id_json (4 fields per tuple)
datasets=(
#  "SPIDER-dev"        "data/omnisql/data/dev_spider.json"                 "data/omnisql/data/spider/database"                         "data/omnisql/data/spider/dev.json"
  "SPIDER-test"       "data/omnisql/data/test_spider.json"                "data/omnisql/data/spider/test_database"                     "data/omnisql/data/spider/test.json"
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
    --litellm_provider "together_ai"
    --num_of_experiments 1
  )

  for ((i=0; i<${#datasets[@]}; i+=4)); do
    local label="${datasets[i]}"
    local dataset="${datasets[i+1]}"
    local db_path="${datasets[i+2]}"
    local id_json="${datasets[i+3]}"

    echo "=== Evaluating on ${label} ==="

    TOKENIZERS_PARALLELISM=true \
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
TEMP=0.6
TOP_P=0.95
TOP_K=0
REP_PENALTY=1.05
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

## majority Voting
TEMP=1.0
TOP_P=0.95
TOP_K=0
REP_PENALTY=1.05
NUM_SAMPLES=8
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

# LLama405
TEMP=0.0
TOP_P=1
TOP_K=0
REP_PENALTY=1.0
NUM_SAMPLES=1
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES

## majority Voting
TEMP=1.0
TOP_P=1
TOP_K=0
REP_PENALTY=1.0
NUM_SAMPLES=8
run_suite $MODEL_NAME $TEMP $TOP_P $TOP_K $REP_PENALTY $NUM_SAMPLES