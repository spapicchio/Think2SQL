#!/bin/bash
#SBATCH --job-name=eval-omnisql7b-bird

CUDA_VISIBLE_DEVICES='6'
export CUDA_VISIBLE_DEVICES

# --- robust shell settings ---
set -Eeuo pipefail

source "${WORK}/.env"
source "${WORK}/scripts/utils/utils.sh"

# ----------- Configuration -----------
export OMP_NUM_THREADS=10
export TOKENIZERS_PARALLELISM=True


LAUNCHER="\
        ${WORK}/src/think2sql/evaluate/main_eval.py \
        --config ${WORK}/config/config_evaluate.yaml"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
TOKENIZERS_PARALLELISM=false \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python $LAUNCHER