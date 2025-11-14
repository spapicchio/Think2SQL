#!/bin/bash

export WANDB_PROJECT='think2sql'

if [ -z "$BASE_WORK" ]; then
  echo "BASE_WORK not set. must be set before sourcing utils.sh"
  exit 1
fi
LOGFILE="${BASE_WORK}/log_sbatch.log"
echo "loading utils.sh with LOGFILE: ${LOGFILE}"

log_section() {
  local msg="$1"
  local job_name="$2"

  local timestamp
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  {
    echo "--------------------------------------------------"
    echo "[$timestamp][$job_name] $msg"
    echo "--------------------------------------------------"
  } >>"$LOGFILE"

  echo "[$timestamp][$job_name] $msg"
}

# ----------- Launch VLLM  -----------
launch_vllm() {
  local devices="${1:-0}"           # default GPU(s)
  local model_name="${2:?MODEL NAME REQUIRED}"
  local host="${3:-127.0.0.1}"
  local port="${4:-0}"              # 0 = auto-pick free port
  local dps="${5:-1}"
  local tps="${6:-1}"
  local gpumemory="${7:-0.8}"
  local max_model_length="${8:-2048}"
  local max_num_seqs="${9:-120}"

  echo "[LAUNCH VLLM] model=${model_name}, host=${host}, port=${port}, devices=${devices}, dps=${dps}, tps=${tps}, gpumemory=${gpumemory}, max_model_length=${max_model_length}, max_num_seqs=${max_num_seqs}"
  #PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \

  local launcher=(
    vllm serve "${model_name}"
    --port "${port}"
    --disable-uvicorn-access-log
    --data-parallel-size "${dps}"
    --tensor-parallel-size "${tps}"
    --gpu-memory-utilization ${gpumemory}
    --dtype "bfloat16"
    --swap-space 42
    --disable-custom-all-reduce
    --max-model-len "${max_model_length}"
    --max-num-seqs "${max_num_seqs}"
  )

  CUDA_VISIBLE_DEVICES=${devices} \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  NCCL_P2P_LEVEL=NVL  \
  "${launcher[@]}" &
}


launch_trl_vllm() {
  local devices="${1:-0}"           # default GPU(s)
  local model_name="${2:?MODEL NAME REQUIRED}"
  local is_slurm="${3:-false}"
  local host="${4:-127.0.0.1}"
  local port="${5:-0}"              # 0 = auto-pick free port
  local dps="${6:-1}"
  local max_model_length="${7:-2048}"

  local launcher=(
  python -m trl.scripts.vllm_serve
  --model "$model_name"
  --host "$host"
  --port "$port"
  --data-parallel-size "${dps}" \
  --gpu-memory-utilization 0.85 \
  --log_level 'warning' \
  --max_model_len "${max_model_length}"
  )
  if [[ "$is_slurm" == "true" ]]; then
    echo "[LAUNCH TRL VLLM] on SLURM node, model=${model_name}, vllm_node=${VLLM_NODE}"
    srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist=$host \
    "${launcher[@]}" &
  else
    echo "[LAUNCH TRL VLLM] locally, model=${model_name}, devices=${devices}"
    CUDA_VISIBLE_DEVICES=${devices} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    NCCL_P2P_LEVEL=NVL \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${launcher[@]}" &
    VLLM_PID=$!
    VLLM_PGID=$(ps -o pgid= "$VLLM_PID" | tr -d ' ')
    export VLLM_PID
    export VLLM_PGID
    echo "vLLM PID: $VLLM_PID (PGID: $VLLM_PGID)"
  fi
}

function setup_idris {
  # Set some env variable for running on compute nodes without internet access
  export GIT_PYTHON_REFRESH=quiet
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export WANDB_MODE=offline

  NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
  export MASTER_ADDR=${NODELIST[0]} # First node for main process
  export MASTER_PORT=6000

  # setup one node for vLLM
  WORLD_SIZE=$(($SLURM_NNODES * SLURM_GPUS_PER_NODE))
  export TRAIN_NODES=("${NODELIST[@]:0:$((SLURM_NNODES - 1))}")  # Last node used for vLLM
  export TRAIN_NODES_STR=$(NODES="${TRAIN_NODES[*]}" python -c 'import os; print(",".join(os.getenv("NODES").split()))')

  export VLLM_NODE=${NODELIST[-1]} # Last node
  export WORLD_SIZE=$((WORLD_SIZE - SLURM_GPUS_PER_NODE)) # exclude vLLM node
  export NUM_NODES=$((SLURM_NNODES - 1)) # exclude vLLM node
}

function cp_files {
  local dest="$1"
  local src_file="$2"
  local job_id="$3"
  log_section "From '${dest}' to '${src_file}'" "${job_id}"
  mkdir -p "${dest}"
  cp -r "${src_file}" "${dest}"
}