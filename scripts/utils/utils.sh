#!/bin/bash

LOGFILE="${WORK}/log_sbatch.log"

echo 'loading utils.sh'

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

  #PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${devices} \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  NCCL_P2P_LEVEL=NVL  \
  vllm serve "${model_name}" \
  --host "${host}" \
  --port "${port}" \
  --disable-uvicorn-access-log \
  --data-parallel-size "${dps}" \
  --tensor-parallel-size "${tps}" \
  --gpu-memory-utilization ${gpumemory} &
}

# kill vLLM on exit or error
cleanup() {
  local code=$?
  echo "[CLEANUP] exit code: $code"
  # kill by saved PID / process group if available
  if [[ -n "${VLLM_PID:-}" ]] && ps -p "$VLLM_PID" >/dev/null 2>&1; then
    # kill the whole process group (covers workers spawned by the server)
    if [[ -n "${VLLM_PGID:-}" ]]; then
      kill -TERM "-${VLLM_PGID}" 2>/dev/null || true
      sleep 1
      kill -KILL "-${VLLM_PGID}" 2>/dev/null || true
    else
      kill -TERM "${VLLM_PID}" 2>/dev/null || true
      sleep 1
      kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
  # final safety: free the port if something else is still listening
  if command -v lsof >/dev/null 2>&1; then
    PID_ON_PORT=$(lsof -t -iTCP:"${VLLM_SERVER_PORT:-0}" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "${PID_ON_PORT:-}" ]]; then
      PGID_ON_PORT=$(ps -o pgid= "${PID_ON_PORT}" | tr -d ' ' || true)
      [[ -n "${PGID_ON_PORT:-}" ]] && kill -TERM "-${PGID_ON_PORT}" 2>/dev/null || kill -TERM "${PID_ON_PORT}" 2>/dev/null || true
      sleep 1
      [[ -n "${PGID_ON_PORT:-}" ]] && kill -KILL "-${PGID_ON_PORT}" 2>/dev/null || kill -KILL "${PID_ON_PORT}" 2>/dev/null || true
    fi
  fi
  # don’t fail the trap
  return $code
}
trap cleanup EXIT ERR INT TERM

