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
trap cleanup EXIT ERR INT TERM # Clenup is a function to trap and remove the online VLLM

