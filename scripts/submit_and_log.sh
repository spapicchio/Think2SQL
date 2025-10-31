#!/bin/bash
# --- robust shell settings ---
set -Eeuo pipefail

# Check if job script is provided
if [ -z "$1" ]; then
  echo "[SUBMIT_AND_LOG] Usage: $0 path_to_script"
  exit 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
  # Determine script path robustly (works when sourced or executed)
  SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
  SCRIPT_REALPATH="$(realpath "$SCRIPT_PATH" 2>/dev/null || echo "$SCRIPT_PATH")"
  BASE_WORK="$(dirname "$(dirname "$SCRIPT_REALPATH")")"
  echo "[SUBMIT_AND_LOG] WORK variable is not set. Using BASE_WORK=${BASE_WORK}."
else
  BASE_WORK="${SCRATCH}/Think2SQL"
fi

export BASE_WORK
echo "[SUBMIT_AND_LOG] Using BASE_WORK=${BASE_WORK}"
source "${BASE_WORK}/scripts/utils/utils.sh"

JOB_SCRIPT="$1"
JOB_NAME=$(awk -F= '/^#SBATCH[[:space:]]+--job-name=/ {print $2; exit}' "$JOB_SCRIPT")

# copy and then launch the file
# Generate a fake job ID based on the current timestamp and a hash
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  FAKE_JOB_ID=$(date +%s | sha256sum | head -c 8)
  echo "[SUBMIT_AND_LOG] Generated fake job ID for date $(date +%s): $FAKE_JOB_ID"
else
  FAKE_JOB_ID="$SLURM_JOB_ID"
  echo "[SUBMIT_AND_LOG] Using SLURM_JOB_ID as job ID: $FAKE_JOB_ID"
fi


MY_SLURM_JOB_ID="${JOB_NAME}-${FAKE_JOB_ID}"


DATE_DIR="$(date +%Y-%m-%d)"
TIME_TAG="$(date +%H-%M-%S)"
DEST_DIR="${BASE_WORK}/scripts/launched/${DATE_DIR}"
mkdir -p "${DEST_DIR}"
FAKE_JOB_PATH="${DEST_DIR}/${TIME_TAG}-${MY_SLURM_JOB_ID}.sh"

cp "$JOB_SCRIPT" "$FAKE_JOB_PATH"
chmod 770 "$FAKE_JOB_PATH"

# Submit the job
LOG_FOLDER="${BASE_WORK}/tmux_log/${DATE_DIR}"
if [ -z "${2:-}" ]; then
  echo '[SUBMIT_AND_LOG]  NOT sending with sbatch'
  LOG_FOLDER="${LOG_FOLDER}/${TIME_TAG}-${MY_SLURM_JOB_ID}"
  mkdir -p "${LOG_FOLDER}"
  tmux new-session -d -s "${MY_SLURM_JOB_ID}" \
    "BASE_WORK=${BASE_WORK} \
    JOB_NAME=${JOB_NAME} \
    MY_SLURM_JOB_ID=${MY_SLURM_JOB_ID} \
    ${FAKE_JOB_PATH} 2>&1 | \
    stdbuf -oL tee -a ${LOG_FOLDER}/all.log | \
    stdbuf -oL tee >(stdbuf -oL grep 'WARNING' >> ${LOG_FOLDER}/warning.log) | \
    stdbuf -oL tee >(stdbuf -oL grep 'ERROR' >> ${LOG_FOLDER}/error.log)"
else
  JOB_OUTPUT=$(sbatch "${FAKE_JOB_PATH}")
  MY_SLURM_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')

  LOG_FOLDER="${LOG_FOLDER}/${TIME_TAG}-${JOB_NAME}-${MY_SLURM_JOB_ID}"
  mkdir -p $LOG_FOLDER
  SLURM_LOG="${BASE_WORK}/logs/rl/${JOB_NAME}-${MY_SLURM_JOB_ID}.out"
  ln -s "$SLURM_LOG" "${LOG_FOLDER}/all.out"

  NEW_PATH="${DEST_DIR}/${TIME_TAG}-${JOB_NAME}-${MY_SLURM_JOB_ID}.sh"
  mv "$FAKE_JOB_PATH" "${NEW_PATH}"
  FAKE_JOB_PATH=$NEW_PATH
fi

log_section "[SUBMIT_AND_LOG] TMUX log ${LOG_FOLDER}/all.out" "${MY_SLURM_JOB_ID}"
log_section "[SUBMIT_AND_LOG] created file $FAKE_JOB_PATH" "${JOB_NAME}-${MY_SLURM_JOB_ID}"
