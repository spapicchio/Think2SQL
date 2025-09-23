#!/bin/bash
set -e
#BASE_WORK='/workspaces/Think2SQL'
BASE_WORK='/home/papicchi/Think2SQL'

# Check if job script is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path_to_script"
  exit 1
fi

if [ -z "$WORK" ]; then
  echo "WORK variable is not set. Assuming running from ${BASE_WORK}."
  export WORK=${BASE_WORK}
fi

source "${WORK}/scripts/utils/utils.sh"

if [ -z "$ALL_CCFRWORK" ]; then
  echo "ALL_CCFRWORK variable is not set. Assuming running from ${BASE_WORK}."
  export ALL_CCFRWORK=${BASE_WORK}
fi

JOB_SCRIPT="$1"

JOB_NAME=$(awk -F= '/^#SBATCH[[:space:]]+--job-name=/ {print $2; exit}' "$JOB_SCRIPT")

# copy and then launch the file
# Generate a fake job ID based on the current timestamp and a hash
FAKE_JOB_ID=$(date +%s | sha256sum | head -c 8)
echo "Generated fake job ID for date $(date +%s): $FAKE_JOB_ID"
FAKE_JOB_PATH="${WORK}/scripts/launched/$(date +%Y-%m-%d)/$(date +%H-%M-%S)-${JOB_NAME}-${FAKE_JOB_ID}.sh"
mkdir -p "$(dirname "$FAKE_JOB_PATH")"
cp "$JOB_SCRIPT" "$FAKE_JOB_PATH"
chmod 770 "$FAKE_JOB_PATH"

MY_SLURM_JOB_ID="${JOB_NAME}-${FAKE_JOB_ID}"

# Submit the job
if [ -z "$2" ]; then
  echo 'NOT sending with sbatch'
  LOG_FOLDER="./tmux_log/$(date +%Y-%m-%d)/$(date +%H-%M-%S)-${JOB_NAME}-${FAKE_JOB_ID}"
  mkdir -p "${LOG_FOLDER}"
  tmux new-session -d -s "${JOB_NAME}-${FAKE_JOB_ID}" \
    "WORK=${WORK} \
    ALL_CCFRWORK=${ALL_CCFRWORK} \
    JOB_NAME=${JOB_NAME} \
    MY_SLURM_JOB_ID=${MY_SLURM_JOB_ID} \
    ${FAKE_JOB_PATH} 2>&1 | \
    stdbuf -oL tee -a ${LOG_FOLDER}/all.log | \
    stdbuf -oL tee >(stdbuf -oL grep 'WARNING' >> ${LOG_FOLDER}/warning.log) | \
    stdbuf -oL tee >(stdbuf -oL grep 'ERROR' >> ${LOG_FOLDER}/error.log)"

else
  JOB_OUTPUT=$(sbatch "${FAKE_JOB_PATH}")
  echo "$JOB_OUTPUT"
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
  # update the file with its job_id
  mkdir -p "${WORK}/scripts/$(date +%Y-%m-%d)"
  mv "$FAKE_JOB_PATH" "${WORK}/scripts/launched/$(date +%Y-%m-%d)/$(date +%H-%M-%S)-${JOB_NAME}-${JOB_ID}.sh"
  FAKE_JOB_PATH="${WORK}/scripts/launched/${JOB_NAME}-${JOB_ID}.sh"
fi


log_section "created file $FAKE_JOB_PATH" "${JOB_NAME}-${JOB_ID}"
