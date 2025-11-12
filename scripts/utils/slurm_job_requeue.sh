function job_requeue {
  local job_name="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
  msg="BASH - trapping signal USR1 - requeueing $job_name"
  log_section "$msg" "$job_name"
  date
  scontrol requeue "$SLURM_JOBID"
}
trap job_requeue USR1