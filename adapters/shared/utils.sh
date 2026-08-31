#!/bin/bash

########################
# Print starting message
# Arguments:
#   $1 - the name of starting module or adapter
#########################
function print_starting_message {
  local name=${1-}
  python3 -c "from savant.utils.welcome import get_starting_message; print(get_starting_message('$name'))"
}

########################
# Terminate the child pipeline, bounding the graceful shutdown so a stuck EOS
# cannot hang the adapter forever. Keep the timeout below the container stop
# grace period.
# Globals:
#   SHUTDOWN_TIMEOUT - seconds to wait for graceful shutdown; default is 5
# Arguments:
#   $1 - PID of the child process
#   $2 - the name of the adapter, used in the log message
# Returns:
#   The exit status of the child process
#########################
function shutdown_child {
  local child_pid=${1}
  local name=${2-adapter}
  local default_timeout=5
  local timeout=${SHUTDOWN_TIMEOUT:-${default_timeout}}

  # A non-numeric timeout makes "sleep" fail at once, which would kill the
  # pipeline with no grace period at all.
  if [[ ! "${timeout}" =~ ^[0-9]+$ ]]; then
    echo "${name}: invalid SHUTDOWN_TIMEOUT ${timeout}, using ${default_timeout}s" >&2
    timeout=${default_timeout}
  fi

  # The signal may arrive before the pipeline is started.
  if [[ -z "${child_pid}" ]]; then
    echo "${name}: shutdown requested before the pipeline started" >&2
    exit 0
  fi

  # The pipeline may have already exited on its own.
  if ! kill -0 "${child_pid}" 2>/dev/null; then
    echo "${name}: pipeline is already gone, nothing to shut down" >&2
    return 0
  fi

  echo "${name}: shutdown requested, sending SIGINT to the pipeline, waiting up to ${timeout}s" >&2
  kill -s SIGINT "${child_pid}" 2>/dev/null || true
  (
    sleep "${timeout}"
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "${name}: graceful shutdown did not finish in ${timeout}s, forcing SIGKILL" >&2
      kill -s SIGKILL "${child_pid}" 2>/dev/null
    fi
  ) &
  local killer_pid="$!"
  # Suppress the shell's own job-kill notice; the status is reported below.
  wait "${child_pid}" 2>/dev/null
  local child_status=$?
  # Graceful path won; cancel the killer before the pid can be reused.
  kill "${killer_pid}" 2>/dev/null
  wait "${killer_pid}" 2>/dev/null || true
  echo "${name}: pipeline exited with status ${child_status}" >&2

  return "${child_status}"
}
