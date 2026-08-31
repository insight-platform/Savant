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
#########################
function shutdown_child {
  local child_pid=${1}
  local name=${2-adapter}
  local timeout=${SHUTDOWN_TIMEOUT:-5}
  echo "${name}: shutdown requested, sending SIGINT to the pipeline, waiting up to ${timeout}s" >&2
  kill -s SIGINT "${child_pid}"
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
  echo "${name}: pipeline exited with status ${child_status}" >&2
}
