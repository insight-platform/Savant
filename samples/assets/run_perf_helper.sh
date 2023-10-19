#!/bin/bash

# run_perf.sh arguments
MULTISTREAM=${1:-0}
YQ_ARGS=("$@")
unset "YQ_ARGS[0]"

########################
# Creates a perf version of the module config
# Arguments:
#   $1 - source module config file path
#   $2 - resulting module config file path
#   $3+ - config updates in yq notation, eg. ".parameters.batch_size=999"
#########################
function config_perf {
  local module_config=${1-}
  local perf_config=${2-}
  shift 2
  local yq_args=("$@")

  # default updates:
  # - reset output_frame
  # - add stat_logger pyfunc
  # - set devnull_sink
  yq_cmd=$(cat <<-END
.parameters.output_frame = null |
.pipeline.elements += {
  "element": "pyfunc",
  "module": "savant.utils.stat_logger",
  "class_name": "StatLogger"
} |
.pipeline.sink = [{"element": "devnull_sink"}]
END
  )

  # additional updates
  if (( ${#yq_args[@]} != 0 )); then
    yq_cmd+=$(printf " |\n%s" "${yq_args[@]}")
  fi

  # apply yq updates to module config, save resulting config file
  docker run -i --rm mikefarah/yq "$yq_cmd" < "$module_config" > "$perf_config"
}

########################
# Set up source, runs multistream source adapter if needed
# Globals:
#   MULTISTREAM
#   YQ_ARGS
# Arguments:
#   $1 - data location, eg. video file path
#########################
function set_source {
  local data_location=${1-}

  if [ "$MULTISTREAM" -gt 0 ]; then
    echo "Starting multi-stream source adapter with $MULTISTREAM stream(s)."
    YQ_ARGS+=('.parameters.shutdown_auth="shutdown"')
    source_adapter=$(./scripts/run_source.py multi-stream --detach \
      --number-of-streams="$MULTISTREAM" \
      --shutdown-auth=shutdown \
      "$data_location")
    trap "docker kill $source_adapter >/dev/null 2>/dev/null" EXIT
    sleep 5

  else
    echo "Starting with uridecodebin source."
    uridecodebin_source=$(cat <<-END
{
  "element": "uridecodebin",
  "properties": {
    "uri": "file:///$data_location"
  }
}
END
    )
    YQ_ARGS+=(".pipeline.source = $uridecodebin_source")
  fi
}
