#!/bin/bash
# you are expected to be in Savant/ directory

MULTISTREAM=${1:0}
MODULE_ARGS=("$@")
unset "MODULE_ARGS[0]"

MODULE_CONFIG=samples/conditional_video_processing/demo_performance.yml
DATA_LOCATION=data/conditional_video_processing.mp4

if [ "$MULTISTREAM" -gt 0 ]; then
  MODULE_ARGS+=(
    "pipeline.source=null"
    "parameters.shutdown_auth=shutdown"
    "parameters.fps_period=1000000"
    "parameters.batched_push_timeout=200000"
  )
  SOURCE_ADAPTER=$(./scripts/run_source.py multi-stream --detach \
    --number-of-streams="$MULTISTREAM" \
    --shutdown-auth=shutdown \
    $DATA_LOCATION)
  trap "docker kill $SOURCE_ADAPTER >/dev/null 2>/dev/null" EXIT
  sleep 5
fi

./scripts/run_module.py $MODULE_CONFIG "${MODULE_ARGS[@]}"
