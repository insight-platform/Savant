#!/bin/bash
# You are expected to be in Savant/ directory.
# Usage: ./samples/opencv_cuda_bg_remover_mog2/run_perf.sh [MULTISTREAM] [MODULE_ARGS]
# Set MULTISTREAM>0 to use multistream source with number-of-streams=MULTISTREAM,
# the default source is `uridecodebin`.

MULTISTREAM=${1:0}
MODULE_ARGS=("$@")
unset "MODULE_ARGS[0]"

MODULE_CONFIG=samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
DATA_LOCATION=data/road_traffic.mp4

if [ "$MULTISTREAM" -gt 0 ]; then
  MODULE_ARGS+=("pipeline.source=" "parameters.shutdown_auth=shutdown" "parameters.fps_period=1000000")
  SOURCE_ADAPTER=$(./scripts/run_source.py multi-stream --detach \
    --number-of-streams="$MULTISTREAM" \
    --shutdown-auth=shutdown \
    $DATA_LOCATION)
  trap "docker kill $SOURCE_ADAPTER >/dev/null 2>/dev/null" EXIT
  sleep 5
fi

./scripts/run_module.py $MODULE_CONFIG "${MODULE_ARGS[@]}"
