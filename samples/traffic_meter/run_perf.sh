#!/bin/bash
# you are expected to be in Savant/ directory

MULTISTREAM=${1:-0}
MODULE_ARGS=("$@")
unset "MODULE_ARGS[0]"

MODULE_CONFIG=samples/traffic_meter/module_performance.yml
DATA_LOCATION=data/AVG-TownCentre.mp4
MODULE_ARGS+=("parameters.detector=yolov8m")

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/traffic_meter/docker-compose.l4t.yml build module
else
  docker compose -f samples/traffic_meter/docker-compose.x86.yml build module
fi

if [ "$MULTISTREAM" -gt 0 ]; then
  MODULE_ARGS+=("pipeline.source=null" "parameters.shutdown_auth=shutdown")
  SOURCE_ADAPTER=$(./scripts/run_source.py multi-stream --detach \
    --number-of-streams="$MULTISTREAM" \
    --shutdown-auth=shutdown \
    $DATA_LOCATION)
  trap "docker kill $SOURCE_ADAPTER >/dev/null 2>/dev/null" EXIT
  sleep 5
fi

./scripts/run_module.py -i traffic_meter-module $MODULE_CONFIG "${MODULE_ARGS[@]}"
