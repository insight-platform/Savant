#!/bin/bash
# you are expected to be in Savant/ directory

MULTISTREAM=${1:0}
MODULE_ARGS=("$@")
unset "MODULE_ARGS[0]"

MODULE_CONFIG=samples/face_reid/module_performance.yml
DATA_LOCATION=data/jumanji_cast.mp4

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/face_reid/docker-compose.l4t.yml --profile demo build
else
  docker compose -f samples/face_reid/docker-compose.x86.yml --profile demo build
fi

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

./scripts/run_module.py -i face_reid-module $MODULE_CONFIG "${MODULE_ARGS[@]}"
