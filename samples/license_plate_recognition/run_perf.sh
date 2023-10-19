#!/bin/bash
# you are expected to be in Savant/ directory
# Usage: run_perf.sh [MULTISTREAM] [YQ_ARGS]...
# The MULTISTREAM argument is an integer, 0 means using uridecodebin source,
# >0 means using the the multistream source adapter with specified number of streams.
# The YQ_ARGS are module configuration updates in yq notation,
# e.g. ".parameters.batch_size=4".

MODULE_CONFIG=samples/license_plate_recognition/module.yml
DATA_LOCATION=data/lpr_test_1080p.mp4
PERF_CONFIG="${MODULE_CONFIG%.*}_perf.yml"

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/license_plate_recognition/docker-compose.l4t.yml build module
else
  docker compose -f samples/license_plate_recognition/docker-compose.x86.yml build module
fi

source samples/assets/run_perf_helper.sh

set_source $DATA_LOCATION

config_perf $MODULE_CONFIG $PERF_CONFIG "${YQ_ARGS[@]}"

./scripts/run_module.py -i license_plate_recognition-module $PERF_CONFIG
