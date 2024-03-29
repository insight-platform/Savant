#!/bin/bash
# you are expected to be in Savant/ directory
# Usage: run_perf.sh [MULTISTREAM] [YQ_ARGS]...
# The MULTISTREAM argument is an integer, 0 means using uridecodebin source,
# >0 means using the the multistream source adapter with specified number of streams.
# The YQ_ARGS are module configuration updates in yq notation,
# e.g. ".parameters.batch_size=4".

MODULE_CONFIG=samples/face_reid/src/module.yml
DATA_LOCATION=data/jumanji_cast.mp4

if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_RUNTIME="--runtime=nvidia"
  docker compose -f samples/face_reid/docker-compose.l4t.yml --profile demo build
else
  DOCKER_RUNTIME="--gpus=all"
  docker compose -f samples/face_reid/docker-compose.x86.yml --profile demo build
fi

source samples/assets/run_perf_helper.sh
set_source $DATA_LOCATION
PERF_CONFIG="${MODULE_CONFIG%.*}_perf.yml"
config_perf $MODULE_CONFIG $PERF_CONFIG "${YQ_ARGS[@]}"

docker run --rm $DOCKER_RUNTIME \
  -e MODEL_PATH=/cache/models/face_reid \
  -e DOWNLOAD_PATH=/cache/downloads/face_reid \
  -e ZMQ_SRC_ENDPOINT=router+bind:ipc:///tmp/zmq-sockets/input-video.ipc \
  -v `pwd`/samples/face_reid/src:/opt/savant/samples/face_reid \
  -v `pwd`/samples/face_reid/index_files:/index \
  -v `pwd`/data:/data:ro \
  -v `pwd`/cache:/cache \
  -v /tmp/zmq-sockets:/tmp/zmq-sockets \
  face_reid-module \
  samples/face_reid/module_perf.yml
