#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_RUNTIME="--runtime=nvidia"
  docker compose -f samples/key_point_detection/docker-compose.l4t.yml build module
else
  DOCKER_RUNTIME="--gpus=all"
  docker compose -f samples/key_point_detection/docker-compose.x86.yml build module
fi

docker run --rm -it $DOCKER_RUNTIME \
  -e BUFFER_QUEUES \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/models/key_point_detection:/models \
  -v `pwd`/downloads/key_point_detection:/downloads \
  key_point_detection-module  \
  samples/key_point_detection/module_performance.yml
