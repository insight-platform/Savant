#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_RUNTIME="--runtime=nvidia"
  docker compose -f samples/license_plate_recognition/docker-compose.l4t.yml build module
else
  DOCKER_RUNTIME="--gpus=all"
  docker compose -f samples/license_plate_recognition/docker-compose.x86.yml build module
fi

docker run --rm -it $DOCKER_RUNTIME \
  -e BUFFER_QUEUES \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/models/license_plate_recognition:/models \
  -v `pwd`/downloads/license_plate_recognition:/downloads \
  license_plate_recognition-module  \
  samples/license_plate_recognition/module_performance.yml
