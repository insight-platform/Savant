#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_RUNTIME="--runtime=nvidia"
  docker compose -f samples/face_reid/docker-compose.l4t.yml --profile demo build
else
  DOCKER_RUNTIME="--gpus=all"
  docker compose -f samples/face_reid/docker-compose.x86.yml --profile demo build
fi

docker run --rm -it $DOCKER_RUNTIME \
  -v `pwd`/samples/face_reid/src:/opt/savant/samples/face_reid \
  -v `pwd`/samples/face_reid/index_files:/index \
  -v `pwd`/data:/data:ro \
  -v `pwd`/downloads/face_reid:/downloads \
  -v `pwd`/models/face_reid:/models \
  face_reid-module \
  samples/face_reid/module_performance.yml
