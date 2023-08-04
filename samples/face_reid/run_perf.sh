#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/face_reid/docker-compose.l4t.yml --profile demo build
else
  docker compose -f samples/face_reid/docker-compose.x86.yml --profile demo build
fi

docker run --rm -it --gpus=all \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/downloads/face_reid:/downloads \
  -v `pwd`/models/face_reid:/models \
  face_reid-module \
  samples/face_reid/module_performance.yml
