#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/traffic_meter/docker-compose.x86.yml build module
else
  docker compose -f samples/traffic_meter/docker-compose.l4t.yml build module
fi

docker run --rm -it --gpus=all \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/models/traffic_meter:/models \
  -v `pwd`/downloads/traffic_meter:/downloads \
  traffic_meter-module \
  samples/traffic_meter/module-yolov8m-performance.yml
