#!/bin/bash
# you are expected to be in Savant/ directory

DOCKER_IMAGE=intersection_traffic_meter-module:latest
if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_RUNTIME="--runtime=nvidia"
  docker compose -f samples/intersection_traffic_meter/docker-compose.l4t.yml build module
else
  DOCKER_RUNTIME="--gpus=all"
  docker compose -f samples/intersection_traffic_meter/docker-compose.x86.yml build module
fi

docker run --rm -it $DOCKER_RUNTIME \
  -e BUFFER_QUEUES \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/models/intersection_traffic_meter:/models \
  -v `pwd`/downloads/intersection_traffic_meter:/downloads \
  $DOCKER_IMAGE \
  samples/intersection_traffic_meter/module_performance.yml
