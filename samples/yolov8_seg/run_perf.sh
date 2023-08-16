#!/bin/bash
# you are expected to be in Savant/ directory
RUNTIME="--gpus=all"
DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest
if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream-l4t:latest
  RUNTIME="--runtime=nvidia"
fi

docker run --rm -it $RUNTIME \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/downloads/yolov8_seg:/downloads \
  -v `pwd`/models/yolov8_seg:/models \
  $DOCKER_IMAGE \
  samples/yolov8_seg/module/module_performance.yml
