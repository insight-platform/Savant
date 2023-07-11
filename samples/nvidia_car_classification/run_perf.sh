#!/bin/bash
# you are expected to be in Savant/ directory

DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest
if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream-l4t:latest
fi

docker run --rm -it --gpus=all \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/models/nvidia_car_classification:/models \
  -v `pwd`/downloads/nvidia_car_classification:/downloads \
  $DOCKER_IMAGE \
  samples/nvidia_car_classification/module_performance.yml
