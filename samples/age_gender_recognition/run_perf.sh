#!/bin/bash
# you are expected to be in Savant/ directory

DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest
DOCKER_RUNTIME="--gpus=all"
if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream-l4t:latest
  DOCKER_RUNTIME="--runtime=nvidia"
fi

docker run --rm -it $DOCKER_RUNTIME \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v `pwd`/downloads/age_gender_recognition:/downloads \
  -v `pwd`/models/age_gender_recognition:/models \
  $DOCKER_IMAGE \
  samples/age_gender_recognition/module_performance.yml
