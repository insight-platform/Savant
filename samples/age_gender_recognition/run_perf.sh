#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    -v `pwd`/downloads:/downloads \
    -v `pwd`/models/age_gender_recognition:/models \
    ghcr.io/insight-platform/savant-deepstream:latest \
    samples/age_gender_recognition/module_performance.yml
else
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    -v `pwd`/downloads:/downloads \
    -v `pwd`/models/age_gender_recognition:/models \
    ghcr.io/insight-platform/savant-deepstream-l4t:latest \
    samples/age_gender_recognition/module_performance.yml
fi
