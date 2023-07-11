#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    -v `pwd`/models/peoplenet_detector:/models \
    -v `pwd`/downloads/peoplenet_detector:/downloads \
    ghcr.io/insight-platform/savant-deepstream:latest \
    samples/conditional_video_processing/demo_performance.yml
else
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    -v `pwd`/models/peoplenet_detector:/models \
    -v `pwd`/downloads/peoplenet_detector:/downloads \
    ghcr.io/insight-platform/savant-deepstream-l4t:latest \
    samples/conditional_video_processing/demo_performance.yml
fi
