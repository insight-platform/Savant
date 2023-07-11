#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    ghcr.io/insight-platform/savant-deepstream-l4t:latest \
    samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
else
  docker run --rm -it --gpus=all \
    -v `pwd`/samples:/opt/savant/samples \
    -v `pwd`/data:/data:ro \
    ghcr.io/insight-platform/savant-deepstream:latest \
    samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
fi
