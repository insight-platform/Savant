#!/bin/bash
# you are expected to be in Savant/ directory

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/peoplenet_detector/docker-compose.l4t.yml build module
else
  docker compose -f samples/peoplenet_detector/docker-compose.x86.yml build module
fi

docker run --rm -it --gpus=all \
 -v `pwd`/samples:/opt/savant/samples \
 -v `pwd`/data:/data:ro \
 -v `pwd`/models/peoplenet_detector:/models \
 -v `pwd`/downloads/peoplenet_detector:/downloads \
 peoplenet_detector-module \
 samples/peoplenet_detector/demo_performance.yml
