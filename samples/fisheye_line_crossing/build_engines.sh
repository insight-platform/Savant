#!/bin/bash
# you are expected to be in Savant/ directory

MODULE_CONFIG=samples/fisheye_line_crossing/module.yml

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/fisheye_line_crossing/docker-compose.l4t.yml build module
else
  docker compose -f samples/fisheye_line_crossing/docker-compose.x86.yml build module
fi

./scripts/run_module.py -i fisheye_line_crossing-module --build-engines $MODULE_CONFIG
