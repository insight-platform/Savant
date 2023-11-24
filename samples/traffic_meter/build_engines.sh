#!/bin/bash
# you are expected to be in Savant/ directory

MODULE_CONFIG=samples/traffic_meter/module.yml

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/traffic_meter/docker-compose.l4t.yml build module
else
  docker compose -f samples/traffic_meter/docker-compose.x86.yml build module
fi

./scripts/run_module.py -i traffic_meter-module --build-engines $MODULE_CONFIG
