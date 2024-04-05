#!/bin/bash
# you are expected to be in Savant/ directory

MODULE_CONFIG=samples/rtdetr/module.yml

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/rtdetr/docker-compose.l4t.yml build module
else
  docker compose -f samples/rtdetr/docker-compose.x86.yml build module
fi

./scripts/run_module.py -i rtdetr-module --build-engines $MODULE_CONFIG
