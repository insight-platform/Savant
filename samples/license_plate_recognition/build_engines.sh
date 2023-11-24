#!/bin/bash
# you are expected to be in Savant/ directory

MODULE_CONFIG=samples/license_plate_recognition/module.yml

if [ "$(uname -m)" = "aarch64" ]; then
  docker compose -f samples/license_plate_recognition/docker-compose.l4t.yml build module
else
  docker compose -f samples/license_plate_recognition/docker-compose.x86.yml build module
fi

./scripts/run_module.py -i license_plate_recognition-module --build-engines $MODULE_CONFIG
