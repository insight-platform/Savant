#!/usr/bin/env bash
# Set/update the per-source converter configuration in Etcd.
#
# Usage: set-config.sh <source-id> <json-config>
#   e.g. ./set-config.sh city-traffic '{"confidence_threshold": 0.2}'
#        ./set-config.sh town-centre  '{"confidence_threshold": 0.7, "nms_iou_threshold": 0.5}'
#
# The real Etcd key is "<watch_path>/source/<source-id>",
# where watch_path is "savant" (see module.yml).

source=${1:-city-traffic}
config=${2:-'{"confidence_threshold": 0.25}'}

docker exec -it etcd etcdctl put "savant/source/$source" "$config"
