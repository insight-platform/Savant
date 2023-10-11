#!/bin/bash
# Usage: source-switch.sh [on/off] [source-id]

state=${1:-off}
source=${2:-city-traffic}

docker exec -it etcd etcdctl put savant/sources/"$source" "$state"
