#!/bin/bash

state=${1:-off}
source=${2:-city-traffic}

echo "savant/sources/$source $state"
#docker exec -it etcd etcdctl put savant/sources/"$source" "$state"
