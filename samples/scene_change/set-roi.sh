#!/bin/bash
# Change the region of interest (ROI) for a source at runtime.
# Usage: set-roi.sh SOURCE_ID [ROI]
#   ROI format: "left,top,right,bottom"
#   If ROI is omitted, the key is removed (the source falls back to the default ROI).

if [ -z "$1" ]; then
    echo "Usage: set-roi.sh SOURCE_ID [ROI]" >&2
    exit 1
fi

SOURCE_ID=$1
ROI=$2

if [ -z "${ROI}" ]; then
    docker exec -it etcd etcdctl del savant/roi/"${SOURCE_ID}"
else
    docker exec -it etcd etcdctl put savant/roi/"${SOURCE_ID}" "${ROI}"
fi
