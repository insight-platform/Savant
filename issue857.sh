#!/usr/bin/env bash

docker rm -f issue857
docker run \
    -it \
    --name issue857 \
    --runtime nvidia \
    -v "$(pwd):/foo" \
    --entrypoint python3 \
    savant-adapters-deepstream-l4t \
    /foo/issue857.py
