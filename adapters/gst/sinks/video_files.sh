#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env DIR_LOCATION
required_env ZMQ_ENDPOINT

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="SUB"}"
ZMQ_SOCKET_BIND="${ZMQ_BIND:="false"}"
CHUNK_SIZE="${CHUNK_SIZE:=10000}"

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 \
    zeromq_src socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" ! \
    video_files_sink location="${DIR_LOCATION}" chunk-size="${CHUNK_SIZE}" \
    &

child_pid="$!"
wait "${child_pid}"
