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
ZEROMQ_SRC_ARGS=(
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
)
if [[ -n "${SOURCE_ID}" ]]; then
    ZEROMQ_SRC_ARGS+=(source-id="${SOURCE_ID}")
fi
if [[ -n "${SOURCE_ID_PREFIX}" ]]; then
    ZEROMQ_SRC_ARGS+=(source-id-prefix="${SOURCE_ID_PREFIX}")
fi

CHUNK_SIZE="${CHUNK_SIZE:=10000}"

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 \
    zeromq_src "${ZEROMQ_SRC_ARGS[@]}" ! \
    video_files_sink location="${DIR_LOCATION}" chunk-size="${CHUNK_SIZE}" \
    &

child_pid="$!"
wait "${child_pid}"
