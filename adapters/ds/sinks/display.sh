#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

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

SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
CLOSING_DELAY="${CLOSING_DELAY:="0"}"

gst-launch-1.0 \
    savant_rs_video_player "${ZEROMQ_SRC_ARGS[@]}" \
    sync="${SYNC_OUTPUT}" closing-delay="${CLOSING_DELAY}"
