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

SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
CLOSING_DELAY="${CLOSING_DELAY:="0"}"

gst-launch-1.0 \
    zeromq_src socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" ! \
    avro_video_player sync="${SYNC_OUTPUT}" closing-delay="${CLOSING_DELAY}"
