#!/bin/bash

set -x

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env SOURCE_ID
required_env LOCATION
required_env ZMQ_ENDPOINT
required_env DOWNLOAD_PATH

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="DEALER"}"
ZMQ_SOCKET_BIND="${ZMQ_BIND:="false"}"
SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
FPS_OUTPUT="${FPS_OUTPUT:="stdout"}"
if [[ -n "${FPS_PERIOD_SECONDS}" ]]; then
    FPS_PERIOD="period-seconds=${FPS_PERIOD_SECONDS}"
elif [[ -n "${FPS_PERIOD_FRAMES}" ]]; then
    FPS_PERIOD="period-frames=${FPS_PERIOD_FRAMES}"
else
    FPS_PERIOD="period-frames=1000"
fi
READ_METADATA="${READ_METADATA:="false"}"
NUMBER_OF_SOURCES="${NUMBER_OF_SOURCES:=1}"

SAVANT_RS_SERIALIZER_OPTS=(
    source-id="${SOURCE_ID}"
    read-metadata="${READ_METADATA}"
    enable-multistream=true
    number-of-sources="${NUMBER_OF_SOURCES}"
)
if [[ -n "${SHUTDOWN_AUTH}" ]]; then
    SAVANT_RS_SERIALIZER_OPTS+=(
        shutdown-auth="${SHUTDOWN_AUTH}"
    )
fi

PIPELINE=(
    media_files_src_bin location="${LOCATION}" file-type=video download-path="${DOWNLOAD_PATH}" !
)
if [[ -n "${NUMBER_OF_FRAMES}" ]]; then
    # Identity drops the last frame when eos-after is set.
    PIPELINE+=(
        identity eos-after="$((NUMBER_OF_FRAMES + 1))" !
    )
fi
PIPELINE+=(
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" !
    adjust_timestamps !
    savant_rs_serializer "${SAVANT_RS_SERIALIZER_OPTS[@]}" !
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
