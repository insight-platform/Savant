#!/bin/bash

set -x

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env LOCATION
required_env ZMQ_ENDPOINT

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
NUMBER_OF_STREAMS="${NUMBER_OF_STREAMS:=1}"

if [[ -n "${RECEIVE_TIMEOUT}" ]]; then
    SENDER_RECEIVE_TIMEOUT="receive-timeout=${RECEIVE_TIMEOUT}"
else
    SENDER_RECEIVE_TIMEOUT=
fi

MEDIA_FILES_SRC_BIN_OPTS=(
    location="${LOCATION}"
    file-type=video
)
if [[ -n "${DOWNLOAD_PATH}" ]]; then
    MEDIA_FILES_SRC_BIN_OPTS+=(
        download-path="${DOWNLOAD_PATH}"
    )
fi

SAVANT_RS_SERIALIZER_OPTS=(
    source-id="${SOURCE_ID}"
    read-metadata="${READ_METADATA}"
    enable-multistream=true
    number-of-streams="${NUMBER_OF_STREAMS}"
)
if [[ -n "${SOURCE_ID_PATTERN}" ]]; then
    SAVANT_RS_SERIALIZER_OPTS+=(
        source-id-pattern="${SOURCE_ID_PATTERN}"
    )
fi
if [[ -n "${SHUTDOWN_AUTH}" ]]; then
    SAVANT_RS_SERIALIZER_OPTS+=(
        shutdown-auth="${SHUTDOWN_AUTH}"
    )
fi

PIPELINE=(
    media_files_src_bin "${MEDIA_FILES_SRC_BIN_OPTS[@]}" !
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
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}" "${SENDER_RECEIVE_TIMEOUT}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
