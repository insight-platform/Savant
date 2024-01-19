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
required_env FILE_TYPE

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="DEALER"}"
ZMQ_SOCKET_BIND="${ZMQ_BIND:="false"}"
SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
FRAMERATE="${FRAMERATE:="30/1"}"
FPS_OUTPUT="${FPS_OUTPUT:="stdout"}"
if [[ -n "${FPS_PERIOD_SECONDS}" ]]; then
    FPS_PERIOD="period-seconds=${FPS_PERIOD_SECONDS}"
elif [[ -n "${FPS_PERIOD_FRAMES}" ]]; then
    FPS_PERIOD="period-frames=${FPS_PERIOD_FRAMES}"
else
    FPS_PERIOD="period-frames=1000"
fi
if [[ -n "${RECEIVE_TIMEOUT_MSECS}" ]]; then
    RECEIVE_TIMEOUT="receive-timeout=${RECEIVE_TIMEOUT_MSECS}"
else
    RECEIVE_TIMEOUT="receive-timeout=5000"
fi
if [[ "${FILE_TYPE}" == "image" ]]; then
    MEASURE_PER_FILE=false
    EOS_ON_FILE_END="${EOS_ON_FILE_END:="false"}"
else
    MEASURE_PER_FILE=true
    EOS_ON_FILE_END="${EOS_ON_FILE_END:="true"}"
fi
SORT_BY_TIME="${SORT_BY_TIME:="false"}"
READ_METADATA="${READ_METADATA:="false"}"

SAVANT_RS_SERIALIZER_OPTS=(
    source-id="${SOURCE_ID}"
    read-metadata="${READ_METADATA}"
    eos-on-file-end="${EOS_ON_FILE_END}"
    eos-on-frame-params-change=true
)
if [[ -n "${SHUTDOWN_AUTH}" ]]; then
    SAVANT_RS_SERIALIZER_OPTS+=(
        shutdown-auth="${SHUTDOWN_AUTH}"
    )
fi


handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown \
    media_files_src_bin location="${LOCATION}" file-type="${FILE_TYPE}" framerate="${FRAMERATE}" sort-by-time="${SORT_BY_TIME}" ! \
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" measure-per-file="${MEASURE_PER_FILE}" ! \
    adjust_timestamps ! \
    savant_rs_serializer "${SAVANT_RS_SERIALIZER_OPTS[@]}" ! \
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}" "${RECEIVE_TIMEOUT}" \
    &

child_pid="$!"
wait "${child_pid}"
