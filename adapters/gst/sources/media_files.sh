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
if [[ "${FILE_TYPE}" == "image" ]]; then
    MEASURE_PER_FILE=false
    EOS_ON_FILE_END="${EOS_ON_FILE_END:="false"}"
    EOS_ON_FRAME_PARAMS_CHANGE="${EOS_ON_FRAME_PARAMS_CHANGE:="true"}"
else
    MEASURE_PER_FILE=true
    EOS_ON_FILE_END="${EOS_ON_FILE_END:="true"}"
    EOS_ON_FRAME_PARAMS_CHANGE="${EOS_ON_FRAME_PARAMS_CHANGE:="true"}"
fi
SORT_BY_TIME="${SORT_BY_TIME:="false"}"
READ_METADATA="${READ_METADATA:="false"}"

USE_ABSOLUTE_TIMESTAMPS="${USE_ABSOLUTE_TIMESTAMPS:="false"}"
SINK_PROPERTIES=(
    source-id="${SOURCE_ID}"
    read-metadata="${READ_METADATA}"
    eos-on-file-end="${EOS_ON_FILE_END}"
    eos-on-frame-params-change="${EOS_ON_FRAME_PARAMS_CHANGE}"
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}"
)
if [[ -n "${RECEIVE_TIMEOUT_MSECS}" ]]; then
    SINK_PROPERTIES+=("receive-timeout=${RECEIVE_TIMEOUT_MSECS}")
else
    SINK_PROPERTIES+=("receive-timeout=5000")
fi
if [[ -n "${SHUTDOWN_AUTH}" ]]; then
    SINK_PROPERTIES+=(
        shutdown-auth="${SHUTDOWN_AUTH}"
    )
fi

PIPELINE=(
    media_files_src_bin location="${LOCATION}" file-type="${FILE_TYPE}" framerate="${FRAMERATE}" sort-by-time="${SORT_BY_TIME}" !
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" measure-per-file="${MEASURE_PER_FILE}" !
    adjust_timestamps !
)
if [[ "${USE_ABSOLUTE_TIMESTAMPS,,}" == "true" ]]; then
    TS_OFFSET="$(date +%s%N)"
    PIPELINE+=(
        shift_timestamps offset="${TS_OFFSET}" !
    )
    SINK_PROPERTIES+=(ts-offset="-${TS_OFFSET}")
fi
PIPELINE+=(
    set_dts !
    zeromq_sink "${SINK_PROPERTIES[@]}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

source "${PROJECT_PATH}/adapters/shared/utils.sh"
print_starting_message "media files source adapter"
gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
