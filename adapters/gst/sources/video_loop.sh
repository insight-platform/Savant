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
MEASURE_FPS_PER_LOOP="${MEASURE_FPS_PER_LOOP:="false"}"
EOS_ON_LOOP_END="${EOS_ON_LOOP_END:="false"}"
READ_METADATA="${READ_METADATA:="false"}"
USE_ABSOLUTE_TIMESTAMPS="${USE_ABSOLUTE_TIMESTAMPS:="false"}"
SINK_PROPERTIES=(
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}"
)

PIPELINE=(
    media_files_src_bin location="${LOCATION}" file-type=video loop-file=true download-path="${DOWNLOAD_PATH}" !
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" measure-per-loop="${MEASURE_FPS_PER_LOOP}" !
    adjust_timestamps !
)
if [[ "${USE_ABSOLUTE_TIMESTAMPS,,}" == "true" ]]; then
    TS_OFFSET="$(date +%s%N)"
    PIPELINE+=(
        shift_timestamps offset="${TS_OFFSET}" !
    )
    SINK_PROPERTIES+=(ts-offset="-${TS_OFFSET}")
fi
if [[ -n "${LOSS_RATE}" ]]; then
    PIPELINE+=(identity drop-probability="${LOSS_RATE}" !)
fi
PIPELINE+=(
    savant_rs_serializer source-id="${SOURCE_ID}" eos-on-loop-end="${EOS_ON_LOOP_END}"
    read-metadata="${READ_METADATA}" !
    zeromq_sink "${SINK_PROPERTIES[@]}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
