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

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="REQ"}"
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
if [[ "${FILE_TYPE}" == "picture" ]]; then
    MEASURE_PER_FILE=false
    EOS_ON_LOCATION_CHANGE=false
else
    MEASURE_PER_FILE=true
    EOS_ON_LOCATION_CHANGE=true
fi
SORT_BY_TIME="${SORT_BY_TIME:="false"}"

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT

gst-launch-1.0 --eos-on-shutdown \
    media_files_src_bin location="${LOCATION}" file-type="${FILE_TYPE}" framerate="${FRAMERATE}" sort-by-time="${SORT_BY_TIME}" ! \
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" measure-per-file="${MEASURE_PER_FILE}" ! \
    adjust_timestamps ! \
    video_to_avro_serializer source-id="${SOURCE_ID}" eos-on-location-change="${EOS_ON_LOCATION_CHANGE}" eos-on-frame-params-change=true ! \
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}" \
    &

child_pid="$!"
wait "${child_pid}"
