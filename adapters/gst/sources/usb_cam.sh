#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env SOURCE_ID
required_env DEVICE
required_env ZMQ_ENDPOINT

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="DEALER"}"
ZMQ_SOCKET_BIND="${ZMQ_BIND:="false"}"
SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
FRAMERATE="${FRAMERATE:="15/1"}"
FPS_OUTPUT="${FPS_OUTPUT:="stdout"}"
if [[ -n "${FPS_PERIOD_SECONDS}" ]]; then
    FPS_PERIOD="period-seconds=${FPS_PERIOD_SECONDS}"
elif [[ -n "${FPS_PERIOD_FRAMES}" ]]; then
    FPS_PERIOD="period-frames=${FPS_PERIOD_FRAMES}"
else
    FPS_PERIOD="period-frames=1000"
fi

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown \
    v4l2src device="${DEVICE}" ! \
    "video/x-raw,framerate=${FRAMERATE}" ! \
    autovideoconvert ! \
    'video/x-raw,format=RGBA' ! \
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" ! \
    savant_rs_serializer source-id="${SOURCE_ID}" ! \
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}" \
    &

child_pid="$!"
wait "${child_pid}"
