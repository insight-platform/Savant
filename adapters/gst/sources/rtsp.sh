#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env SOURCE_ID
required_env RTSP_URI
required_env ZMQ_ENDPOINT

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="DEALER"}"
ZMQ_SOCKET_BIND="${ZMQ_BIND:="false"}"
SYNC_OUTPUT="${SYNC_OUTPUT:="false"}"
if [[ -n "${SYNC_DELAY}" ]]; then
    # Seconds to nanoseconds
    SYNC_DELAY=$((SYNC_DELAY * 1000000000))
else
    SYNC_DELAY=0
fi
FPS_OUTPUT="${FPS_OUTPUT:="stdout"}"
if [[ -n "${FPS_PERIOD_SECONDS}" ]]; then
    FPS_PERIOD="period-seconds=${FPS_PERIOD_SECONDS}"
elif [[ -n "${FPS_PERIOD_FRAMES}" ]]; then
    FPS_PERIOD="period-frames=${FPS_PERIOD_FRAMES}"
else
    FPS_PERIOD="period-frames=1000"
fi
RTSP_TRANSPORT="${RTSP_TRANSPORT:="tcp"}"
BUFFER_LEN="${BUFFER_LEN:="50"}"
FFMPEG_LOGLEVEL="${FFMPEG_LOGLEVEL:="info"}"

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

PIPELINE=(
    ffmpeg_src uri="${RTSP_URI}" params="rtsp_transport=${RTSP_TRANSPORT}"
    queue-len="${BUFFER_LEN}" loglevel="${FFMPEG_LOGLEVEL}" !
    savant_parse_bin !
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" !
    savant_rs_serializer source-id="${SOURCE_ID}" !
    zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}" ts-offset="${SYNC_DELAY}"
)

gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
