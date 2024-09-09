#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env SOURCE_ID
required_env URI
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
BUFFER_LEN="${BUFFER_LEN:="50"}"
FFMPEG_LOGLEVEL="${FFMPEG_LOGLEVEL:="info"}"
USE_ABSOLUTE_TIMESTAMPS="${USE_ABSOLUTE_TIMESTAMPS:="false"}"
SINK_PROPERTIES=(
    source-id="${SOURCE_ID}"
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}"
    ts-offset="${SYNC_DELAY}"
)

FFMPEG_SRC=(ffmpeg_src uri="${URI}" queue-len="${BUFFER_LEN}" loglevel="${FFMPEG_LOGLEVEL}")
if [[ -n "${FFMPEG_PARAMS}" ]]; then
    FFMPEG_SRC+=("params=${FFMPEG_PARAMS}")
fi
if [[ -n "${FFMPEG_TIMEOUT_MS}" ]]; then
    FFMPEG_SRC+=("timeout-ms=${FFMPEG_TIMEOUT_MS}")
fi
PIPELINE=(
    "${FFMPEG_SRC[@]}" !
    savant_parse_bin !
)
if [[ "${USE_ABSOLUTE_TIMESTAMPS,,}" == "true" ]]; then
    TS_OFFSET="$(date +%s%N)"
    PIPELINE+=(
        shift_timestamps offset="${TS_OFFSET}" !
    )
    SYNC_DELAY="$((SYNC_DELAY - TS_OFFSET))"
fi
PIPELINE+=(
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" !
    zeromq_sink "${SINK_PROPERTIES[@]}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

source "${PROJECT_PATH}/adapters/shared/utils.sh"
print_starting_message "ffmpeg source adapter"
gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
