#!/bin/bash

required_env() {
    if [[ -z "${!1}" ]]; then
        echo "Environment variable ${1} not set"
        exit 1
    fi
}

required_env SOURCE_ID
required_env CAMERA_NAME
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
ENCODE="${ENCODE:="false"}"

arv-camera-test-0.8 --name="${CAMERA_NAME}" --duration 1
arv-tool-0.8 --name="${CAMERA_NAME}" control Width Height | tee resolution.txt
CAMERA_WIDTH="$(grep -E '^Width = ([0-9]+) .*' resolution.txt | sed -re 's/^Width = ([0-9]+) .*/\1/')"
CAMERA_HEIGHT="$(grep -E '^Height = ([0-9]+) .*' resolution.txt | sed -re 's/^Height = ([0-9]+) .*/\1/')"
if [[ -z "${CAMERA_WIDTH}" ]] || [[ -z "${CAMERA_HEIGHT}" ]]; then
    echo "Failed to get camera resolution"
    exit 1
fi
INPUT_CAPS="video/x-bayer,width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT}"

if [[ "${ENCODE,,}" == "true" ]]; then
    OUTPUT_CAPS="video/x-raw"
else
    OUTPUT_CAPS="video/x-raw,format=RGBA"
fi
if [[ -n "${WIDTH}" ]]; then OUTPUT_CAPS="${OUTPUT_CAPS},width=${WIDTH}"; fi
if [[ -n "${HEIGHT}" ]]; then OUTPUT_CAPS="${OUTPUT_CAPS},height=${HEIGHT}"; fi
if [[ -n "${FRAMERATE}" ]]; then OUTPUT_CAPS="${OUTPUT_CAPS},framerate=${FRAMERATE}"; fi

ADDITIONAL_ARAVISSRC_ARGS=()
if [[ -n "${AUTO_PACKET_SIZE}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("auto-packet-size=${AUTO_PACKET_SIZE}"); fi
if [[ -n "${PACKET_SIZE}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("packet-size=${PACKET_SIZE}"); fi
if [[ -n "${EXPOSURE}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("exposure=${EXPOSURE}"); fi
if [[ -n "${EXPOSURE_AUTO}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("exposure-auto=${EXPOSURE_AUTO}"); fi
if [[ -n "${GAIN}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("gain=${GAIN}"); fi
if [[ -n "${GAIN_AUTO}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("gain-auto=${GAIN_AUTO}"); fi
if [[ -n "${FEATURES}" ]]; then ADDITIONAL_ARAVISSRC_ARGS+=("features=${FEATURES}"); fi

USE_ABSOLUTE_TIMESTAMPS="${USE_ABSOLUTE_TIMESTAMPS:="false"}"
SINK_PROPERTIES=(
    source-id="${SOURCE_ID}"
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}"
)

PIPELINE=(
    aravissrc camera-name="${CAMERA_NAME}" "${ADDITIONAL_ARAVISSRC_ARGS[@]}" !
    capsfilter caps="${INPUT_CAPS}" !
    bayer2rgb !
    videoscale method=nearest-neighbour !
    capsfilter caps="${OUTPUT_CAPS}" !
)
if [[ "${ENCODE,,}" == "true" ]]; then
    ENCODE_BITRATE="${ENCODE_BITRATE:=2048}"
    ENCODE_KEY_INT_MAX="${ENCODE_KEY_INT_MAX:=30}"
    ENCODE_SPEED_PRESET="${ENCODE_SPEED_PRESET:=medium}"
    ENCODE_TUNE="${ENCODE_TUNE:=zerolatency}"
    PIPELINE+=(
        queue max-size-buffers=1 !
        videoconvert !
        x265enc bitrate="${ENCODE_BITRATE}" key-int-max="${ENCODE_KEY_INT_MAX}"
        speed-preset="${ENCODE_SPEED_PRESET}" tune="${ENCODE_TUNE}" !
        'video/x-h265,profile=main' !
        h265parse config-interval=-1 !
        'video/x-h265,stream-format=byte-stream,alignment=au' !
    )
fi
if [[ "${USE_ABSOLUTE_TIMESTAMPS,,}" == "true" ]]; then
    TS_OFFSET="$(date +%s%N)"
    if [[ "${ENCODE,,}" == "true" ]]; then
        # x265enc adds offset to timestamps to avoid negative timestamps
        TS_OFFSET="$((TS_OFFSET - 3600000000000000))"
    fi
    PIPELINE+=(
        shift_timestamps offset="${TS_OFFSET}" !
    )
    SINK_PROPERTIES+=(ts-offset="-${TS_OFFSET}")
fi
PIPELINE+=(
    queue max-size-buffers=1 !
    fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" !
    zeromq_sink "${SINK_PROPERTIES[@]}"
)

handler() {
    kill -s SIGINT "${child_pid}"
    wait "${child_pid}"
}
trap handler SIGINT SIGTERM

source "${PROJECT_PATH}/adapters/shared/utils.sh"
print_starting_message "gige-cam source adapter"
gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
