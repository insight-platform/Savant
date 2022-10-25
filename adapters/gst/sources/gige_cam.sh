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

ZMQ_SOCKET_TYPE="${ZMQ_TYPE:="REQ"}"
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
INPUT_CAPS="${INPUT_CAPS:="ANY"}"

OUTPUT_CAPS="video/x-raw,format=RGBA"
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

handler() {
  kill -s SIGINT "${child_pid}"
  wait "${child_pid}"
}
trap handler SIGINT SIGTERM

gst-launch-1.0 --eos-on-shutdown \
  aravissrc camera-name="${CAMERA_NAME}" "${ADDITIONAL_ARAVISSRC_ARGS[@]}" ! \
  capsfilter caps="${INPUT_CAPS}" ! \
  videoconvert ! \
  capsfilter caps="${OUTPUT_CAPS}" ! \
  fps_meter "${FPS_PERIOD}" output="${FPS_OUTPUT}" ! \
  video_to_avro_serializer source-id="${SOURCE_ID}" ! \
  zeromq_sink socket="${ZMQ_ENDPOINT}" socket-type="${ZMQ_SOCKET_TYPE}" bind="${ZMQ_SOCKET_BIND}" sync="${SYNC_OUTPUT}" \
  &

child_pid="$!"
wait "${child_pid}"
