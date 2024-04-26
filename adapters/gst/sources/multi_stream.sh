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

MEDIA_FILES_SRC_BIN_OPTS=(
    location="${LOCATION}"
    file-type=video
)
if [[ -n "${DOWNLOAD_PATH}" ]]; then
    MEDIA_FILES_SRC_BIN_OPTS+=(
        download-path="${DOWNLOAD_PATH}"
    )
fi


USE_ABSOLUTE_TIMESTAMPS="${USE_ABSOLUTE_TIMESTAMPS:="false"}"
SINK_PROPERTIES=(
    source-id="${SOURCE_ID}"
    read-metadata="${READ_METADATA}"
    enable-multistream=true
    number-of-streams="${NUMBER_OF_STREAMS}"
    socket="${ZMQ_ENDPOINT}"
    socket-type="${ZMQ_SOCKET_TYPE}"
    bind="${ZMQ_SOCKET_BIND}"
    sync="${SYNC_OUTPUT}"
)
if [[ -n "${RECEIVE_TIMEOUT}" ]]; then
    SINK_PROPERTIES+=("receive-timeout=${RECEIVE_TIMEOUT}")
fi
if [[ -n "${SOURCE_ID_PATTERN}" ]]; then
    SINK_PROPERTIES+=(
        source-id-pattern="${SOURCE_ID_PATTERN}"
    )
fi
if [[ -n "${SHUTDOWN_AUTH}" ]]; then
    SINK_PROPERTIES+=(
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
print_starting_message "multistream source adapter"
gst-launch-1.0 --eos-on-shutdown "${PIPELINE[@]}" &

child_pid="$!"
wait "${child_pid}"
