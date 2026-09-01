#!/bin/sh
# Container probe for the stall_detector heartbeat file. Fails closed: healthy only
# if the file is fresh and its first line is exactly "running"/"starting".

HEALTH_FILEPATH="${PIPELINE_HEALTH_FILEPATH:="${PROJECT_PATH}/pipeline_health.txt"}"
MAX_AGE="${PIPELINE_HEALTH_MAX_AGE:="30"}"

# No file means the detector is not enabled here, so the probe must not fail it:
# this runs with the image's env and cannot see what the entrypoint exported.
if [ ! -f "${HEALTH_FILEPATH}" ]; then
    exit 0
fi

# A malformed max age must not make the freshness check pass silently.
case "${MAX_AGE}" in
    ''|*[!0-9]*)
        echo "PIPELINE_HEALTH_MAX_AGE \"${MAX_AGE}\" is not a whole number of seconds."
        exit 1
        ;;
esac

MTIME="$(stat -c %Y "${HEALTH_FILEPATH}")" || exit 1
AGE=$(($(date +%s) - MTIME))
if [ "${AGE}" -gt "${MAX_AGE}" ]; then
    echo "Health file ${HEALTH_FILEPATH} is ${AGE}s old, max age is ${MAX_AGE}s."
    exit 1
fi

STATUS="$(head -n 1 "${HEALTH_FILEPATH}")"
case "${STATUS}" in
    running|starting)
        exit 0
        ;;
    *)
        echo "Health file ${HEALTH_FILEPATH} reports status \"${STATUS}\"."
        exit 1
        ;;
esac
