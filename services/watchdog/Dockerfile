FROM python:3.10.12-slim

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=services/watchdog/requirements/common.txt,target=/tmp/requirements.txt \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY services/watchdog/watchdog watchdog

ARG CYTHONIZE
RUN --mount=type=bind,source=utils/cythonize.sh,target=./cythonize.sh \
    if [ -n "$CYTHONIZE" ]; then ./cythonize.sh; fi

ENTRYPOINT ["python", "-m", "watchdog"]
