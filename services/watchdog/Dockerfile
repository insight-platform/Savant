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

RUN --mount=type=bind,source=services/watchdog/watchdog,target=/tmp/watchdog \
    --mount=type=bind,source=utils/copy_code.sh,target=/tmp/copy_code.sh \
    /tmp/copy_code.sh /tmp/watchdog /app/watchdog

ENTRYPOINT ["python", "-m", "watchdog"]
