FROM python:3.10.12-slim

RUN --mount=type=bind,source=services/watchdog/requirements/common.txt,target=/tmp/requirements.txt \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY services/watchdog/watchdog .

ARG CYTHONIZE
RUN --mount=type=bind,source=utils/cythonize.sh,target=./cythonize.sh \
    if [[ -n "$CYTHONIZE" ]]; then ./cythonize.sh; fi

ENTRYPOINT ["python", "-m", "watchdog"]
