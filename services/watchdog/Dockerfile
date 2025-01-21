FROM python:3.10.12-slim

COPY requirements/common.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf requirements.txt

WORKDIR /app
COPY watchdog/ watchdog/

ENTRYPOINT ["python", "-m", "watchdog.run"]
