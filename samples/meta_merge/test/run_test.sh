#!/bin/bash
# Run the meta_merge test: source -> infer_blackbox -> sink
set -e
cd "$(dirname "$0")"

# Create test image if missing
if [ ! -f test_image.jpeg ]; then
    echo "Creating test image..."
    pip install -q Pillow 2>/dev/null || true
    python create_test_image.py
fi

echo "Building and running test..."
docker compose -f docker-compose.test.yml build
docker compose -f docker-compose.test.yml up --abort-on-container-exit

# Check sink exit code (0 = pass)
SINK_EXIT=$(docker compose -f docker-compose.test.yml ps -q sink 2>/dev/null | xargs docker inspect -f '{{.State.ExitCode}}' 2>/dev/null || echo "1")
if [ "$SINK_EXIT" = "0" ]; then
    echo "Test PASSED"
    exit 0
else
    echo "Test FAILED (sink exit code: $SINK_EXIT)"
    exit 1
fi
