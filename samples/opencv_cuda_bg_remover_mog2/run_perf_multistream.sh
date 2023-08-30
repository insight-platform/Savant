#!/bin/bash
# you are expected to be in Savant/ directory

DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest
SOURCE_ADAPTER_DOCKER_IMAGE=ghcr.io/insight-platform/savant-adapters-gstreamer:latest
DOCKER_RUNTIME="--gpus=all"
if [ "$(uname -m)" = "aarch64" ]; then
  DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream-l4t:latest
  SOURCE_ADAPTER_DOCKER_IMAGE=ghcr.io/insight-platform/savant-adapters-gstreamer-l4t:latest
  DOCKER_RUNTIME="--runtime=nvidia"
fi

SOURCE_ADAPTER_CONTAINER=$(docker run --rm -d \
  --entrypoint /opt/savant/adapters/gst/sources/multi_stream.sh \
  -e LOCATION=/data/road_traffic.mp4 \
  -e NUMBER_OF_STREAMS=4 \
  -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/opencv_cuda_bg_remover_mog2/input-video.ipc \
  -e SHUTDOWN_AUTH=opencv_cuda_bg_remover_mog2 \
  -e SYNC_OUTPUT=False \
  -v `pwd`/data:/data:ro \
  -v /tmp/zmq-sockets/opencv_cuda_bg_remover_mog2:/tmp/zmq-sockets/opencv_cuda_bg_remover_mog2 \
  $SOURCE_ADAPTER_DOCKER_IMAGE)

sleep 5

docker run --rm -it $DOCKER_RUNTIME \
  -e BUFFER_QUEUES \
  -e ZMQ_SRC_ENDPOINT=router+bind:ipc:///tmp/zmq-sockets/opencv_cuda_bg_remover_mog2/input-video.ipc \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  -v /tmp/zmq-sockets/opencv_cuda_bg_remover_mog2:/tmp/zmq-sockets/opencv_cuda_bg_remover_mog2 \
  $DOCKER_IMAGE \
  samples/opencv_cuda_bg_remover_mog2/demo_multistream_performance.yml
EXIT_CODE=$?
docker kill "${SOURCE_ADAPTER_CONTAINER}" >/dev/null 2>/dev/null
exit $EXIT_CODE
