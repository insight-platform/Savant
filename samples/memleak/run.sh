RUNTIME="--gpus=all"
# leak
DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest
# also leak
# DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:0.2.4-6.2

docker run --rm -it $RUNTIME \
  -e ZMQ_SRC_ENDPOINT=router+bind:ipc:///tmp/zmq-sockets/input-video.ipc \
  -e ZMQ_SINK_ENDPOINT=pub+bind:ipc:///tmp/zmq-sockets/output-video.ipc \
  -v /tmp/zmq-sockets:/tmp/zmq-sockets \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  $DOCKER_IMAGE \
  samples/memleak/module.yml
