RUNTIME="--gpus=all"
DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest

docker run --rm -it $RUNTIME \
  -e ZMQ_SRC_ENDPOINT=router+bind:ipc:///tmp/zmq-sockets/input-video.ipc \
  -e ZMQ_SINK_ENDPOINT=pub+bind:ipc:///tmp/zmq-sockets/output-video.ipc \
  -v /tmp/zmq-sockets:/tmp/zmq-sockets \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data:ro \
  $DOCKER_IMAGE \
  samples/memleak/module.yml
