RUNTIME="--gpus=all"
# leak
DOCKER_IMAGE=ghcr.io/insight-platform/savant-deepstream:latest


docker run --rm -it $RUNTIME \
  -v `pwd`/samples:/opt/savant/samples \
  -v `pwd`/data:/data \
  --entrypoint python \
  $DOCKER_IMAGE \
  samples/memleak/stream_pool_script.py
  # samples/memleak/memleak_script.py


