# `Makefile` for local development.
# Use to build docker images for your platform, run docker command, and format your code.

SHELL := /bin/bash
SAVANT_VERSION := $(shell cat savant/VERSION | awk -F= '$$1=="SAVANT"{print $$2}' | sed 's/"//g')
DEEPSTREAM_VERSION := $(shell cat savant/VERSION | awk -F= '$$1=="DEEPSTREAM"{print $$2}' | sed 's/"//g')
DOCKER_FILE := Dockerfile.deepstream
PLATFORM := linux/amd64
ifeq ("$(shell uname -m)", "aarch64")
	PLATFORM := linux/arm64
endif
PLATFORM_SUFFIX :=
ifeq ("$(PLATFORM)", "linux/arm64")
    PLATFORM_SUFFIX := -l4t
endif

PROJECT_PATH := /opt/savant

build:
	docker buildx build \
		--platform $(PLATFORM) \
		--target base \
		--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
		-f docker/$(DOCKER_FILE) \
		-t savant-deepstream$(PLATFORM_SUFFIX) .
	#docker tag savant-deepstream$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-deepstream$(PLATFORM_SUFFIX)

build-adapters-deepstream:
	docker buildx build \
		--platform $(PLATFORM) \
		--target adapters \
		--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
		-f docker/$(DOCKER_FILE) \
		-t savant-adapters-deepstream$(PLATFORM_SUFFIX) .
	#docker tag savant-adapters-deepstream$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-deepstream$(PLATFORM_SUFFIX)

build-adapters-gstreamer:
	docker buildx build \
		--platform $(PLATFORM) \
		-f docker/Dockerfile.adapters-gstreamer \
		-t savant-adapters-gstreamer$(PLATFORM_SUFFIX) .
	#docker tag savant-adapters-gstreamer$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-gstreamer$(PLATFORM_SUFFIX)

build-adapters-py:
	docker buildx build \
		--platform $(PLATFORM) \
		-f docker/Dockerfile.adapters-py \
		-t savant-adapters-py$(PLATFORM_SUFFIX) .
	#docker tag savant-adapters-py$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-py$(PLATFORM_SUFFIX)

build-adapters-all: build-adapters-py build-adapters-gstreamer build-adapters-deepstream

build-docs:
	rm -rf docs/source/reference/api/generated
	docker buildx build \
		--target docs \
		--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
		--build-arg USER_UID=`id -u` \
		--build-arg USER_GID=`id -g` \
		-f docker/$(DOCKER_FILE) \
		-t savant-docs:$(SAVANT_VERSION) .

build-opencv: opencv-build-amd64 opencv-build-arm64 opencv-cp-amd64 opencv-cp-arm64

opencv-build-amd64:
	docker buildx build \
		--platform linux/amd64 \
		--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
		-f docker/Dockerfile.deepstream-opencv \
		-t savant-ds-opencv .

opencv-cp-amd64:
	docker run --rm \
		--platform linux/amd64 \
		-v `pwd`:/out \
		savant-ds-opencv

opencv-build-arm64:
	docker buildx build \
		--platform linux/arm64 \
		--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
		-f docker/Dockerfile.deepstream-opencv \
		-t savant-ds-opencv-l4t .

opencv-cp-arm64:
	docker run --rm \
		--platform linux/arm64 \
		-v `pwd`:/out \
		savant-ds-opencv-l4t

run-docs:
	docker run -it --rm \
		-v `pwd`/savant:$(PROJECT_PATH)/savant \
		-v `pwd`/docs:$(PROJECT_PATH)/docs \
		-v `pwd`/samples:$(PROJECT_PATH)/samples \
		--name savant-docs \
		savant-docs:$(SAVANT_VERSION)

run-dev:
	xhost +local:docker
	docker run -it --rm --gpus=all \
		--net=host --privileged \
		-e DISPLAY=$(DISPLAY) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-e ZMQ_SRC_ENDPOINT=router+bind:ipc:///tmp/zmq-sockets/input-video.ipc \
		-e ZMQ_SINK_ENDPOINT=pub+bind:ipc:///tmp/zmq-sockets/output-video.ipc \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp/.docker.xauth:/tmp/.docker.xauth \
		-v `pwd`/data:/data \
		-v `pwd`/downloads:/downloads \
		-v `pwd`/models:/models \
		-v `pwd`/gst_plugins:$(PROJECT_PATH)/gst_plugins \
		-v `pwd`/samples:$(PROJECT_PATH)/samples \
		-v `pwd`/savant:$(PROJECT_PATH)/savant \
		-v `pwd`/scripts:$(PROJECT_PATH)/scripts \
		-v `pwd`/var:$(PROJECT_PATH)/var \
		-v /tmp/zmq-sockets:/tmp/zmq-sockets \
		--entrypoint /bin/bash \
		savant-deepstream$(PLATFORM_SUFFIX)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} \+

check-black:
	black --check .

check-unify:
	unify --check-only --recursive savant | grep -- '--- before' | sed 's#--- before/##'
	unify --check-only --recursive savant > /dev/null

check: check-black check-unify check-isort

run-unify:
	unify --in-place --recursive savant adapters gst_plugins samples scripts

run-black:
	black .

reformat: run-unify run-black run-isort

check-isort:
	isort savant adapters gst_plugins samples scripts -c

run-isort:
	isort savant adapters gst_plugins samples scripts
