# `Makefile` for local development.
# Use to build docker images for your platform, run docker command, and format your code.

SHELL := /bin/bash
SAVANT_VERSION := $(shell cat savant/VERSION | awk -F= '$$1=="SAVANT"{print $$2}' | sed 's/"//g')
DEEPSTREAM_VERSION := $(shell cat savant/VERSION | awk -F= '$$1=="DEEPSTREAM"{print $$2}' | sed 's/"//g')
DOCKER_FILE := Dockerfile.deepstream
PLATFORM_SUFFIX :=
PROJECT_PATH := /opt/savant

ifeq ("$(shell uname -m)", "aarch64")
    PLATFORM_SUFFIX := -l4t
    DOCKER_FILE := Dockerfile.deepstream-l4t
endif

build:
	DOCKER_BUILDKIT=1 docker build \
	--target base \
	--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
	-f docker/$(DOCKER_FILE) \
	-t savant-deepstream$(PLATFORM_SUFFIX) \
	-t savant-deepstream$(PLATFORM_SUFFIX):$(SAVANT_VERSION)-$(DEEPSTREAM_VERSION) .
	#docker tag savant-deepstream$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-deepstream$(PLATFORM_SUFFIX)

build-adapters-deepstream:
	DOCKER_BUILDKIT=1 docker build \
	--target adapters \
	--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
	-f docker/$(DOCKER_FILE) \
	-t savant-adapters-deepstream$(PLATFORM_SUFFIX) \
	-t savant-adapters-deepstream$(PLATFORM_SUFFIX):$(SAVANT_VERSION)-$(DEEPSTREAM_VERSION) .
	#docker tag savant-adapters-deepstream$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-deepstream$(PLATFORM_SUFFIX)

build-adapters-gstreamer:
	DOCKER_BUILDKIT=1 docker build \
	-f docker/Dockerfile.adapters-gstreamer \
	-t savant-adapters-gstreamer$(PLATFORM_SUFFIX) \
	-t savant-adapters-gstreamer$(PLATFORM_SUFFIX):$(SAVANT_VERSION) .
	#docker tag savant-adapters-gstreamer$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-gstreamer$(PLATFORM_SUFFIX)

build-adapters-py:
	DOCKER_BUILDKIT=1 docker build \
	-f docker/Dockerfile.adapters-py \
	-t savant-adapters-py$(PLATFORM_SUFFIX) \
	-t savant-adapters-py$(PLATFORM_SUFFIX):$(SAVANT_VERSION) .
	#docker tag savant-adapters-py$(PLATFORM_SUFFIX) ghcr.io/insight-platform/savant-adapters-py$(PLATFORM_SUFFIX)

build-adapters-all: build-adapters-py build-adapters-gstreamer build-adapters-deepstream

build-docs:
	DOCKER_BUILDKIT=1 docker build \
	--target docs \
	--build-arg DEEPSTREAM_VERSION=$(DEEPSTREAM_VERSION) \
	--build-arg USER_UID=`id -u` \
	--build-arg USER_GID=`id -g` \
	-f docker/$(DOCKER_FILE) \
	-t savant-docs:$(SAVANT_VERSION) .

run-docs:
	docker run -it --rm \
		-v `pwd`/savant:$(PROJECT_PATH)/savant \
		-v `pwd`/docs:$(PROJECT_PATH)/docs \
		--name savant-docs \
		savant-docs:$(SAVANT_VERSION)

run-dev:
	xhost +local:docker
	docker run -it --rm --gpus=all \
		--net=host --privileged \
		-e DISPLAY=$(DISPLAY) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp/.docker.xauth:/tmp/.docker.xauth \
		-v `pwd`/var:$(PROJECT_PATH)/var \
		-v `pwd`/samples:$(PROJECT_PATH)/samples \
		-v `pwd`/gst_plugins:$(PROJECT_PATH)/gst_plugins \
		-v `pwd`/savant:$(PROJECT_PATH)/savant \
		--entrypoint /bin/bash \
		savant-deepstream$(PLATFORM_SUFFIX)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} \+

check-black:
	black --check .

check-unify:
	unify --check-only --recursive savant | grep -- '--- before' | sed 's#--- before/##'
	unify --check-only --recursive savant > /dev/null

check: check-black check-unify

run-unify:
	unify --in-place --recursive savant

run-black:
	black .

reformat: run-unify run-black
