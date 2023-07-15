# `Makefile` for local development.
# Use to build and run module container for your platform.

# module/container name
MODULE_NAME := module

SHELL := /bin/bash
DOCKER_FILE := Dockerfile.x86
PROJECT_PATH := /opt/savant

ifeq ("$(shell uname -m)", "aarch64")
    DOCKER_FILE := Dockerfile.l4t
endif

# build module container
build:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/$(DOCKER_FILE) \
		-t $(MODULE_NAME) .

# run module
run:
	docker run -it --rm --gpus=all -v $(shell pwd)/modelcache:/models $(MODULE_NAME)

# run module in dev mode
run-dev:
	xhost +local:docker
	docker run -it --rm --gpus=all \
		--net=host --privileged \
		-e DISPLAY=$(DISPLAY) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp/.docker.xauth:/tmp/.docker.xauth \
		-v $(shell pwd)/modelcache:/models \
		-v $(shell pwd)/module:$(PROJECT_PATH)/module \
		--entrypoint /bin/bash \
		$(MODULE_NAME)
