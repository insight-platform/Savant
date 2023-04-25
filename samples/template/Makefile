# `Makefile` for local development.
# Use to develop and test your module, format module code,
# build and run module container for your platform.
# TODO: describe commands

# module/container name
MODULE_NAME := module

SHELL := /bin/bash
DOCKER_FILE := Dockerfile.x86
PROJECT_PATH := /opt/savant

ifeq ("$(shell uname -m)", "aarch64")
    PLATFORM_SUFFIX := -l4t
    DOCKER_FILE := Dockerfile.deepstream-l4t
endif

build:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/$(DOCKER_FILE) \
		-t $(MODULE_NAME) .

run:
	docker run -it --rm --gpus=all $(MODULE_NAME)

run-dev:
	xhost +local:docker
	docker run -it --rm --gpus=all \
		--net=host --privileged \
		-e DISPLAY=$(DISPLAY) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp/.docker.xauth:/tmp/.docker.xauth \
		-v `pwd`/module:$(PROJECT_PATH)/module \
		--entrypoint /bin/bash \
		$(MODULE_NAME)

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
