#!/usr/bin/env bash

# Usage: ./run_benchmarks.sh [FRAME_NUM]

if [[ -n "${1}" ]]; then
    FRAME_NUM="${1}"
else
    FRAME_NUM=1000
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
    DOCKER_IMAGE="savant-deepstream-l4t:0.2.0-6.2-base"
else
    DOCKER_IMAGE="savant-deepstream:0.2.0-6.2-base"
fi

GPU_BENCHMARK_NAMES=(
    "overlay"
    "overlay-single"
    "draw-rectangles"
    "blur-faces"
    "blur-faces-parallel"
    "blur-faces-single-stream"
    "blur-faces-in-cpu"
    "download-upload"
)
CPU_BENCHMARK_NAMES=(
    "overlay"
    "draw-rectangles"
    "blur-faces"
)

echo "name,device,frame_num,min,max,mean,median,80%,90%,95%,99%,stdev" >metrics.csv

for BENCHMARK_NAME in "${GPU_BENCHMARK_NAMES[@]}"; do
    echo
    date
    echo "Running GPU benchmark ${BENCHMARK_NAME}"
    docker run \
        --name test \
        --rm -it \
        --gpus all \
        -e GST_DEBUG=1 \
        -e LOGLEVEL=INFO \
        -e PYTHONUNBUFFERED=1 \
        --workdir /benchmarks \
        --entrypoint ./benchmark.py \
        -v "$(pwd):/benchmarks" \
        "${DOCKER_IMAGE}" "${BENCHMARK_NAME}" "gpu" "${FRAME_NUM}"
done

for BENCHMARK_NAME in "${CPU_BENCHMARK_NAMES[@]}"; do
    echo
    date
    echo "Running CPU benchmark ${BENCHMARK_NAME}"
    docker run \
        --name test \
        --rm -it \
        --gpus all \
        -e GST_DEBUG=1 \
        -e LOGLEVEL=INFO \
        -e PYTHONUNBUFFERED=1 \
        --workdir /benchmarks \
        --entrypoint ./benchmark.py \
        -v "$(pwd):/benchmarks" \
        "${DOCKER_IMAGE}" "${BENCHMARK_NAME}" "cpu" "${FRAME_NUM}"
done
date
