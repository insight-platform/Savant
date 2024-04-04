#!/bin/bash
# Builds extra packages.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${DIR}/build_pycuda.sh"
bash "${DIR}/build_torch2trt.sh"
