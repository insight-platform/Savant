#!/bin/bash
# Builds pycuda from source (amd64/arm64).

: "${PYCUDA_VERSION:=v2024.1}"
: "${OUTPUT_DIR:=/opt}"
: "${TMP_DIR:=/tmp}"

cd $TMP_DIR || exit 1
git clone --branch=${PYCUDA_VERSION} --depth=1 --recursive https://github.com/inducer/pycuda
cd pycuda || exit 1

./configure.py

python3 setup.py build_ext --inplace bdist_wheel
cp dist/pycuda*.whl "$OUTPUT_DIR"

rm -rf $TMP_DIR/pycuda
