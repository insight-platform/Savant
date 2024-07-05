#!/bin/bash
# Builds torch2trt from source (amd64/arm64).
# Requires nvidia runtime to share some host libs with the container on Jetson.

: "${TORCH2TRT_VERSION:=v0.5.0}"
: "${OUTPUT_DIR:=/opt}"
: "${TMP_DIR:=/tmp}"

cd $TMP_DIR || exit 1
git clone --branch=${TORCH2TRT_VERSION} --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt || exit 1

# patch for python >= 3.10
sed 's|collections.Sequence|collections.abc.Sequence|g' -i torch2trt/converters/interpolate.py
#cat torch2trt/converters/interpolate.py | grep Sequence

# install requirements
python3 -m pip install tensorrt~=8.6 torch packaging

python3 setup.py bdist_wheel
cp dist/torch2trt*.whl "$OUTPUT_DIR"

if [ "$(uname -m)" = "aarch64" ]; then
  # Orin
  CUDA_ARCHITECTURES=87
  sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt
  cmake -B build -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} .
else
  cmake -B build .
fi
cmake --build build
cp build/libtorch2trt_plugins.so "$OUTPUT_DIR"

rm -rf $TMP_DIR/torch2trt
