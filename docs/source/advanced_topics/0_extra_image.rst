Module Image Types
==================

Savant provides two types of module images for both Jetson and X86+dGPU:

- basic image;
- extra-dependencies image.

The basic image is a minimal image that contains only the necessary components to run the module.

The extra-dependencies image contains additional dependencies that are not required for the module to run, but are useful in particular situations, for development and debugging: PyTorch, Torchvision, Torchaudio, TensorRT, Torch2TRT, ONNX, ONNX Runtime (GPU), PyCUDA, Cython, Pandas, Polars, Scikit-learn, JupyterLab.

Minimal images:

- ``savant-deepstream:<tag>``
- ``savant-deepstream-l4t:<tag>``

To use extra-dependencies image, you need add `-extra` suffix to the image name:

- ``savant-deepstream-extra:<tag>``
- ``savant-deepstream-l4t-extra:<tag>``

Extra component versions
------------------------

.. list-table:: Component Versions
    :header-rows: 1

    * - Component
      - x86 Version
      - Jetson Version
      - Notes
    * - PyTorch
      - 2.2.2
      - 2.1.0
      - With CUDA support
    * - Torchaudio
      - 2.2.2
      - 2.1.0
      -
    * - Torchvision
      - 0.17.2
      - 0.16.0
      -
    * - TensorRT
      - 8.6.1
      - 8.6.2
      -
    * - Torch2TRT
      - 0.4.0
      - 0.4.0
      -
    * - ONNX
      - 1.16.0
      - 1.16.0
      -
    * - ONNX Runtime
      - 1.17.1
      - 1.17.0
      - With CUDA support
    * - PyCUDA
      - 2024.1
      - 2024.1
      -
    * - Pandas
      - 2.2.1
      - 2.2.1
      -
    * - Polars
      - 0.20.18
      - 0.19.12
      -
    * - Scikit-learn
      - 1.4.1
      - 1.4.1
      -
    * - JupyterLab
      - 4.1.5
      - 4.1.5
      -
