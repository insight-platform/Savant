Module Image Types
==================

Savant provides two types of module images for both Jetson and X86+dGPU:

- basic image;
- extra-dependencies image.

The basic image is a minimal image that contains only the necessary components to run the module.

The extra-dependencies image contains additional dependencies that are not required for the module to run, but are useful in particular situations, for development and debugging: PyTorch, Torchvision, Torchaudio, TensorRT, Torch2TRT, ONNX, Onnx Runtime (GPU), PyCUDA, Cython, Pandas, Polars, Scikit-learn, JupyterLab.

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
      - Version
      - Notes
    * - PyTorch
      - 2.4
      - With CUDA support
