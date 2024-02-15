Utilities
=========

General utilities
-----------------

.. automodule:: savant.utils

.. currentmodule:: savant.utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    registry.Registry
    image.CPUImage
    image.GPUImage
    artist.Position
    artist.Artist
    logging.LoggerMixin

GPU Memory Formats
------------------

.. currentmodule:: savant.utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/function.rst

    memory_repr.opencv_gpu_mat_as_cupy_array
    memory_repr_pytorch.opencv_gpu_mat_as_pytorch_tensor
    memory_repr.cupy_array_as_opencv_gpu_mat
    memory_repr_pytorch.pytorch_tensor_as_opencv_gpu_mat

DeepStream utilities
--------------------

.. currentmodule:: savant.deepstream.utils.surface

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/function.rst

    get_nvds_buf_surface

.. currentmodule:: savant.deepstream.nvinfer.build_engine

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/function.rst

    build_engine

OpenCV utilities
--------------------

.. currentmodule:: savant.deepstream.opencv_utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/function.rst

    nvds_to_gpu_mat
    alpha_comp
    apply_cuda_filter
    draw_rect
