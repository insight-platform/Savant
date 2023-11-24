Conversions Between GPU Memory Formats
---------------------------------------------

When working with images, there are many ways to represent them as arrays of pixels. Working with different models you may encounter representation of an image using OpenCV GpuMat class, PyTorch tensor or CuPy array.

The Savant framework aims to maximize GPU utilization without unnecessary data copying and conversion. To achieve this, Savant provides functions for converting between different image representations. Data exchange is performed with zero-copying between different views, except for some cases of conversion to GpuMat OpenCV.

Conversion to OpenCV
^^^^^^^^^^^^^^^^^^^^

**From PyTorch tensor**

.. py:currentmodule:: savant.utils.memory_repr_pytorch

:py:func:`pytorch_tensor_as_opencv_gpu_mat <pytorch_tensor_as_opencv_gpu_mat>` - the function allows you to convert a PyTorch tensor into an OpenCV GpuMat on GPU. The input tensor must be on GPU and must have shape in CHW format.

.. code-block:: python

    import torch
    import cv2
    from savant.utils.memory_repr_pytorch import pytorch_tensor_as_opencv_gpu_mat

    pytorch_tensor = torch.randint(0, 255, size=(channels, 10, 20), device='cuda').to(torch.uint8).cuda()
    opencv_gpu_mat = pytorch_tensor_as_opencv_gpu_mat(torch_tensor)


**From CuPy array**


.. py:currentmodule:: savant.utils.memory_repr

:py:func:`cupy_as_opencv_gpu_mat <cupy_as_opencv_gpu_mat>` - the function allows you to convert a CuPy array into an OpenCV GpuMat on GPU. The input array must have shape in HWC format.

.. code-block:: python

    import cupy as cp
    import cv2
    from savant.utils.memory_repr import cupy_as_opencv_gpu_mat

    cupy_array = cp.random.randint(0, 255, (10, 20, 3)).astype(cp.uint8)
    opencv_gpu_mat = cupy_as_opencv_gpu_mat(cupy_array)


Conversion to PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**From OpenCV GpuMat**

.. py:currentmodule:: savant.utils.memory_repr_pytorch

:py:func:`opencv_gpu_mat_as_pytorch <opencv_gpu_mat_as_pytorch>` - the function allows you to convert an OpenCV GpuMat into a PyTorch tensor on GPU.


.. code-block:: python

    import cv2
    from savant.utils.memory_repr_pytorch import opencv_gpu_mat_as_pytorch

    opencv_gpu_mat = cv2.cuda_GpuMat()
    opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, channels)).astype(input_type))

    torch_tensor = opencv_gpu_mat_as_pytorch(opencv_gpu_mat)


**From CuPy Array**

Conversion from CuPy array to PyTorch tensor is performed by using standard PyTorch function `torch.as_tensor <https://pytorch.org/docs/stable/generated/torch.as_tensor.html>`__. The function allows you to convert a CuPy array into a PyTorch tensor on GPU.

.. code-block:: python

    import cupy as cp
    import torch

    cupy_array = cp.random.randint(0, 255, (10, 20, 3)).astype(input_type)
    torch_tensor = torch.as_tensor(cupy_array)


Conversion to CuPy
^^^^^^^^^^^^^^^^^^

**From OpenCV GpuMat**

.. py:currentmodule:: savant.utils.memory_repr

:py:func:`opencv_gpu_mat_as_cupy <opencv_gpu_mat_as_cupy>` - the function allows you to convert an OpenCV GpuMat into a CuPy array on GPU.

.. code-block:: python

    import cv2
    import cupy as cp
    import numpy as np
    from savant.utils.memory_repr import opencv_gpu_mat_as_cupy

    opencv_gpu_mat = cv2.cuda_GpuMat()
    opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, 3)).astype(np.uint8))

    cupy_array = opencv_gpu_mat_as_cupy(opencv_gpu_mat)


**From PyTorch tensor**

Conversion from PyTorch tensor to CuPy is performed by using standard CuPy function `cupy.asarray <https://docs.cupy.dev/en/stable/reference/generated/cupy.asarray.html>`__ . The function allows you to convert PyTorch tensor to CuPy array on GPU.

.. code-block:: python

    import torch
    import cupy as cp

    torch_tensor = torch.randint(0, 255, size=(3, 10, 20), device='cuda').to(torch.uint8).cuda()
    cupy_array = cp.asarray(torch_tensor)