Conversions Between GPU Memory Formats
--------------------------------------

When working with images, there are many ways to represent them as arrays of pixels. Working with different models you may encounter representation of an image using OpenCV GpuMat class, PyTorch tensor or CuPy array.

The Savant framework aims to use GPU efficiently without excessive data transfers. To achieve this, Savant provides functions for converting between different image representations. Data exchange is performed with zero-copying between different views, except for some cases of conversion to GpuMat OpenCV.

Conversion to OpenCV
^^^^^^^^^^^^^^^^^^^^

**From PyTorch tensor**

.. py:currentmodule:: savant.utils.memory_repr_pytorch

:py:func:`pytorch_tensor_as_opencv_gpu_mat <pytorch_tensor_as_opencv_gpu_mat>` allows you to convert a PyTorch tensor into an OpenCV GpuMat. The input tensor must be on GPU, must have shape in HWC format and be in C-contiguous layout.

.. code-block:: python

    import torch
    from savant.utils.memory_repr_pytorch import pytorch_tensor_as_opencv_gpu_mat

    # original in HWC
    pytorch_tensor = torch.randint(0, 255, size=(10, 20, 3), device='cuda').to(torch.uint8)
    # map to opencv gpu mat (zero-copy)
    opencv_gpu_mat = pytorch_tensor_as_opencv_gpu_mat(torch_tensor)

If the shape format of the tensor is different, you can transform it into the required format using e.g. `Tensor.permute() <https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html>`__. You should keep in mind that such transformations usually lead to data copying and additionally require the tensor to be converted to contiguous in memory layout. You can do this with `Tensor.contiguous() <https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html>`__.

.. code-block:: python

    import torch
    from savant.utils.memory_repr_pytorch import pytorch_tensor_as_opencv_gpu_mat

    # original in CHW
    tensor0 = torch.randint(0, 255, size=(3, 10, 20), device='cuda').to(torch.uint8)
    # transform to HWC
    tensor1 = tensor0.permute(1, 2, 0)
    # to contiguous (copy)
    tensor2 = tensor1.contiguous()
    # map to opencv gpu mat (zero-copy)
    gpu_mat = pytorch_tensor_as_opencv_gpu_mat(tensor2)

**From CuPy array**

.. py:currentmodule:: savant.utils.memory_repr

:py:func:`cupy_array_as_opencv_gpu_mat <cupy_array_as_opencv_gpu_mat>` allows you to convert a CuPy array into an OpenCV GpuMat. The input array must have shape in HWC format, 2 or 3 dimensions and be in C-contiguous layout.

.. code-block:: python

    import cupy as cp
    from savant.utils.memory_repr import cupy_array_as_opencv_gpu_mat

    # original in HWC
    cupy_array = cp.random.randint(0, 255, (10, 20, 3)).astype(cp.uint8)
    # map to opencv gpu mat (zero-copy)
    opencv_gpu_mat = cupy_array_as_opencv_gpu_mat(cupy_array)

If the shape format of the array is different, you can transform it into the required format using e.g. `cupy.transpose() <https://docs.cupy.dev/en/stable/reference/generated/cupy.transpose.html>`__. You should keep in mind that such transformations usually lead to data copying and additionally require the array to be converted to contiguous in memory layout. You can do this with `cupy.ascontiguousarray() <https://docs.cupy.dev/en/stable/reference/generated/cupy.ascontiguousarray.html>`__.

.. code-block:: python

    import cupy as cp
    from savant.utils.memory_repr import cupy_array_as_opencv_gpu_mat

    # original in CHW
    arr0 = cp.random.randint(0, 255, (3, 10, 20)).astype(cp.uint8)
    # transform to HWC
    arr1 = arr0.transpose((1, 2, 0))
    # to contiguous (copy)
    arr2 = cp.ascontiguousarray(arr1)
    # map to opencv gpu mat (zero-copy)
    gpu_mat = cupy_array_as_opencv_gpu_mat(arr2)


Conversion to PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**From OpenCV GpuMat**

.. py:currentmodule:: savant.utils.memory_repr_pytorch

:py:func:`opencv_gpu_mat_as_pytorch_tensor <opencv_gpu_mat_as_pytorch_tensor>` allows you to convert an OpenCV GpuMat into a PyTorch tensor on GPU.


.. code-block:: python

    import cv2
    from savant.utils.memory_repr_pytorch import opencv_gpu_mat_as_pytorch_tensor

    opencv_gpu_mat = cv2.cuda_GpuMat()
    opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, 3)).astype(np.uint8))
    # zero-copy, HWC format
    torch_tensor = opencv_gpu_mat_as_pytorch_tensor(opencv_gpu_mat)


**From CuPy Array**

Conversion from CuPy array to PyTorch tensor is performed by using standard PyTorch function `torch.as_tensor <https://pytorch.org/docs/stable/generated/torch.as_tensor.html>`__.

.. code-block:: python

    import cupy as cp
    import torch

    cupy_array = cp.random.randint(0, 255, (10, 20, 3)).astype(cp.uint8)
    # zero-copy, original array format
    torch_tensor = torch.as_tensor(cupy_array)


Conversion to CuPy Array
^^^^^^^^^^^^^^^^^^^^^^^^

**From OpenCV GpuMat**

.. py:currentmodule:: savant.utils.memory_repr

:py:func:`opencv_gpu_mat_as_cupy_array <opencv_gpu_mat_as_cupy_array>` allows you to convert an OpenCV GpuMat into a CuPy array.

.. code-block:: python

    import cv2
    import cupy as cp
    import numpy as np
    from savant.utils.memory_repr import opencv_gpu_mat_as_cupy_array

    opencv_gpu_mat = cv2.cuda_GpuMat()
    opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, 3)).astype(np.uint8))
    # zero-copy, HWC format
    cupy_array = opencv_gpu_mat_as_cupy_array(opencv_gpu_mat)


**From PyTorch tensor**

Conversion from PyTorch tensor to CuPy is performed by using standard CuPy function `cupy.asarray <https://docs.cupy.dev/en/stable/reference/generated/cupy.asarray.html>`__ .

.. code-block:: python

    import torch
    import cupy as cp

    torch_tensor = torch.randint(0, 255, size=(3, 10, 20), device='cuda').to(torch.uint8)
    # zero-copy, original tensor format
    cupy_array = cp.asarray(torch_tensor)
