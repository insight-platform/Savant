Conversions Between GPU Memory Formats
---------------------------------------------

When working with images, there are many ways to represent them as arrays of pixels. Working with different models you may encounter representation of an image using OpenCV Mat class, PyTorch tensor or CuPy array.

The Savant framework aims to maximize GPU utilization without unnecessary data copying and conversion. To achieve this, Savant provides functions for converting between different image representations. Data exchange is performed with zero-copying between different views, except for some cases of conversion to GpuMat OpenCV.

Conversion to PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def opencv_gpu_mat_as_pytorch(gpu_mat: cv2.cuda.GpuMat) -> torch.Tensor:
            """Returns PyTorch tensor for specified OpenCV GpuMat."""

This function allows you to convert an OpenCV GpuMat into a PyTorch tensor on GPU.

Conversion to OpenCV
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def pytorch_tensor_as_opencv_gpu_mat(tensor: torch.Tensor) -> cv2.cuda_GpuMat:
        """Returns OpenCV GpuMat for specified PyTorch tensor.
        Supports 2 and 3 dims arrays in CHW format for shape and `channel_first`
        memory format only.
        """

This function allows you to convert an image from PyTorch tensor to OpenCV GpuMat. The input tensor must be on GPU and must have shape in CHW format. Also note that the function support only ``channel_first`` memory format.


.. code-block:: python

    def cupy_as_opencv_gpu_mat(arr: cp.ndarray) -> cv2.cuda.GpuMat:
        """Returns OpenCV GpuMat for specified CuPy ndarray.
        Supports 2 and 3 dims arrays in HWC format for shape and `channel_last`
        memory format only. (OpenCV format).
        """

This function allows you to convert an image from CuPy array to OpenCV GpuMat. The input array have shape in HWC format. Also note that the function support only ``channel_last`` memory format.

Conversion to CuPy
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def opencv_gpu_mat_as_cupy(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
        """Returns CuPy ndarray for specified OpenCV GpuMat."""

This function allows you to convert an image OpenCV GpuMat to CuPy array.