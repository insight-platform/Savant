Functions for converting image representation
---------------------------------------------

When working with images, there are many ways to represent them as arrays of points. Working with different models you may encounter representation of an image using OpenCV Mat class, PyTorch tensor or CuPy array.

The Savant framework aims to maximize GPU utilization without unnecessary data copying and conversion. To achieve this, Savant provides functions for converting between different image representations. Data exchange is performed with zero-copying between different views, except for some cases of conversion to GpuMat OpenCV.

Conversion to PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def as_pytorch(
        img: Union[cv2.cuda.GpuMat, cp.ndarray, np.ndarray],
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        device: Optional[str] = None,
    ):

This function allows you to convert an OpenCV or CuPy array into a PyTorch tensor.

- **img** - image in OpenCV format (``cv2.cuda.GpuMat`` for GPU or ``numpy.ndarray`` for CPU) or CuPy array (``cupy.ndarray``).

- **input_format** - input format of image shape. Can be one of the following values:

    - ``channels_first`` - the image is represented as a array of shape (channels, height, width).
    - ``channels_last`` - the image is represented as a array of shape (height, width, channels).
    - ``None`` - the input format is determined automatically based on the type of input array. If the input array is a GpuMat or CuPy, then the format is channels_last.

- **output_format** - Output format of image shape. Can be one of the following values:

    - ``channels_first`` - The image is represented as a array of shape (channels, height, width).
    - ``channels_last`` - The image is represented as a array of shape (height, width, channels).
    - ``None`` - The output format will be channels_last, since pytorch uses tensors in this format.

- **device** - Device on which the resulting tensor will be located. Can be one of the following values: ```cuda`` or ``cpu``. If the device is not specified, then the tensor will be located on the same device as the input array.


Conversion to OpenCV
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def as_opencv(
            img,
            device: Optional[str] = None,
            input_format: Optional[str] = None
    ) -> Union[cv2.cuda.GpuMat, np.ndarray]:

This function allows you to convert an image from PyTorch tensor or CuPy array to OpenCV format. The function always returns an OpenCV array in the `channels_last` format.

- **img** - image in PyTorch tensor format or CuPy array.
- **input_format** - Input format of image shape. Can be one of the following values:

    - ``channels_first`` - The image is represented as a array of shape (channels, height, width).
    - ``channels_last`` - The image is represented as a array of shape (height, width, channels).
    - ``None`` - The input format will be channels_first for pytorch, since pytorch uses tensors in this format, and channels_last for CuPy.

- **device** - Device on which the resulting image will be located. Can be one of the following values: ``cuda`` or ``cpu``. If the device is not specified, then the image will be located on the same device as the input array.

Conversion to CuPy
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def as_cupy(
        img,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> cp.ndarray:

This function allows you to convert an image from PyTorch tensor or OpenCV array to CuPy format. The function always returns an CuPy array on the ``cuda`` device.

- **img** - image in PyTorch tensor format or OpenCV array.
- **input_format** - Input format of image shape. Can be one of the following values:

    - ``channels_first`` - The image is represented as a array of shape (channels, height, width).
    - ``channels_last`` - The image is represented as a array of shape (height, width, channels).
    - ``None`` - The input format will be channels_first for pytorch, since pytorch uses tensors in this format, and channels_last for OpenCV.

- **output_format** - Output format of image shape. Can be one of the following values:

        - ``channels_first`` - The image is represented as a array of shape (channels, height, width).
        - ``channels_last`` - The image is represented as a array of shape (height, width, channels).
        - ``None`` - The output format will be channels_last.