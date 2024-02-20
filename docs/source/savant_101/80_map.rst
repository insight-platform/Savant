Frame Memory Mapping
====================

.. warning::
    You must avoid changing frame dimensions and memory layout with the current functionality. Doing so will result in a memory error.

Frame Mapping is a method to access the frame data using NumPy. This approach can be less efficient than :doc:`technique <80_opencv_cuda>` based on OpenCV CUDA. However, it is beneficial if you need to process the whole frame on the CPU with NumPy-compatible tools. Remember that the whole RGBA frame will be copied to the CPU and back to the GPU, which is inefficient.

The API to access the frame with NumPy is provided with the utility function ``get_nvds_buf_surface``:

.. code-block:: python

    from savant.deepstream.utils.surface import get_nvds_buf_surface
    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        with get_nvds_buf_surface(buffer, frame_meta.frame_meta) as frame_mat:
            # frame_mat is a numpy.ndarray

As a result, the ``frame_mat`` variable will have the ``numpy.ndarray`` type and contain the RGBA array of the frame being processed. Upon exiting the context, the changes made to the array will be saved in the frame. However, it is worth considering that the restrictions specified in the DeepStream `documentation <https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/Methods/methodsdoc.html#get-nvds-buf-surface>`__ apply.
