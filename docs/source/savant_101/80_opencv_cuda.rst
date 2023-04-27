OpenCV CUDA Usage
=================

Sometimes video analytics pipelines need to modify frames to preprocess pictures before sending them to models, like making affine transformations for detected faces before sending them to facial models; or displaying auxiliary dashboard information on a frame to highlight valuable data.

Savant provides the user with advanced instruments to access and modify video frames located in the GPU memory with Python and OpenCV CUDA functionality. Raw frames allocated in the GPU memory occupy a significant amount of RAM. E.g. for an RGBA frame with a resolution of 1280x720, the amount is larger than 3.6 MB.

Copying them to the CPU and back introduces additional latency, which decreases the performance. The naive approach requires two transmissions: from GPU to CPU and back. OpenCV CUDA extensions provide highly optimized mechanisms to work with frames directly in GPU. However, such functions are limited in comparison with OpenCV CPU functionality. Three approaches can be used when working with in-GPU frames:

1. Run CUDA-accelerated algorithms supported with OpenCV CUDA on in-GPU frames. This is the most efficient approach that can be used when working with filters or segmentation techniques.

2. Run algorithms on sprites allocated in CPU RAM, then upload and apply them on the in-GPU frame using alpha-channelized overlaying. This approach is beneficial if you need to modify parts of the image without downloading the modified part to the CPU. E.g., to apply a bounding box or a textual label to an object.

3. Download the parts of the frame required for modification to CPU RAM, modify them, and upload them to the same place without using the alpha channel. This is the least efficient approach; however, if the changes are localized, you can still get a significant performance improvement compared to downloading and uploading the entire frame.

All the above-mentioned strategies can be implemented with the API provided by Savant. To further improve the performance of such operations, you should consider using asynchronous Nvidia CUDA API based on streams. Streams enable sending the operations into a non-blocking executor while the Python code can handle the next operation. You wait for the CUDA stream to complete background processing at the end of the frame processing.

Direct access to frames on the GPU is achieved by calling the ``nvds_to_gpu_mat`` helper function, for example:

.. code-block:: python

    from savant.deepstream.opencv_utils import nvds_to_gpu_mat
    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
        # frame_mat is a cv2.cuda.GpuMat


The variable ``frame_mat`` acting inside the context in the example will refer to the memory of the frame being processed and will have the type ``cv2.cuda.GpuMat``. The work with the frame will be carried out through OpenCV CUDA methods, for example, you can apply Gaussian blur to a certain area of the frame:

.. code-block:: python

    gaussian_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC4, cv2.CV_8UC4, (9, 9), 2
    )
    roi = cv2.cuda_GpuMat(frame_mat, (0, 0, 100, 100))
    gaussian_filter.apply(roi , roi)


You can read more about the capabilities of OpenCV CUDA in the OpenCV `documentation <https://docs.opencv.org/4.7.0/d1/d1a/namespacecv_1_1cuda.html>`__.

