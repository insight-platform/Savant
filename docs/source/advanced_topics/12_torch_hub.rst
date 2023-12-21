TorchHub
--------

`TorchHub <https://pytorch.org/hub/>`__ is a tool for publishing and using pre-trained models. This approach is not applicable for production use, but using TorchHub and Savant you can test models on real video streams or videos.

The `official documentation <https://pytorch.org/docs/stable/hub.html#loading-models-from-hub>`__  describes how to load the model and perform the inference.

With Savant it is easy to run Pytorch inference on a frame or for an object of interest. Look at the `panoptic_driving_perception <https://github.com/insight-platform/Savant/tree/develop/samples/panoptic_driving_perception>`__ sample to learn how. Note how OpenCV GPUMat <-> Pytorch GPU tensor conversion function (:ref:`advanced_topics/11_memory_representation_function:Conversions Between GPU Memory Formats`) makes it possible to get the image as a Pytorch GPU tensor and perform the necessary preprocessing exactly as in a strictly Pytorch pipeline, while avoiding copying data to the host memory and back to the GPU.

.. note::

    Be careful when installing dependencies and additional packages. Savant docker container comes with OpenCV with CUDA support preinstalled, and using other OpenCV versions may cause errors and make it impossible to run the code.