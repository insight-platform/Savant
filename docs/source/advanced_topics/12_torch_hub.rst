TorchHub
--------

`TorchHub <https://pytorch.org/hub/>`__ is a tool for publishing and using pre-trained models. This approach is not applicable for production use, but using TorchHub and Savant you can test models on real video streams or videos.

The `official documentation <https://pytorch.org/docs/stable/hub.html#loading-models-from-hub>`__  describes how to load the model and perform the inference you can.

Using Savant you can easily make an inference on a frame or for an object of interest (`example <https://github.com/insight-platform/Savant/tree/develop/samples/panoptic_driving_perception>`__. Also using the conversion function (:ref:`advanced_topics/11_memory_representation_function:Conversions Between GPU Memory Formats`), you can get the image as a pytorch tensor on GPU and perform the necessary preprocessing. This will avoid copying data to the host memory and back to the GPU.

.. note::
    Be careful when installing dependencies necessary for the model to work, because explicitly or implicitly a version of OpenCV without CUDA support may be installed in the Savant docker container. This will cause errors and make it impossible to run the code.