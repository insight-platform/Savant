Additional NvInfer Parameters
=============================

In this section, we will describe other NvInfer model parameters that are not covered in the previous sections. The list is not exhaustive, but covers the most frequently used parameters.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Default Value
     - Required
     - Example

   * - ``model.mean_file``
     - Pathname of mean data file in PPM format.
     - None
     - No
     - ``model.mean_file: mean.ppm``

   * - ``model.label_file``
     - Pathname of a text file containing the labels for the model.
     - None
     - No
     - ``model.label_file: labels.txt``

   * - ``model.tlt_model_key``    
     - Key for the TAO toolkit encoded model.
     - None
     - No
     - ``model.tlt_model_key: tlt_encode``

   * - ``model.gpu_id``
     - Device ID of GPU to use for pre-processing/inference (dGPU only).
     - 0
     - No
     - ``model.gpu_id: 0``

   * - ``model.interval``
     - Specifies the number of consecutive batches to be skipped for inference. We recommend do not use it but reset the top-level ROI instead if the frame must be skipped.
     - 0
     - No
     - ``model.interval: 0``

   * - ``model.workspace_size``
     - Workspace size to be used by the engine, in MB. The parameter defines the amount of GPU memory that TensorRT will use for the engine building.    
     - 6144
     - No
     - ``model.workspace_size: 6144``

   * - ``model.custom_lib_path``
     - Absolute pathname of a library containing custom method implementations for custom models.
     - None
     - No
     - ``model.custom_lib_path: /opt/savant/lib/libnvdsinfer_custom_impl_Yolo.so``

   * - ``model.engine_create_func_name``
     - Name of the custom TensorRT CudaEngine creation function.
     - None
     - No
     - ``model.engine_create_func_name: NvDsInferYoloCudaEngineGet``

   * - ``model.layer_device_precision``
     - Specifies the device type and precision for any layer in the network. List of items of format ``<layer1-name>:<precision>:<device-type>``.
     - None
     - No
     - ``model.layer_device_precision: [conv1:fp16:gpu, conv2:int8:gpu]``

   * - ``model.enable_dla``
     - Specifies that DLA engines to be used for inference. Only for Nvidia Jetson devices.
     - None
     - No
     - ``model.enable_dla: true``

   * - ``model.use_dla_core``
     - Specifies the DLA core to be used for inference. Only for Nvidia Jetson devices.
     - 0
     - No
     - ``model.use_dla_core: 1``

   * - ``model.scaling_compute_hw``
     - Specifies the hardware to be used for scaling compute. Mostly used for Nvidia Jetson devices.
     - AUTO (other options are GPU and VIC)
     - No
     - ``model.scaling_compute_hw: AUTO``

   * - ``model.scaling_filter``
     - Specifies the algorithm to be used for scaling.
     - AUTO (other options are NEAREST, BILINEAR, GPU_CUBIC, VIC_5_TAP, GPU_SUPER, VIC_10_TAP, GPU_LANCZOS, VIC_SMART, GPU_IGNORED, VIC_NICEST)
     - No
     - ``model.scaling_filter: AUTO``

   * - ``model.input.layer_name``
     - Specifies the name of the input layer.
     - None
     - No
     - ``model.input.layer_name: input_1``

   * - ``model.input.shape``
     - Specifies the shape of the input layer.
     - None
     - No
     - ``model.input.shape: [3, 544, 960]``

   * - ``model.input.maintain_aspect_ratio``
     - Specifies whether the input preprocessing should maintain image aspect ratio.
     - False
     - No
     - ``model.input.maintain_aspect_ratio: True``

   * - ``model.input.symmetric_padding``
     - Specifies whether the input preprocessing should symmetrically pad the image when it's scaled.
     - False
     - No
     - ``model.input.symmetric_padding: True``

   * - ``model.input.scale_factor``
     - Specifies the input preprocessing scale factor.
     - 1.0
     - No
     - ``model.input.scale_factor: 0.0039215697906911373``

   * - ``model.input.offsets``
     - Specifies the input preprocessing offsets.
     - [0.0, 0.0, 0.0]
     - No
     - ``model.input.offsets: [0.0, 0.0, 0.0]``

   * - ``model.input.color_format``
     - Specifies the input preprocessing color format.
     - RGB (other options are BGR and GRAY)
     - No
     - ``model.input.color_format: RGB``

   * - ``model.input.object_min_width``
     - Specifies the minimum width of the object to be processed.
     - None
     - No
     - ``model.input.object_min_width: 32``

   * - ``model.input.object_min_height``
     - Specifies the minimum height of the object to be processed.
     - None
     - No
     - ``model.input.object_min_height: 32``

   * - ``model.input.object_max_width``
     - Specifies the maximum width of the object to be processed.
     - None
     - No
     - ``model.input.object_max_width: 1024``

   * - ``model.input.object_max_height``
     - Specifies the maximum height of the object to be processed.
     - None
     - No
     - ``model.input.object_max_height: 1024``