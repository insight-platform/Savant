Using DLA on Nvidia Jetson
--------------------------

Nvidia Jetson devices provide dedicated deep learning acceleration units (DLA) allowing using them to run convolutional neural networks on a dedicated hardware. DLA work completely independent of the GPU and can be used to run a separate network in parallel to the GPU. This allows to run particular networks in parallel on the DLA and GPU, which can be beneficial for performance.

However, DLA have their own constraints which are described in Nvidia documentation and usually slower than the GPU; therefore, they are beneficial for low-demand tasks and mostly suit for running small networks in secondary inference tasks.

To use DLA, you need to specify the target device in the YAML `config`. Use the following syntax for a inference block.

.. code-block:: yaml

    - element: nvinfer@detector
      name: yolo
      model:
        format: onnx
        model_file: yolo.onnx
        precision: int8
        int8_calib_file: calib.bin
        batch_size: 16
        enable_dla: true   # allocate this model on DLA
        use_dla_core: 1    # use DLA core 1
        input:
          ...
        output:
          ...

