Batching
========

Batching is an important function affecting the performance of the pipeline. Inference units utilize batching to process images more efficiently. Primary models (which operate on the whole image) consider the batch as a number of frames simultaneously passed to the model. Secondary models consider the batch as a number of objects simultaneously passed to the model.

Savant uses two batching mechanisms:

- batching of video streams;
- batching of objects sent to models for inference;

Stream Multiplexing Batching
----------------------------

In Savant, a batch consists of the frames from streams processed. It is important to note that the batch can include multiple frames from a single stream, so if a pipeline processes N streams, then the batch size can be from 1 to N (ideally N), but not necessarily all streams will be presented in every batch; also, a batch is not always full: when streams delay the frames, the batch may be gathered partially, decreasing the efficiency of the processing. Thus, usually, you want to maximize batch filling.

Depending on the first model architecture, increasing the batch size may not improve performance. In this case, you may stay safe using an average batch size, which results in optimal performance (not necessary to increase it to the theoretical number of parallel streams processed).

.. tip::

    E.g., you expect the pipeline to handle up to 16 streams; you also discovered that the model saturates its performance with a batch size of 4. Therefore, you should use four as a pipeline batch size.

The pipeline configuration file defines parameters regulating the batch size. They are ``batch_size`` and ``batched_push_timeout``.

The ``batch_size`` parameter is configured for video multiplexing. When it is **not** configured, it is set based on the value defined for the first pipeline model. The default value is ``1``.

The ``batched_push_timeout`` defines the maximum time the pipeline waits for the batch to be fully formed, pushing out partially formed batches if the wait time exceeds this threshold. The smaller value decreases latency, while the larger one results in greater throughput. The default value is: ``4000000`` (``4`` seconds).

.. tip::

    Typically, when handling the real-time streams at 30 FPS, you need to set ``batched_push_timeout`` to ``35000-40000`` (``35-40`` ms). However, when injecting video from file sources, keep it as low as ``1000`` (``1`` ms).


.. code-block:: yaml

    # base module parameters
    parameters:
      batch_size: 1
      batched_push_timeout: 40000


Model Batching
--------------

Models operate on objects detected by previous models or defined with ROIs (externally defined pseudo-objects). Usually, there are several objects in every frame, and there is a batch of frames. Thus, predicting how many objects you get after the previous model may be difficult.

However, the rule of thumb is to set the batch size to a number between one and an average number of objects in the frame. You may experiment with various batch sizes to tweak the performance.

.. code-block:: yaml

    - element: nvinfer@classifier
      name: Secondary_CarColor
      model:
        batch_size: 16

