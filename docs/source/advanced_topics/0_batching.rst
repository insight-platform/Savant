Batching
========

Batching is an important function influencing the performance of the pipeline a lot. Inference units utilize batching to process images more efficiently. Primary models (which operate on the whole image) consider the batch as a number of frames simultaneously passed to the model. Secondary models consider the batch as a number of frames simultaneously passed to the model.

Primary Batching
----------------

In Savant, the primary batch consists of the frames from streams processed. It is important to note that the batch cannot include multiple frames from a single stream, so if a pipeline processes N streams, then the batch size can be from 1 to N (ideally N); however, the batch is not always complete: when streams delay the frames, the batch can be gathered partially, decreasing the efficiency of the processing. Thus, usually, you want to maximize batch filling.

Increasing the batch size may not improve performance depending on the primary model. In this case, you may stay safe using the minimal batch size, which results in optimal performance (not necessary to increase it to the theoretical number of parallel streams processed). E.g., you expect the pipeline to handle up to 16 streams; you also discovered that the model saturates its performance with a batch size of 4. Therefore, you should use four as a pipeline batch size.

The pipeline configuration file has parameters that regulate the batch size. They are ``batch_size`` and ``batched_push_timeout``.

The ``batch_size`` parameter may be configured for the first inference element or globally. However, the same value must be configured for the first inference element when it is configured globally. The global configuration is usually necessary when the pipeline doesn't include inference elements; however, it benefits from batching somehow. E.g., if the code uses asynchronous cuda operations, batched processing may result in better performance. However, usually, you must specify the ``batch_size`` parameter for the first inference element. By default, the value for the parameter is set to ``1``.

The ``batched_push_timeout`` defines how long the pipeline waits until the batch is full. The smaller value decreases latency, while the larger one results in greater processing speed. Typically, when handling the real-time streams at 30 FPS, you need to set ``batched_push_timeout`` to ``35000-40000`` (``35-40`` ms). However, when injecting video from file sources, keep it as low as ``1000000`` (``1`` ms).

Default value is: ``4000000`` (``4`` seconds).

.. code-block:: yaml

    # base module parameters
    parameters:
      ...
      batch_size: 1
      batched_push_timeout: 40000


Secondary Batching
------------------

Secondary models operate on objects detected by primary models. Usually, there are several objects in every frame, and there is a batch of frames. Thus, predicting how many objects you get after the primary model may be difficult. However, the rule of thumb is to select the batch size typically set to a number between one and an average number of objects in the frame. You may experiment with various batch sizes to tweak the performance.

.. code-block:: yaml

    - element: nvinfer@classifier
      name: Secondary_CarColor
      model:
        ...
        batch_size: 16

