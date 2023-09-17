Benchmarking And Optimization
=============================

When a pipeline is ready, developers often need to measure its performance. There are three types of benchmarks developers want to run:

- non-real-time single stream performance;
- non-real-time multiple stream performance;
- real-time multiple stream performance.

There is one more dimension to benchmark:

- with adapters (end-to-end);
- without adapters (only pipeline).

When benchmarking, we usually measure the aggregate throughput of the pipeline in FPS (frames-per-second) it can handle. Sometimes, usually, when the pipeline is intended to work in real-time, we also measure end-to-end latency, but this document does not discuss it.

.. tip::

    Remember, that GPU/CPU overheating significantly hits the performance. Plan the benchmarks to address the situation. Read our `article <https://betterprogramming.pub/real-time-video-analytics-challenges-and-approaches-to-surpass-them-b07793192649?source=friends_link&sk=10ff3e46cc2ea642f7c8d4da1c91ce9a>`_ on Medium to find out more.


.. warning::

    Benchmarking in the shared cloud may produce results changing greatly from launch to launch: as you do not control all the host resources, other users may influence a lot on the results. We recommend running benchmarks on dedicated bare-metal servers when possible.

Measuring Non-Real-Time Single Stream Performance
-------------------------------------------------

The benchmark allows an understanding of the performance of the pipeline in a batch mode. A typical case is file-by-file processing, when you start the pipeline ingesting a file into it and want to measure how fast the file can be processed. This sort of processing is often used when processing archived videos.

To measure the performance, use the ``uridecodebin`` Gstreamer source:


.. _isolated_uridecodebin_benchmark:

.. code-block::

    pipeline:
      # local video file source, not using source adapter
      source:
        element: uridecodebin
        properties:
          uri: file:///data/file.mp4

      # define pipeline's main elements
      elements:
        ...

      # noop pipeline sink, not using sink adapter
      sink:
        - element: devnull_sink

At the end of operation, you will see the FPS result.

Measuring Non-Real-Time Multiple Stream Performance
---------------------------------------------------

This kind of benchmarking is valuable to discover the maximum aggregate number of FPS the pipeline can reach. It may occur that the maximum value will be reached when ingesting 32 pipelines, but each of them will be processed at the rate of 5 FPS. That is why it is not real-time performance.

To run such benchmarks we implemented a special adapter: :ref:`Multi-Stream Source Adapter <multi_stream_source_adapter>`, allowing the ingesting of a selected video file in parallel in the pipeline under benchmarking. By changing the number of parallel streams you may find out the value which is the maximum for the pipeline.

To measure non-real-time performance with it, use ``SYNC_OUTPUT=False``.

Measuring Real-Time Multiple Stream Performance
-----------------------------------------------

This kind of benchmarking is valuable to find out the maximum aggregate number of FPS the pipeline can handle in real time. Considering the per-stream FPS is 30, you are looking for working configurations satisfying the equation ``N = X / 30``, where ``N`` is the number of streams ingested in the pipeline, and ``X`` is the aggregate FPS.

To run such benchmarks, you also can use the :ref:`Multi-Stream Source Adapter <multi_stream_source_adapter>` but set ``SYNC_OUTPUT=True``. By changing the number of parallel streams, you need to determine the value which is the maximum for the pipeline.

End-To-End or Isolated Benchmarking
-----------------------------------

In the above-mentioned :ref:`listing <isolated_uridecodebin_benchmark>` you may see that the sink for the pipeline is set to:

.. code-block::

    pipeline:
      # noop pipeline sink, not using sink adapter
      sink:
        - element: devnull_sink


It represents benchmarking without real-life sinks which can form a bottleneck. If you want to test CV/ML performance it is what you are looking for. However, often you need to test end-to-end, including a specific sink implementation used practically. In such a situation, you need to include in the benchmark additional components, like a sink adapter you are planning to use and 3rd-party systems.

Major Performance-Related Parameters
------------------------------------

There are two kinds of factors influencing the performance of a pipeline: related to implementation and configuration. Let us begin with configuration parameters.

Configurable parameters influencing performance include:

- :doc:`batching <0_batching>`;
- :doc:`buffer queues unlocking Python multithreading </recipes/1_python_multithreading>`;
- `inference parameters <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>`_;
- `tracking parameters <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_NvMultiObjectTracker_parameter_tuning_guide.html>`_;

When benchmarking, you need to mix and match them to empirically discover the combinations giving the best performance. It may take a decent amount of time to measure the pipeline performance in different configurations, so accessing several benchmarking nodes and having an automated environment helps a lot.

Implementation factors relate to models and code design. The performance is affected by:

- :doc:`the use FFI GIL-free code </recipes/1_python_multithreading>`;
- model quantization with TensorRT;
- `model pruning <https://blog.savant-ai.io/yolov7-inference-acceleration-with-structural-pruning-7a74a9cbfc73?source=friends_link&sk=41dffd9312b8fd55a9c4eb77481b8997>`_;
- synchronization with 3rd-party systems;
- the amount of GPU-CPU memory transfers: :doc:`/savant_101/80_opencv_cuda`, :doc:`/savant_101/80_map`.

We do not discuss them here because they require experimenting and in-depth analysis.

Tools
-----

The tools for monitoring the benchmarking environment include but are not limited by:

- ``nvidia-smi``, ``tegrastats``: analyze GPU performance;
- ``sar``: analyze host CPU/RAM utilization;
- ``nvtop``: monitor GPU utilization;
- ``htop``: monitor CPU/RAM utilization;
- :doc:`OpenTelemetry <9_open_telemetry>` and :doc:`ClientSDK <10_client_sdk>`: profile the code.

Real-Time Data Sources And The Pipeline is a Bottleneck
-------------------------------------------------------

If real-time sources are used and the pipeline is a bottleneck, to avoid data loss, the sources must be connected to the pipeline with an in-memory or persistent queue system like Apache Kafka. The same is true for communication between the pipeline and sinks.

GIL-Bound Pipelines
--------------------

Pipeline performance may be limited by GIL. This is a frequent case when a lot of unoptimized Python code is used. Such code utilizes a single CPU core to 100%, while other cores remain underutilized. If ``htop`` shows such a picture while ``nvtop`` demonstrates that GPU resources are underutilized, the pipeline is GIL-bound.

What to do:

- switch from VPS to bare metal;
- consider using high-frequency CPUs with small number of cores, fast memory and large cache;
- move heavyweight operations out of the pipeline (e.g., use Apache Spark or Flink);
- unlock GIL by introducing GIL-free FFI code (Cython, C, C++, Rust), replace naive code with optimized computations made with NumPy, Numba, OpenCV;
- try pipeline :doc:`chaining <6_chaining>` to split workload among several Python processes;
- launch multiple instances of a pipeline on a single GPU to distribute the workload between more CPU cores and fully utilize GPU resources.

CPU-Bound Pipelines
-------------------

It may occur that the pipeline utilizes proper optimizing techniques and utilizes all CPU cores available, while GPU remains underutilized.

What to do:

- switch from VPS to bare metal;
- consider choosing CPU with large number of cores;
- move heavyweight operations out of the pipeline to a separate host (e.g., use Apache Spark or Flink);
- reconfigure a platform, selecting less capable GPU keeping the same CPU.

GPU-Bound Pipelines
-------------------

This is normally a good situation. What approaches may improve the performance:

- network pruning;
- network quantization;
- try pipeline :doc:`chaining <6_chaining>` and multiple GPUs;
- choosing a more capable GPU model.
