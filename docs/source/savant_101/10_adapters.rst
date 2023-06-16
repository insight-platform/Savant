Adapters
========

The adapter is a standalone program typically executed in a Docker container. Adapters communicate with modules via ZeroMQ sockets: source adapters send data into, and sink adapters receive data from them.

The decoupled nature of adapters ensures better reliability because an error-prone data source affects the adapter operation, not the module operation. As a result, it is always operational regardless of data sources.

The adapters can transfer video streams and metadata through the network or locally. We have already implemented several handy adapters, and you can implement the required one if needed - the protocol is based on standard, widely supported open-source technologies.

Savant Adapter Protocol
-----------------------

There is a protocol based on ZeroMQ and Apache Avro, which adapters use to communicate. It can be used to connect adapter with adapter, adapter with module, module with module, etc. The protocol is universal for sources and sinks. It allows transferring the following:

- video frames [optional];
- video stream information;
- frame-related metadata;
- the hierarchy of objects related to the frame.

The protocol is described in the :doc:`API <../reference/avro>` section.

Communication Sockets
---------------------

The adapters and modules can use various ZeroMQ socket pairs to establish communication. The chosen type defines the possible topologies and quality of service. Currently, the following pairs are supported:

- DEALER/ROUTER: reliable, asynchronous pair with backpressure (default choice);
- REQ/REP: reliable, synchronous pair (paranoid choice);
- PUB/SUB: unreliable, real-time pair (default choice for strict real-time operation or broadcasting).

There are two URL schemes supported:

- Unix domain sockets;
- TCP sockets.

You must prefer Unix domain sockets over TCP sockets when communication is established locally, especially when RAW frame formats are used (e.g. when using GigE Vision industrial cams or USB cams).

Unix domain URLs look like as follows:

.. code-block:: bash

    pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc
    pub+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    sub+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc
    router+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    # etc...

TCP socket URL look like as follows:

.. code-block:: bash

    pub+bind:tcp://0.0.0.0:3332
    sub+bind:tcp://0.0.0.0:3331
    pub+connect:tcp://10.0.0.10:3331
    # etc

When used in the environment variables it may look like:

.. code-block:: bash

    ZMQ_ENDPOINT="dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc"
    # is equal to
    ZMQ_ENDPOINT="ipc:///tmp/zmq-sockets/input-video.ipc"
    ZMQ_TYPE="DEALER"
    ZMQ_BIND="False"

Or:

.. code-block:: bash

    ZMQ_ENDPOINT="pub+bind:tcp://1.1.1.1:3333"
    # is equal to
    ZMQ_ENDPOINT="tcp://1.1.1.1:3333"
    ZMQ_TYPE="PUB"
    ZMQ_BIND="True"

Read more about ZeroMQ socket pairs on ZeroMQ `website <https://zeromq.org/socket-api/>`__.

You usually want to use combinations, which are marked with Green color:

.. image:: ../_static/img/10_adapters_normal_pairs.png

The Rules Of Thumb
-----------------

Typically we recommend following the next ideas when planning how your adapters communicate with the module:

- Try to use the framework in bind mode, and adapters in connect mode first; change only if it does not work for you.
- The part which delivers multiplexed stream usually has the bind type; the part which handles a single (non-multiplexed) stream usually has the connect type.
- Use the ``PUB/SUB`` pair only when the pipeline or adapter is capable to handle the traffic in real-time.

Typical Patterns
----------------

There are typical patterns widely used, try to start from them when designing pipelines.

- ``D`` - dealer;
- ``R`` - router;
- ``P`` - publisher (PUB);
- ``S`` - subscriber (SUB).

The pairs are explained after the patterns section in detail.

Data-Center Patterns
^^^^^^^^^^^^^^^^^^^^

Data-center patterns are designed to reliably process video streams with increased latency in situations when the pipeline is overwhelmed with data. 0MQ socket pairs used in data-center patterns are ``DEALER/ROUTER`` (default recommended) or ``REQ/REP``. These pairs implement a backpressure mechanism which causes the processing to be delayed when watermarks are reached.

.. image:: ../_static/img/10_adapters_dc_patterns.png

The first one is a typical scenario when the adapter reads multiplexed streams from an external queue system (like Kafka) and passes them to the framework instance. The framework, in turn, transfers analytics results (and video) to the adapter, which places the results into a database or another queue system.

The second is typical when adapters are used to aggregate data from multiple streams (e.g. ``RTSP`` cams) into the framework instance. The right side of the pipeline stays the same as in the previous case.

Edge Patterns
^^^^^^^^^^^^^

Edge patterns often aim at providing real-time operations for data sources with the lowest latency possible. To implement that, you may utilize the ``PUB/SUB`` socket pair because it drops the packets that the ``SUB`` part cannot process in a timely manner.

This mechanism works absolutely great when used with streams delivering ``MJPEG``, ``RAW``, ``JPEG``, ``PNG``, and other independent video frames. Using the pattern with video-encoded streams is troublesome because drops cause video corruption.

.. image:: ../_static/img/10_adapters_edge_patterns.png

The first pattern may be used when neither adapters nor framework must be frozen because of the sink stalling. The second pattern is beneficial when the sink guarantees the processing, and you do not concern that it can be overwhelmed, causing the framework pipeline to stall too.

DEALER/ROUTER
-------------

This is a recommended pair to use when you don't need to copy the same messages to multiple subscribers or can implement such duplication programmatically. It is a reliable socket pair: the ``DEALER`` will block if the ``ROUTER``'s queue is full.

**Source/CONNECT-to-Framework/BIND communication**. This is a typical scheme of communication.

.. image:: ../_static/img/10_adapters_dr_scfb.png

**Framework/CONNECT-to-Sink/BIND communication**. This is a normal pattern, when you have the sink adapter communicating with the external system like Kafka and wish to send data from multiple framework instances.

.. image:: ../_static/img/10_adapters_dr_fcsb.png

**Source/BIND-to-Framework/CONNECT communication**. This is an exotic pattern. Although, It may do the job when you handle raw frames or isolated image streams and don't care about per-stream order. In this scheme, the source will distribute data berween connected frameworks according to ``LRU`` strategy, so it is impossible to use the scheme when you work with video.

.. image:: ../_static/img/10_adapters_dr_sbfc.png

**Framework/BIND-to-Sink/CONNECT communication**. This is a valid pattern, when sinks communicating with an external system are slow or require multiple operations and the order of data appending is not critical.


.. image:: ../_static/img/10_adapters_dr_fbsc.png

REQ/REP
-------

The ``REQ/REP`` pair works the same way as the ``DEALER/ROUTER`` except that the ``REQ`` part receives replies from the ``REP`` part every time the ``REP`` part handles the message. It can be useful to modify the injecting pace on the ``REQ`` part. This is a generally recommended pair to use when you don't need multiple subscribers or can implement such duplication somehow. It is reliable socket pair: the ``REQ`` sends the next frame only when received the response previously sent from ``REP``.

PUB/SUB
-------

The ``PUB/SUB`` is convenient to use when you need to handle the same data by multiple subscribers. Another use case for ``PUB/SUB`` is when you are processing the real-time data: when excessive elements are silently dropped if the pipeline or adapter is unable to handle the traffic burst.

**Source/BIND-to-Framework/CONNECT communication**. The source is initialized as a server (bind), the framework connects to it as a client. This scheme is typically can be used when the source already delivers multiple streams or the frameworks handles a single stream provided by the source. In this scenario the source can duplicate the same stream to multiple frameworks simultaneously.

.. image:: ../_static/img/10_adapters_ps_sbfc.png

**Framework/BIND-to-Sink/CONNECT communication**. This is a typical scheme which can be used widely. The framework as a server can stream results to multiple sink adapters. Every such adapter can filter out only required information.

.. image:: ../_static/img/10_adapters_ps_fbsc.png

**Source/CONNECT-to-Framework/BIND communication**. This is a typical when the framework handles multiple streams. The framework binds to a socket and clients connect to that socket.

.. image:: ../_static/img/10_adapters_ps_scfb.png

**Framework/CONNECT-to-Sink/BIND communication**. This is not a typical but a legal scheme. The sink handles multiple outputs from frameworks to deliver them some storage, e.g. Kafka or ClickHouse.

.. image:: ../_static/img/10_adapters_ps_fcsb.png

Examples:

- you want to pass frames from a single camera to two different pipelines;
- you want to pass resulting video analytics to two different adapters (e.g. RTSP streaming and somewhere else).

``PUB/SUB`` is not a reliable communication pair, which means that if the subscriber is slow the frames will be dropped; the ``PUB`` part never blocks. To overcome that the adapter must handle incoming frames in an advanced way (e.g. using internal queueing).

Generally we recommend using the PUB/SUB in the following scenarios:

- you work with independently encoded frames from a cam (``MJPEG``, ``RGB``, etc.) so when processing is slow you can afford dropping frames;
- you implemented an adapter to read frames from the socket fast and know how to queue them internally.

**Antipattern**: passing video files over ``PUB/SUB`` to the framework with no ``SYNC`` flag set.

**Pattern example (Sink)**: Always-On RTSP Sink Adapter when multiple streams are cast.

We provide adapters to address the everyday needs of users. The current list of adapters enables the implementation of many typical scenarios in real life. Every adapter can be used as an idea to implement a specific one required in your case.

Source Adapters
---------------

Source adapters are used to deliver data from external sources (files, RTSP, devices) to a framework module.

Currently, the following `source <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md#source-adapters>`_ adapters are available:

- Video loop adapter;
- Local video file;
- Local directory of video files;
- Local image file;
- Local directory of image files;
- Image URL;
- Video URL;
- RTSP stream;
- USB/CSI camera;
- GigE (Genicam) industrial cam.

All adapters accept the following parameters:

- ``SOURCE_ID`` - unique identifier for the source adapter; this option is **required**;
- ``ZMQ_ENDPOINT`` - adapter output (should be equal to module input) ZeroMQ socket endpoint; schema: ``[<socket_type>+(bind|connect):]<endpoint>``;
- ``ZMQ_TYPE`` - adapter output ZeroMQ socket type; default is ``DEALER``, also can be set to ``PUB`` or ``REQ``;
- ``ZMQ_BIND`` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to ``True``); default is ``False``;
- ``FPS_PERIOD_FRAMES`` - number of frames between FPS reports; default is ``1000``;
- ``FPS_PERIOD_SECONDS`` - Number of seconds between FPS reports; default is ``None``;
- ``FPS_OUTPUT`` - path to the file where the FPS reports will be written; Default is ``stdout``.

Image File Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

The Image File Source Adapter reads ``JPEG`` or ``PNG`` files from specified ``LOCATION``, which can be:

- a local path to a single file;
- a local path to a directory with files (not necessarily in the same encoding);
- HTTP URL to a single file.

The adapter is useful for development purposes. It also can be used to process image streams efficiently in production.

The adapter parameters are set with environment variables:

- ``FILE_TYPE`` - must be set to ``picture``;
- ``LOCATION`` - image file(s) location or URL;
- ``FRAMERATE`` - desired framerate for the video stream formed from the input image files (if sync mode is chosen);
- ``SYNC_OUTPUT`` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``EOS_ON_FILE_END`` - flag indicates whether to send ``EOS`` message at the end of each file; default is ``False``;
- ``SORT_BY_TIME`` - flag indicates whether the files from ``LOCATION`` are sorted by modification time (ascending order); by default, it is ``False`` and the files are sorted lexicographically.
- ``READ_METADATA`` - flag indicates the need to read and send the object's metadata from a ``JSON`` file that has the identical name as the source file; default is ``False``.

Example:

.. code-block:: bash

    docker run --rm -it --name source-pictures-files-test \
        --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/images \
        -e FILE_TYPE=picture \
        -e SORT_BY_TIME=False \
        -e READ_METADATA=False \
        -v /path/to/images:/path/to/images:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest


The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py pictures --source-id=test /path/to/images

Video File Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

The video file source adapter reads video files from ``LOCATION``, which can be:

- a local path to a single file;
- a local path to a directory with one or more files;
- HTTP URL to a single file;

The adapter parameters are set with environment variables:

- ``FILE_TYPE`` - must be set to ``video``;
- ``LOCATION`` - video file(s) location or URL;
- ``EOS_ON_FILE_END`` - flag indicates whether to send ``EOS`` message at the end of each file; default is ``True``;
- ``SYNC_OUTPUT`` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``SORT_BY_TIME`` - flag indicates whether files from ``LOCATION`` are sorted by modification time (ascending order); by default, it is ``False`` and files are sorted lexicographically;
- ``READ_METADATA`` - flag indicates the need to read the object's metadata from a ``JSON`` file that has the identical name as the source file; default is ``False``.

Example:

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
        --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
        -e FILE_TYPE=video \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/data/test.mp4 \
        -e SORT_BY_TIME=False \
        -e READ_METADATA=False \
        -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py videos --source-id=test /path/to/data/test.mp4

.. note::

    Resulting video stream framerate is equal to the framerate of the first encountered video file, possibly overriding the framerate of the rest of input files.

Video Loop File Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The video loop file source adapter plays continuously a video file from ``LOCATION``, which can be:

- a local path to a single file;
- HTTP URL to a single file;

.. note::
    The adapter helps developers create infinite video streams to benchmark, demonstrate, and test pipelines. It allows configuring arbitrary frame loss to test processing in an unstable environment.

The adapter parameters are set with environment variables:

- ``LOCATION`` - video file location or URL;
- ``EOS_ON_LOOP_END`` - flag indicates whether to send ``EOS`` message at the end of each loop; default is ``False``;
- ``READ_METADATA`` - flag indicates the need to read the object's metadata from a JSON file that has the identical name as the source file; default is ``False``;
- ``SYNC_OUTPUT`` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``DOWNLOAD_PATH`` - target directory to download files from remote storage in the first loop and reuse it in the next loops;
- ``LOSS_RATE`` - probability to drop the frames.

Example:

.. code-block:: bash

    docker run --rm -it --name source-video-loop-test \
        --entrypoint /opt/savant/adapters/gst/sources/video_loop.sh \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/data/test.mp4 \
        -e READ_METADATA=False \
        -e DOWNLOAD_PATH=/tmp/video-loop-source-downloads \
        -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        -v /tmp/video-loop-source-downloads:/tmp/video-loop-source-downloads \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py video-loop --source-id=test /path/to/data/test.mp4

RTSP Source Adapter
^^^^^^^^^^^^^^^^^^^

The RTSP source adapter delivers RTSP stream to a module. The adapter parameters are set with environment variables:

- ``RTSP_URI`` - RTSP URI of the stream; this option is required;
- ``SYNC_OUTPUT`` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``SYNC_DELAY`` - delay in seconds before sending frames; useful when the source has B-frames to avoid sending frames in batches; default is ``0``;
- ``CALCULATE_DTS`` - flag indicates whether the adapter should calculate DTS for frames; set this flag when the source has B-frames; default is ``False``;
- ``BUFFER_MAX_BYTES`` - maximum amount of data in the buffer; default is ``10485760`` (``10`` MB).

Example:

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
        --entrypoint /opt/savant/adapters/gst/sources/rtsp.sh \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e RTSP_URI=rtsp://192.168.1.1 \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py rtsp --source-id=test rtsp://192.168.1.1

USB Cam Source Adapter
^^^^^^^^^^^^^^^^^^^^^^

The USB cam source adapter captures video from a V4L2 device specified in ``DEVICE`` parameter.

The adapter parameters are set with environment variables:

- ``DEVICE`` - USB camera device; default value is ``/dev/video0``;
- ``FRAMERATE`` - desired framerate for the video stream captured from the device; note that if the input device does not support specified video framerate, results may be unexpected;

Example:

.. code-block:: bash

    docker run --rm -it --name source-pictures-files-test \
        --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/images \
        -e FILE_TYPE=picture \
        -e SORT_BY_TIME=False \
        -e READ_METADATA=False \
        -v /path/to/images:/path/to/images:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py pictures --source-id=test /path/to/images

GigE Source Adapter
^^^^^^^^^^^^^^^^^^^

he adapter is designed to take video streams from GigE/Genicam industrial cameras. It passes the frames captured from the camera to the framework without encoding (`#18 <https://github.com/insight-platform/Savant/issues/18>`__) which may introduce significant network payload. We recommend using it locally with the framework deployed at the same host.

The adapter parameters are set with environment variables:

* ``WIDTH`` - the width of the video frame, in pixels;
* ``HEIGHT`` - the height of the video frame, in pixels;
* ``FRAMERATE`` - the framerate of the video stream, in frames per second;
* ``INPUT_CAPS`` - the format of the video stream, in GStreamer caps format (e.g. video/x-raw,format=RGB);
* ``PACKET_SIZE`` - the packet size for GigEVision cameras, in bytes;
* ``AUTO_PACKET_SIZE`` - whether to negotiate the packet size automatically for GigEVision cameras;
* ``EXPOSURE`` - the exposure time for the camera, in microseconds;
* ``EXPOSURE_AUTO`` - the auto exposure mode for the camera, one of ``off``, ``once``, or ``on``;
* ``GAIN`` - the gain for the camera, in decibels;
* ``GAIN_AUTO`` - the auto gain mode for the camera, one of ``off``, ``once``, or ``on``;
* ``FEATURES`` - additional configuration parameters for the camera, as a space-separated list of features;
* ``HOST_NETWORK`` - host network to use;
* ``CAMERA_NAME`` - name of the camera, in the format specified in the command description;

Example:

.. code-block:: bash

docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/savant/adapters/gst/sources/gige_cam.sh \
    -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
    -e SOURCE_ID=test \
    -e CAMERA_NAME=test-camera \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The adapter can be run using a script:

.. code-block:: bash

    ./scripts/run_source.py gige --source-id=test test-camera


Sink Adapters
-------------

There are basic `sink <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md#sink-adapters>`_ adapters implemented:

- Inference results are placed into JSON file stream;
- Resulting video overlay displayed on a screen (per source);
- MP4 file (per source);
- image directory (per source);
- Always-On RTSP Stream Sink.
