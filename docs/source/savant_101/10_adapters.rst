How Savant Communicates With External Video Sources
===================================================

Let us get acquainted with the top-level components of Savant. A "module" is a docker container where a pipeline running computer vision resides. Modules must somehow communicate with the external world: they capture video or images from cams, files, queues, and streaming servers and send results to other external systems like video archives, databases, broadcast systems, etc. In Savant, **normally** modules don't do these things by themselves, delegating them to adapters, but direct interaction is also possible.

GStreamer implements pluggable architecture. There is a number of plugin types supported: sources, sinks, transformers. By default, it provides ready-to-use plugins to communicate with external data, like ``uridecodebin``, ``videotestsrc`` or ``fakesink``.

Savant also translates its manifest to a GStreamer pipeline, but with custom ZeroMQ source and ZeroMQ sink elements enabling the communication with data via special external processes called adapters (we introduce them in the next sections).

Those elements are defined in the module definition by default and usually are hidden from developers. However, the interested person may change the default source and sink to GStreamer elements like ``uridecodebin`` if the pipeline benefits from it.

Currently, the single source- and multiple sink declarations are supported.

Adapters
========

The adapters are standalone programs executed in separate Docker containers. They communicate with modules via `ZeroMQ <https://zeromq.org/>`__: source adapters ingest data, sink adapters consume data from modules and bridge adapters consume data from other sources and ingest it.

The decoupled nature of adapters guarantees high reliability because errors happening outside of the pipeline don't propagate to the module. Thus, adapters deliver two main functions: abstracting the module from data sources and destinations and providing a foundation for fault-tolerant operations.

Adapters transfer video streams and metadata over network or locally. We implemented several handy adapters; interested parties can implement the specific adapters to address their situations: the protocol is based on open-source technologies.

Savant Adapter Protocol
-----------------------

Savant uses a protocol based on `ZeroMQ <https://zeromq.org/>`__ and `Savant-RS <https://insight-platform.github.io/savant-rs/>`__ for communication between adapters and modules. It can be used to connect an adapter with other adapter, an adapter with a module, a module with a module, etc. The protocol is universal for source, sink and bridge adapters.

With the protocol, one may build oriented graphs representing data sources, sinks, and modules, arranging them within a single host or in a distributed environment like ``K8s``.

It supports transferring:

- video frames [optional];
- video stream-level information (encoding, fps, resolution, etc);
- frame-related metadata (global per-frame attributes);
- the hierarchy of objects and their attributes related to the frame.

The protocol is described in the `Savant-RS serialization <https://insight-platform.github.io/savant-rs/modules/savant_rs/utils_serialization.html#savant_rs.utils.serialization.Message>`__ section.

Communication Sockets
---------------------

Adapters and modules may use three different ZeroMQ patterns to communicate. The chosen pattern defines possible topologies and quality of service.

Currently, the following patterns are supported:

- ``DEALER/ROUTER``: reliable, asynchronous pair with backpressure (default choice);
- ``REQ/REP``: reliable, synchronous pair (paranoid choice);
- ``PUB/SUB``: unreliable, real-time pair (default choice for strict real-time operation or broadcasting).

You can read read more about ZeroMQ sockets on ZeroMQ `website <https://zeromq.org/socket-api/>`__.

Savant currently supports two network transports:

- Unix domain sockets;
- TCP sockets (unicast).

You must prefer Unix domain sockets over TCP sockets when communication is established locally, especially when uncompressed frame formats are used (e.g., when using GigE Vision industrial cams or USB cams).

In Savant Unix domain sockets URLs look like as follows:

.. code-block:: bash

    pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc
    pub+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    sub+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc
    router+bind:ipc:///tmp/zmq-sockets/input-video.ipc
    # etc...

TCP sockets URLs look like as follows:

.. code-block:: bash

    pub+bind:tcp://0.0.0.0:3332
    sub+bind:tcp://0.0.0.0:3331
    pub+connect:tcp://10.0.0.10:3331
    # etc

When the transports are specified  with the environment variables it looks like:

.. code-block:: bash

    # Unix domain socket communication
    ZMQ_ENDPOINT="dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc"

    # which is equal to
    ZMQ_ENDPOINT="ipc:///tmp/zmq-sockets/input-video.ipc"
    ZMQ_TYPE="DEALER"
    ZMQ_BIND="False"

Or:

.. code-block:: bash

    # tcp socket communication
    ZMQ_ENDPOINT="pub+bind:tcp://1.1.1.1:3333"

    # which is equal to
    ZMQ_ENDPOINT="tcp://1.1.1.1:3333"
    ZMQ_TYPE="PUB"
    ZMQ_BIND="True"


Not all socket pairs form "sane" communication patterns, so, you must use combinations colored green:

.. image:: ../_static/img/10_adapters_normal_pairs.png

The Rules Of Thumb
------------------

Consider the following ideas when planning ``source-module-sink`` topologies:

- Use the module in the bind mode, adapters in the connect mode; change if it does not work for you.
- The party which delivers multiplexed stream usually has the bind type; the party which handles a single (non-multiplexed) stream usually has the connect type.
- Use the ``PUB/SUB`` pair when the pipeline or adapter is capable of handling the traffic in real-time.

Typical Patterns
----------------

We recommend starting with typical patterns when designing pipelines.

Legend:

- ``D`` - dealer;
- ``R`` - router;
- ``P`` - publisher (PUB);
- ``S`` - subscriber (SUB).

The pair combinations are discussed after the patterns-related sections.

Data-Center Patterns
^^^^^^^^^^^^^^^^^^^^

Data-center patterns are used to reliably process video streams with increased latency in situations when the pipeline is overwhelmed with data. Typical ZeroMQ socket pairs used in data-center patterns are ``DEALER/ROUTER`` (recommended) or ``REQ/REP``.

These pairs implement backpressure causing processing to be delayed when thresholds are reached.

.. image:: ../_static/img/10_adapters_dc_patterns.png

The first represents a typical scenario when an adapter reads multiplexed streams from an external queue system (like Kafka) and passes them to a module. The module, in turn, transfers results (and video) to an adapter delivering them into an external system.

The second is typical when adapters deliver data from several sources (e.g. RTSP cams) into a module instance. The right side of the pipeline stays the same as in the previous case.

Edge Patterns
^^^^^^^^^^^^^

Edge is usually used to serve low-latency real-time video processing. To implement that, we establish the ``PUB/SUB`` connection because it drops the packets that the ``SUB`` part cannot process on time.

This mechanism works great with streams delivering ``MJPEG``, ``RAW``, ``JPEG``, ``PNG``, and other independently encoded video frames. Using it with keyframe-encoded streams leads to video corruption.

.. image:: ../_static/img/10_adapters_edge_patterns.png

The first pattern can be used when neither adapters nor the module must get stuck because of the sink stalling. The second pattern is beneficial when a sink guarantees processing, and you do not worry that it may cause stalling.

DEALER/ROUTER
^^^^^^^^^^^^^

This is the recommended pair when you don't need to copy the same messages to multiple subscribers. It is a reliable socket pair: the ``DEALER`` will block if the ``ROUTER``'s queue is full.

**Source/CONNECT, Module/BIND**. This is a typical scheme.

.. image:: ../_static/img/10_adapters_dr_scfb.png

**Module/CONNECT, Sink/BIND**. This is a normal pattern when a sink adapter communicates with an external system like Kafka and wishes to send data from multiple module instances.

.. image:: ../_static/img/10_adapters_dr_fcsb.png

**Source/BIND, Module/CONNECT**. This is an exotic pattern. Nevertheless, it does the job when a module handles independent images without the need to maintain per-source order. In this scheme, the source will evenly distribute data between connected modules according to the ``LRU`` strategy, so it is impossible to use the scheme when you work with video.

.. image:: ../_static/img/10_adapters_dr_sbfc.png

**Module/BIND, Sink/CONNECT**. This is a valid pattern when sinks communicating with an external system require partitioning and data appending order is not critical.

.. image:: ../_static/img/10_adapters_dr_fbsc.png

REQ/REP
^^^^^^^

The ``REQ/REP`` pair is similar to ``DEALER/ROUTER`` except that the ``REQ`` part receives replies from the ``REP`` part every time the ``REP`` part reads the message.

PUB/SUB
^^^^^^^

The ``PUB/SUB`` is convenient when you need to duplicate the same data to multiple subscribers. Another use case is real-time data processing: excessive elements are dropped if the pipeline cannot handle the traffic.

**Source/BIND, Module/CONNECT**. A source is initialized as a server (bind), and a module connects to it. This scheme can be used when the source already delivers multiple streams or the module handles a single stream provided by the source. In this scenario, the source can duplicate the same stream to multiple modules simultaneously.

.. image:: ../_static/img/10_adapters_ps_sbfc.png

**Module/BIND, Sink/CONNECT**. This scheme is used widely. A module duplicates the same data to multiple sinks. A sink can filter out only required data.

.. image:: ../_static/img/10_adapters_ps_fbsc.png

**Source/CONNECT, Module/BIND**. A typical scheme when a module handles multiple streams. The module binds to a socket and adapters connect to that socket.

.. image:: ../_static/img/10_adapters_ps_scfb.png

**Module/CONNECT, Sink/BIND**. This is unusual but a correct scheme. A sink handles multiple outputs from modules to deliver them in a storage, e.g. Kafka or ClickHouse.

.. image:: ../_static/img/10_adapters_ps_fcsb.png

``PUB/SUB`` examples:

- delivering frames from a single camera to two different pipelines;
- delivering resulting video analytics to two different adapters (e.g. for RTSP streaming and database ingestion).

``PUB/SUB`` is unreliable (no backpressure), which means that if the subscriber is slow the frames may be lost because ``PUB`` never blocks. The adapter must handle incoming frames smartly (using internal queueing) to overcome that.

We recommend using the PUB/SUB in the following scenarios:

- when processing independently encoded frames from a cam (``MJPEG``, ``RGB``, etc.), so when processing is slow, you can afford to drop frames;
- when an adapter is implemented in a way to read frames from the socket fast and know how to queue them internally.

**Antipattern**: passing video files over ``PUB/SUB`` to the module with no ``SYNC`` flag set.

**Pattern example (Sink)**: Connecting multiple Always-On RTSP Sink instances to the module instance to cast multiple streams.

We provide adapters to address the common needs of users. The current list of adapters covers many typical scenarios in real life. Provided adapters can be used as an idea to implement a specific one required in your case.

Source Adapters
---------------

Source adapters deliver data from external sources (files, RTSP, devices) to a module.

Currently, the following source adapters are available:

- Video loop adapter;
- Local video file;
- Local directory of video files;
- Local image file;
- Local directory of image files;
- Image URL;
- Video URL;
- RTSP stream;
- USB/CSI camera;
- GigE (Genicam) industrial cam;
- FFmpeg;
- Multi-stream;
- Kafka-Redis Source;
- Message Dump Player.

Most source adapters accept the following common parameters:

- ``SOURCE_ID``: a string identifier for a stream processed; this option is **required**; every stream must have a unique identifier, if identifiers collide, processing may cause unpredictable results; the identifier may encode user-defined semantics in a prefix, like ``rtsp.stream.1``; many sink adapters can filter out streams by prefix or full ``SOURCE_ID``;
- ``ZMQ_ENDPOINT``: adapter's socket where it sends media stream; it must form a valid ZeroMQ pair with module's input socket; the endpoint coding scheme is ``[<socket_type>+(bind|connect):]<endpoint>``;
- ``ZMQ_TYPE``: a socket type; default is ``DEALER``, also can be set to ``PUB`` or ``REQ``; **warning**: this parameter is deprecated, consider encoding the type in ``ZMQ_ENDPOINT``;
- ``ZMQ_BIND``; a socket mode (the ``bind`` mode is when the parameter is set to ``True``); default is ``False``; **warning**: this parameter is deprecated, consider encoding the type in ``ZMQ_ENDPOINT``;
- ``FPS_PERIOD_FRAMES``; a number of frames between FPS reports; FPS reporting helps to estimate the performance of the pipeline components deployed; default is ``1000``;
- ``FPS_PERIOD_SECONDS``; a number of seconds between FPS reports; default is ``None`` which means that FPS reporting uses ``FPS_PERIOD_FRAMES``;
- ``FPS_OUTPUT``; a path to the file for FPS reports; default is ``stdout``;
- ``USE_ABSOLUTE_TIMESTAMPS``; when ``True`` the adapter puts absolute timestamps into the frames, i.e. the timestamps of the frames start from the time of adapter launch; default is ``False``.

Image File Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

The Image File Source Adapter sends ``JPEG`` or ``PNG`` files to a module. It may be used to generate video streams from separate images or process independent images.

The images are served from:

- a local path to a single file;
- a local path to a directory with files (not necessarily in the same encoding);
- an HTTP URL to a single file.

.. note::

    The adapter is useful for development purposes: a developer can associate every image with extra metadata in JSON format to implement pipeline testing. E.g., you may add metadata for expected bounding boxes and evaluate assertions in the pipeline to validate that the model predicts them.

.. note::
    The adapter also can be used to implement asynchronous image processing pipelines. Metadata allows passing per-image identification information over the pipeline to access the results when those images are processed.

.. note::
    It is advisable to have all pictures in the same resolution to avoid issues with the pipeline. Sending images with different resolutions will degrade the performance of the pipeline. To avoid this there are few options:
        1. Resize all images to the same resolution.
        2. Set ``EOS_ON_FRAME_PARAMS_CHANGE=True`` in the adapter and ``pipeline.source.properties.eos_on_frame_resolution_change: true`` in module configuration. This should work on dGPU but won't work on Jetson.
        3. Sort images by resolution. This will decrease the amount of generated EOS.

**Parameters**:

- ``FILE_TYPE``: a flag specifying that the adapter is used for images; it must always be set to ``image``;
- ``LOCATION``: a filesystem location (path or directory) or HTTP URL from where images are served;
- ``FRAMERATE``: a desired framerate for the output video stream generated from image files (the parameter is used only if ``SYNC_OUTPUT=True``);
- ``SYNC_OUTPUT``: a flag indicating that images are delivered into a module as a video stream; otherwise, the files are sent as fast as the module is capable processing them; default is ``False``;
- ``EOS_ON_FILE_END``: a flag configuring sending of ``EOS`` message after every image; the ``EOS`` message is important to trackers, helping them to reset tracking when a video stream is no longer continuous; default is ``False``;
- ``EOS_ON_FRAME_PARAMS_CHANGE``: a flag configuring sending of ``EOS`` message after every change in image resolution; the ``EOS`` message is important to trackers, helping them to reset tracking when a video stream is no longer continuous; default is ``True``;
- ``SORT_BY_TIME``: a flag specifying sorting by modification time (ascending); by default, it is ``False``, causing the files to be sorted lexicographically;
- ``READ_METADATA``: a flag specifying the need to augment images with metadata from ``JSON`` files with the corresponding names as the source files; default is ``False``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name images-source \
        --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
        -e FILE_TYPE=image \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/images \
        -v /path/to/images:/path/to/images:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest


Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py images --source-id=test /path/to/images

Video File Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

The Video File Source Adapter sends video files to a module as a single stream.

The video files are served from:

- a local path to a single file;
- a local path to a directory with one or more files;
- HTTP URL to a single file;

**Parameters**:

- ``FILE_TYPE``: must be set to ``video``;
- ``LOCATION``: a video file(s) location or URL;
- ``EOS_ON_FILE_END``: a flag indicating whether to send the ``EOS`` message at the end of each file; default is ``True``; the ``EOS`` message is crucial for trackers to recognize when a video stream is no longer continuous; when sending ordered parts of a single video file without gaps usually must be set to ``False``;
- ``EOS_ON_FRAME_PARAMS_CHANGE``: a flag indicating whether to send the ``EOS`` message after every change in video parameters (resolution, framerate); default is ``True``; the ``EOS`` message is crucial for trackers to recognize when a video stream is no longer continuous;
- ``SYNC_OUTPUT``: flag specifying if to send frames synchronously (i.e. at the source file rate); default is ``False``;
- ``SORT_BY_TIME``: a flag indicating whether files are sorted by modification time (ascending) before sending to a module; by default, it is ``False`` (lexicographical sorting);
- ``READ_METADATA``: a flag specifying the need to augment video frames with metadata from ``JSON`` files with the corresponding names as the source files; default is ``False``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
        --entrypoint /opt/savant/adapters/gst/sources/media_files.sh \
        -e FILE_TYPE=video \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/data/test.mp4 \
        -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py videos --source-id=test /path/to/data/test.mp4

.. note::

    The resulting video stream framerate is set to the framerate of the first video file; subsequent files are delivered with the same FPS. Consider having the same framerate for all video files or serving each file separately. The adapter is lightweight, and the cost of launching is negligible.

Video Loop Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

The Video Loop Source Adapter sends a video file continuously in a loop.

The file location can be:

- a local file;
- an HTTP URL;

.. note::
    The adapter helps developers create infinite video streams for benchmarking, demonstrating, and testing purposes. It allows configuring a frame loss ratio to test processing in an unstable environment.

**Parameters**:

- ``LOCATION``: a video file local path or URL;
- ``EOS_ON_LOOP_END``: a flag indicating whether to send ``EOS`` message at the end of each loop; default is ``False``;
- ``READ_METADATA``: a flag indicating the need to augment the stream with metadata from a JSON file corresponding to the source file; default is ``False``;
- ``SYNC_OUTPUT``: a flag indicating the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``DOWNLOAD_PATH``: a directory to download the file from remote storage before playing it;
- ``LOSS_RATE``: a probability to drop the frames.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-video-loop-test \
        --entrypoint /opt/savant/adapters/gst/sources/video_loop.sh \
        -e SYNC_OUTPUT=True \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e LOCATION=/path/to/data/test.mp4 \
        -e DOWNLOAD_PATH=/tmp/video-loop-source-downloads \
        -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        -v /tmp/video-loop-source-downloads:/tmp/video-loop-source-downloads \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py video-loop --source-id=test /path/to/data/test.mp4

FFmpeg Source Adapter
^^^^^^^^^^^^^^^^^^^^^

The adapter delivers video stream using FFmpeg library. It can be used to read video files, RTSP streams, and other sources supported by FFmpeg.

**Parameters**:

* ``URI`` (**required**): an URI of the stream;
* ``FFMPEG_PARAMS``: a comma separated string ``key=value`` with parameters for FFmpeg (e.g. ``rtsp_transport=tcp``, ``input_format=mjpeg,video_size=1280x720``);
* ``FFMPEG_LOGLEVEL``: a log level for FFmpeg; default is ``info``;
* ``BUFFER_LEN``: a maximum amount of frames in FFmpeg buffer; default is ``50``;
* ``SYNC_OUTPUT``: a flag indicating the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
* ``SYNC_DELAY``: a delay in seconds before sending frames; default is ``0``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-ffmpeg-test \
        --entrypoint /opt/savant/adapters/gst/sources/ffmpeg.sh \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e URI=rtsp://192.168.1.1 \
        -e FFMPEG_PARAMS=rtsp_transport=tcp \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py ffmpeg --source-id=test --ffmpeg-params=input_format=mjpeg,video_size=1280x720 --device=/dev/video0 /dev/video0

RTSP Source Adapter
^^^^^^^^^^^^^^^^^^^

The RTSP Source Adapter delivers RTSP stream to a module.

**Parameters**:

- ``RTSP_URI`` (**required**): an RTSP URI of the stream;
- ``SYNC_OUTPUT``: a flag indicating the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``;
- ``SYNC_DELAY``: a delay in seconds before sending frames; when the source has ``B``-frames the flag allows avoiding sending frames in batches; default is ``0``;
- ``RTSP_TRANSPORT``: a transport protocol to use; default is ``tcp``;
- ``BUFFER_LEN``: a maximum amount of frames in the buffer; default is ``50``;

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-rtsp-test \
        --entrypoint /opt/savant/adapters/gst/sources/rtsp.sh \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e RTSP_URI=rtsp://192.168.1.1 \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py rtsp --source-id=test rtsp://192.168.1.1

USB Cam Source Adapter
^^^^^^^^^^^^^^^^^^^^^^

The USB/CSI cam source adapter captures frames from a V4L2-compatible device. Savant does not include a dedicated USB- or CSI-cam adapter; instead, it uses the the FFmpeg source adapter to capture frames from the device.

In the compose:

.. code-block:: yaml

  usb-cam:
    image: ghcr.io/insight-platform/savant-adapters-gstreamer:latest
    restart: unless-stopped
    volumes:
      - zmq_sockets:/tmp/zmq-sockets
    environment:
      - URI=/dev/video0
      - FFMPEG_PARAMS=input_format=mjpeg,video_size=1920x1080
      - ZMQ_ENDPOINT=pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc
      - SOURCE_ID=video
    devices:
      - /dev/video0:/dev/video0
    entrypoint: /opt/savant/adapters/gst/sources/ffmpeg.sh
    depends_on:
      module:
        condition: service_healthy

In plain Docker:

.. code-block:: bash

    docker run --rm -it --name source-usb-cam-test \
        --entrypoint /opt/savant/adapters/gst/sources/ffmpeg.sh \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e URI=/dev/video0 \
        -e FFMPEG_PARAMS=input_format=mjpeg,video_size=1920x1080 \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        --device=/dev/video0:/dev/video0 \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

        ./scripts/run_source.py ffmpeg --source-id=test --ffmpeg-params=input_format=mjpeg,video_size=1920x1080 --device=/dev/video0 /dev/video0

.. note::
    You can work with USB/CSI cameras that are compatible with the V4L2 API. The adapter uses FFmpeg to capture frames from the device. Currently, cameras supporting MJPEG, raw color formats, and H.264/HEVC are supported.

Related articles in the Savant blog:

- `How To Work With MJPEG USB Camera in Savant <https://b.savant-ai.io/2024/03/14/how-to-work-with-mjpeg-usb-camera-in-savant/>`_;
- `Emulating USB Camera In Linux With FFmpeg and V4L2 Loopback <https://b.savant-ai.io/2024/02/23/emulating-usb-camera-in-linux/>`_.

GigE Vision Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::
    Additional information on `GigE Vision cameras support <https://blog.savant-ai.io/savant-explained-gige-vision-camera-support-a4e472275280?source=friends_link&sk=eb341e416b32696bc781f4cfb62ef2e1>`_ in the blog.

The adapter is designed to take video streams from Ethernet GigE Vision industrial cams. It passes the frames captured from the camera to the module without encoding (`#18 <https://github.com/insight-platform/Savant/issues/18>`__) which may introduce significant network load. We recommend using it locally with the module deployed at the same host.

**Parameters**:

* ``WIDTH``: the width of the video frame, in pixels;
* ``HEIGHT``: the height of the video frame, in pixels;
* ``FRAMERATE``: the framerate of the video stream, in frames per second;
* ``INPUT_CAPS``: the format of the video stream, in GStreamer caps format (e.g. video/x-raw,format=RGB);
* ``PACKET_SIZE``: the packet size for GigEVision cameras, in bytes;
* ``AUTO_PACKET_SIZE``: whether to negotiate the packet size automatically for GigEVision cameras;
* ``EXPOSURE``: the exposure time for the camera, in microseconds;
* ``EXPOSURE_AUTO``: the auto exposure mode for the camera, one of ``off``, ``once``, or ``on``;
* ``GAIN``: the gain for the camera, in decibels;
* ``GAIN_AUTO``: the auto gain mode for the camera, one of ``off``, ``once``, or ``on``;
* ``FEATURES``: additional configuration parameters for the camera, as a space-separated list of features;
* ``HOST_NETWORK``: host network to use;
* ``CAMERA_NAME``: name of the camera, in the format specified in the command description;
* ``ENCODE``: a flag indicating the need to encode video stream with HEVC codec; default is ``False``;
* ``ENCODE_BITRATE``: the bitrate for the encoded video stream, in kbit/sec; default is ``2048``;
* ``ENCODE_KEY_INT_MAX``: the maximum interval between two keyframes, in frames; default is ``30``;
* ``ENCODE_SPEED_PRESET``: preset name for speed/quality tradeoff options; one of ``ultrafast``, ``superfast``, ``veryfast``, ``faster``, ``fast``, ``medium``, ``slow``, ``slower``, ``veryslow``, ``placebo``; default is ``medium``;
* ``ENCODE_TUNE``: preset name for tuning options; one of ``psnr``, ``ssim``, ``grain``, ``zerolatency``, ``fastdecode``, ``animation``; default is ``zerolatency``.


Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-gige-test \
        --entrypoint /opt/savant/adapters/gst/sources/gige_cam.sh \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e CAMERA_NAME=test-camera \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py gige --source-id=test test-camera

.. _multi_stream_source_adapter:

Multi-stream Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Multi-stream Source Adapter sends the same video file to multiple streams.

.. note::

    The purpose of the adapter is to benchmark modules under the load to find out their FPS capacity.

The file location is:

- a local file;
- an HTTP URL;

**Parameters**:

- ``LOCATION`` (**required**): a video file local path or URL;
- ``SOURCE_ID_PATTERN``: a pattern for stream source identifiers; use ``%d``, ``%03d`` placeholders for stream idx. Default is ``source-%d``, usually no need to change it;
- ``NUMBER_OF_STREAMS``: a number of parallel streams; default is ``1``;
- ``NUMBER_OF_FRAMES``: a number of frames to be sent to each stream; if not specified, all frames from the video file will be sent;
- ``SHUTDOWN_AUTH``: an authentication key to shutdown the module after all frames were sent. Must match ``parameters.shutdown_auth`` in the module configuration to have an effect;
- ``READ_METADATA``: a flag indicating the need to augment the stream with metadata from a JSON file corresponding to the source file; default is ``False``;
- ``SYNC_OUTPUT``: a flag indicating the need to send frames from source synchronously (i.e. at the source file rate); default is ``False``; the parameter can be used to simulate real-time sources like RTSP-cams;
- ``DOWNLOAD_PATH``: a directory to download the file from remote storage before playing it.

.. note::

    The adapter doesn't have ``SOURCE_ID`` parameter.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-multi-stream-test \
        --entrypoint /opt/savant/adapters/gst/sources/multi_stream.sh \
        -e SYNC_OUTPUT=True \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID_PATTERN='camera-%d' \
        -e NUMBER_OF_STREAMS=4 \
        -e SHUTDOWN_AUTH=shutdown-key \
        -e LOCATION=/path/to/data/test.mp4 \
        -e DOWNLOAD_PATH=/tmp/video-loop-source-downloads \
        -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        -v /tmp/video-loop-source-downloads:/tmp/video-loop-source-downloads \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py multi-stream --source-id-pattern='camera-%d' --number-of-sources=4 --shutdown-auth=shutdown-key /path/to/data/test.mp4

Kafka-Redis Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Kafka-Redis Source Adapter takes video stream metadata from Kafka and fetches frame content from Redis. Frame content location is encoded as ``<redis-host>:<redis-port>:<redis-db>/<redis-key>``.


**Parameters**:

- ``KAFKA_BROKERS`` (**required**): a comma-separated list of Kafka brokers;
- ``KAFKA_TOPIC`` (**required**): a Kafka topic to read messages from;
- ``KAFKA_GROUP_ID`` (**required**): a Kafka consumer group ID;
- ``KAFKA_CREATE_TOPIC``: a flag indicating whether to create a Kafka topic if it does not exist; default is ``False``;
- ``KAFKA_CREATE_TOPIC_NUM_PARTITIONS``: a number of partitions for a Kafka topic to create; default is ``1``;
- ``KAFKA_CREATE_TOPIC_REPLICATION_FACTOR``: a replication factor for a Kafka topic to create; default is ``1``;
- ``KAFKA_CREATE_TOPIC_CONFIG``: a json dict of a Kafka topic configuration, see `topic configs <https://kafka.apache.org/documentation.html#topicconfigs>`__ (e.g. ``{"retention.ms": 300000}``); default is ``{}``;
- ``KAFKA_POLL_TIMEOUT``: a timeout for Kafka consumer poll, in seconds; default is ``1``;
- ``KAFKA_AUTO_COMMIT_INTERVAL_MS``: a frequency in milliseconds that the consumer offsets are auto-committed to Kafka; default is ``1000``;
- ``KAFKA_AUTO_OFFSET_RESET``: a position to start reading messages from Kafka topic when the group is created; default is ``latest``;
- ``KAFKA_PARTITION_ASSIGNMENT_STRATEGY``: a strategy to assign partitions to consumers; default is ``roundrobin``;
- ``KAFKA_MAX_POLL_INTERVAL_MS``: a maximum delay in milliseconds between invocations of poll() when using consumer group management; default is ``300000``;
- ``QUEUE_SIZE``: a maximum amount of messages in the queue; default is ``50``.

.. note::
    The adapter doesn't have ``SOURCE_ID``, ``ZMQ_TYPE``, ``ZMQ_BIND``, ``USE_ABSOLUTE_TIMESTAMPS`` parameters.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-kafka-redis-test \
        --entrypoint python \
        -e ZMQ_ENDPOINT=pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e KAFKA_BROKERS=kafka:9092 \
        -e KAFKA_TOPIC=kafka-redis-adapter-demo \
        -e KAFKA_GROUP_ID=kafka-redis-adapter-demo \
        -e KAFKA_CREATE_TOPIC=True \
        -e KAFKA_CREATE_TOPIC_NUM_PARTITIONS=4 \
        -e KAFKA_CREATE_TOPIC_REPLICATION_FACTOR=1 \
        -e 'KAFKA_CREATE_TOPIC_CONFIG={"retention.ms": 300000}' \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-py:latest \
        -m adapters.python.sources.kafka_redis



Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py kafka-redis --brokers=kafka:9092 --topic=kafka-redis-adapter-demo --group-id=kafka-redis-adapter-demo

Kinesis Video Stream Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Kinesis Video Stream Source Adapter takes video frames from Kinesis Video Stream.

.. list-table:: Parameters
    :header-rows: 1

    * - Parameter
      - Description
      - Default
      - Example
    * - ``AWS_REGION``
      - An AWS region.
      - Unset
      - ``us-west-2``
    * - ``AWS_ACCESS_KEY``
      - An AWS access key ID.
      - Unset
      - ``AKIAIOSFODNN7EXAMPLE``
    * - ``AWS_SECRET_KEY``
      - An AWS secret access key.
      - Unset
      - ``wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY``
    * - ``STREAM_NAME``
      - Name of the Kinesis Video Stream.
      - Unset
      - ``test-stream``
    * - ``TIMESTAMP``
      - Either timestamp in format `%Y-%m-%dT%H:%M:%S` or delay from current time in `-<delay>(s\|m)`. E.g. `2024-03-12T06:57:00`, `-30s`, `-1m`.
      - Unset
      - ``-1m``
    * - ``SYNC_OUTPUT``
      - A flag specifying if to send frames synchronously (i.e. at the source file rate).
      - ``False``
      - ``True``
    * - ``PLAYING``
      - A flag specifying if the stream should start playing immediately.
      - ``True``
      - ``False``
    * - ``API_PORT``
      - A port for REST API.
      - ``18367``
      - ``9999``
    * - ``SAVE_STATE``
      - A flag indicating whether to save state to the state file.
      - ``False``
      - ``True``
    * - ``STATE_PATH``
      - A path to the state file.
      - ``state.json``
      - ``/foo/bar.json``

.. note::
    The adapter doesn't have ``ZMQ_TYPE``, ``ZMQ_BIND``, ``USE_ABSOLUTE_TIMESTAMPS`` parameters.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name source-kinesis-video-stream-test \
        --entrypoint python \
        -p 18367:18367 \
        -e ZMQ_ENDPOINT=pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e AWS_REGION=us-west-2 \
        -e AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE \
        -e AWS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
        -e STREAM_NAME=test-stream \
        -e TIMESTAMP=-1m \
        -e SYNC_OUTPUT=True \
        -e PLAYING=True \
        -e API_PORT=18367 \
        -e SAVE_STATE=True \
        -e STATE_PATH=/state/test.json \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        -v "$(pwd)/source-kinesis-video-stream-test:/state" \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest \
        -m adapters.python.sources.kvs

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py kvs \
        --aws-region=us-west-2 \
        --aws-access-key='AKIAIOSFODNN7EXAMPLE' \
        --aws-secret-key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY' \
        --stream-name=test-stream \
        --timestamp=-1m \
        --source-id=test

REST API
""""""""

The adapter provides a REST API to control the stream. The API is available at ``http://<host>:<API_PORT>``. The OpenAPI documentation is available at ``http://<host>:<API_PORT>/docs``.

The API provides the following endpoints:

``PUT /stream``: Update stream configuration.

**Request body:**

- ``name (string)``: name of the Kinesis Video Stream;
- ``source_id (string)``: source ID;
- ``timestamp (string)``: timestamp in format `%Y-%m-%dT%H:%M:%S`;
- ``credentials (object)``: AWS credentials:

  - ``region (string)``: AWS region;
  - ``access_key (string)``: AWS access key ID;
  - ``secret_key (string)``: AWS secret access key;

- ``is_playing (bool)``: a flag specifying if the stream should start playing immediately.

.. note::

    If any of the parameters is not specified, the value from the current stream configuration will be used.

Response body the same as the request body.

Example:

.. code-block:: bash

    curl -X PUT 'http://localhost:18367/stream' \
        -H "Content-Type: application/json" \
        -d '{
        "name": "test-stream",
        "source_id": "test",
        "timestamp": "2024-03-12T06:57:00",
        "credentials": {
            "region": "us-west-2",
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        },
        "is_playing": true
    }'

.. code-block:: json

    {
        "name": "test-stream",
        "source_id": "test",
        "timestamp": "2024-03-12T06:57:00",
        "is_playing": true
    }

``GET /stream``: Get current stream configuration.

Response body the same as for the ``PUT /streams`` request.

Example:

.. code-block:: bash

    curl -X GET 'http://localhost:18367/stream'

.. code-block:: json

    {
        "name": "test-stream",
        "source_id": "test",
        "timestamp": "2024-03-12T06:57:00",
        "is_playing": true
    }

``PUT /stream/play``: Start playing the stream.

The request has no body.

Response body the same as for the ``PUT /streams`` request.

Example:

.. code-block:: bash

    curl -X PUT 'http://localhost:18367/stream/play'

.. code-block:: json

    {
        "name": "test-stream",
        "source_id": "test",
        "timestamp": "2024-03-12T06:57:00",
        "is_playing": true
    }

``PUT /stream/stop``: Stop playing the stream.

The request has no body.

Response body the same as for the ``PUT /streams`` request.

Example:

.. code-block:: bash

    curl -X PUT 'http://localhost:18367/stream/stop'

.. code-block:: json

    {
        "name": "test-stream",
        "source_id": "test",
        "timestamp": "2024-03-12T06:57:00",
        "is_playing": false
    }

Message Dump Player Source Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Message Dump Player Adapter plays video dumps sequentially from a playlist file and sends them to a module.
Playlist file contains a list of message dump files, one per line.
It's one shot adapter, i.e. it stops after playing all files from the playlist.

**Parameters**:

- ``PLAYLIST_PATH``: a path to the playlist file;
- ``SYNC_OUTPUT``: flag specifying if to send frames synchronously (i.e. at the source file rate); default is ``False``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name message-dump-player-test \
        --entrypoint python \
        -e PLAYLIST_PATH=/path/to/playlist/file.txt \
        -e SYNC_OUTPUT=False \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -v /path/to/playlist/file.txt:/path/to/playlist/file.txt:ro \
        -v /path/to/dump/files:/path/to/dump/files \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-py:latest \
        -m adapters.python.sources.message_dump_player

Running with the helper script:

.. code-block:: bash

    ./scripts/run_source.py mdp --playlist /path/to/playlist/file.txt --dump-files-dir /path/to/dump/files

Sink Adapters
-------------

There is a number of sink adapters implemented:

- JSON Metadata;
- Image File;
- Video File;
- Display;
- Always-On RTSP;
- Kafka-Redis;
- Multistream Kinesis Video Stream.

All sync adapters accept the following parameters:

- ``ZMQ_ENDPOINT``: a ZeroMQ socket for data input matching the one specified in module's output;  the endpoint coding scheme is ``[<socket_type>+(bind|connect):]<endpoint>``;
- ``ZMQ_TYPE``: a ZeroMQ socket type for the adapter's input; the default value is ``SUB``, can also be set to ROUTER or ``REP``; **warning**: this parameter is deprecated, consider encoding the type in ``ZMQ_ENDPOINT``;
- ``ZMQ_BIND``: a parameter specifying whether the adapter's input should be bound or connected to the specified endpoint; If ``True``, the input is bound; otherwise, it's connected; the default value is ``False``; **warning**: this parameter is deprecated, consider encoding the type in ``ZMQ_ENDPOINT``.

JSON Metadata Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^

The JSON Metadata Sink Adapter writes received messages as newline-delimited JSON streaming files specified as:

- a local path to a single file;
- a local path with substitution patterns:

  a. ``%source_id`` inserts ``SOURCE_ID`` value into resulting filename;
  b. ``%src_filename`` inserts source filename into resulting filename.

**Parameters**:

- ``DIR_LOCATION``: a location to write files to; can be a plain location or a pattern; supported substitution parameters are ``%source_id`` and ``%src_filename``;
- ``CHUNK_SIZE``: a chunk size in a number of frames; the stream is split into chunks and is written to separate folders with consecutive numbering; default is ``10000``; a value of ``0`` disables chunking, resulting in a continuous stream of frames by ``source_id``;
- ``SKIP_FRAMES_WITHOUT_OBJECTS``: a flag indicating whether frames without detected objects are ignored in output; the default value is ``False``;
- ``SOURCE_ID``: an optional filter to filter out frames with a specific ``source_id`` only;
- ``SOURCE_ID_PREFIX`` an optional filter to filter out frames with a matching ``source_id`` prefix only.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/savant/adapters/python/sinks/metadata_json.py \
    -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
    -e LOCATION=/path/to/output/%source_id-%src_filename \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    ghcr.io/insight-platform/savant-adapters-py:latest


Running with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py meta-json /path/to/output/%source_id-%src_filename


Image File Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^

The image file sink adapter extends the JSON metadata adapter by writing image files along with metadata JSON files to a directory specified with ``DIR_LOCATION``.

**Parameters**:

- ``DIR_LOCATION``: a location to write files to; can be a regular path or a path template; supported substitution parameters are ``%source_id`` and ``%src_filename``;
- ``CHUNK_SIZE``: a chunk size in a number of frames; the stream is split into chunks and is written to separate directories with consecutive numbering; default is ``10000``; A value of ``0`` disables chunking, resulting in a continuous stream of frames by ``source_id``;
- ``SKIP_FRAMES_WITHOUT_OBJECTS``: a flag indicating whether frames without objects are ignored in output; the default value is ``False``;
- ``SOURCE_ID``: an optional filter to filter out frames with a specific ``source_id`` only;
- ``SOURCE_ID_PREFIX`` an optional filter to filter out frames with a matching ``source_id`` prefix only.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
        --entrypoint /opt/savant/adapters/python/sinks/image_files.py \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
        -e CHUNK_SIZE=0 \
        -v /path/to/output/:/path/to/output/ \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-py:latest


Running with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py image-files /path/to/output/%source_id-%src_filename

Video File Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^

The video file sink adapter extends the JSON metadata adapter by writing video files along with metadata JSON files to a directory specified with ``DIR_LOCATION``.

**Parameters**:

- ``DIR_LOCATION``: a location to write files to; can be a regular path or a path template; supported substitution parameters are ``%source_id`` and ``%src_filename``;
- ``CHUNK_SIZE``: a chunk size in a number of frames; the stream is split into chunks and is written to separate folders with consecutive numbering; default is ``10000``; A value of ``0`` disables limit for number of frames in a chunk: the stream will be split into chunks only by EOS messages;
- ``SOURCE_ID``: an optional filter to filter out frames with a specific ``source_id`` only;
- ``SOURCE_ID_PREFIX`` an optional filter to filter out frames with a matching ``source_id`` prefix only.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
        --entrypoint /opt/savant/adapters/gst/sinks/video_files.py \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
        -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
        -e CHUNK_SIZE=0 \
        -v /path/to/output/:/path/to/output/ \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py video-files /path/to/output/%source_id-%src_filename

Display Sink Adapter
^^^^^^^^^^^^^^^^^^^^

The Display Sink Adapter is a visualizing adapter designed for development purposes. To use this adapter, you need a working X server and monitor. The adapter is used with synchronous streams, so for expected operation, the data source must be served with ``SYNC=True``. The adapter also allows specifying the ``SYNC`` flag, but it is better to configure it on the source side.

**Parameters**:

- ``CLOSING_DELAY``: a delay in seconds before closing the window after the video stream has finished, the default value is ``0``;
- ``SYNC_INPUT``: a flag indicating whether to show the frames on the sink synchronously with the source (i.e., at the source file rate); if you are intending to use ``SYNC`` processing, consider ``DEALER/ROUTER`` or ``REQ/REP`` sockets, because ``PUB/SUB`` may drop packets when queues are overflown;
- ``SOURCE_ID``: an optional filter to filter out frames with a specific ``source_id`` only;
- ``SOURCE_ID_PREFIX``: an optional filter to filter out frames with a ``source_id`` prefix only.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-display \
        --entrypoint /opt/savant/adapters/ds/sinks/display.py \
        -e SYNC_INPUT=False \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e DISPLAY \
        -e XAUTHORITY=/tmp/.docker.xauth \
        -e CLOSING_DELAY=0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /tmp/.docker.xauth:/tmp/.docker.xauth \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        --gpus=all \
        ghcr.io/insight-platform/savant-adapters-deepstream:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py display

Always-On RTSP Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Always-On RTSP Sink Adapter broadcasts the video stream as RTSP/LL-HLS/WebRTC.

This adapter **always** performs transcoding of the incoming stream to ensure continuous streaming even when its source stops operating. In this case, the adapter continues to stream a static image waiting for the source to resume sending data.

When Nvidia Runtime is available this adapter uses DeepStream SDK and performs hardware transcoding and scaling of the incoming stream, otherwise it performs software transcoding and scaling. Software-based encoding/decoding must be used only when hardware-based encoding is not available (Jetson Orin Nano, A100, H100). If the hardware-based encoding is available, run the adapter with Nvidia runtime enabled to activate hardware-based decoding/encoding.

The adapter provides API to control video streams. API is available at ``http://<container-host>:<API_PORT>``. API documentation is available at ``http://<container-host>:<API_PORT>``.

.. note::

    Software-based encoding/decoding is available only for ``H264`` codec.

The simplified design of the adapter is depicted in the following diagram:

.. image:: ../_static/img/10_adapters_ao_rtsp.png

.. note::

    We use Always-On RTSP Adapter in our demos. Take a look at one of `them <https://github.com/insight-platform/Savant/tree/develop/samples/opencv_cuda_bg_remover_mog2>`__ to get acquainted with its use.

.. list-table:: Parameters
    :header-rows: 1

    * - Parameter
      - Description
      - Default
      - Example
    * - ``RTSP_URI``
      - A URI of the RTSP server where to cast the stream, this parameter is required only when ``DEV_MODE=False``.
        The sink sends video stream to ``RTSP_URI/{source-id}``.
      - Unset
      - ``rtsp://1.1.1.1:554``
    * - ``DEV_MODE``
      - Enables the use of embedded MediaMTX to serve a stream.
      - ``False``
      - ``True``
    * - ``STUB_FILE_LOCATION``
      - The location of a stub image file; the image file must be in ``JPEG`` format, this parameter is required; the stub image file is shown when there is no input data; its dimensions define the resolution of the output stream.
      - Unset
      - ``/path/to/stub_file/test.jpg``
    * - ``MAX_DELAY_MS``
      - A maximum delay in milliseconds to wait after the last frame received before the stub image is displayed.
      - ``1000``
      - ``5000``
    * - ``TRANSFER_MODE``
      - A transfer mode specification; one of: ``scale-to-fit``, ``crop-to-fit``; the parameter defines how the incoming video stream is mapped to the resulting stream.
      - ``scale-to-fit``
      - ``crop-to-fit``
    * - ``PROTOCOLS``
      - Enabled transport protocols; the parameter is required only when ``DEV_MODE=False``.
      - ``tcp``
      - ``tcp+udp``
    * - ``LATENCY_MS``
      - The resulting RTSP stream buffer size in milliseconds.
      - ``100``
      - ``1000``
    * - ``KEEP_ALIVE``
      - Whether to send RTSP keep alive packets; set it to ``False`` for old incompatible server.
      - ``True``
      - ``False``
    * - ``CODEC``
      - An encoding codec; one of: ``h264``, ``hevc``.
      - ``h264`` (browser-compatible)
      - ``hevc``
    * - ``ENCODER_PROFILE``
      - An encoding profile:
          - For ``h264``, is one of:
              - ``Baseline``
              - ``Main``
              - ``High``
          - For ``hevc``, is one of:
              - ``Main``
              - ``Main10``
              - ``FREXT``
      - Depending on the selected codec:
          - ``h264```: ``High``
          - ``hevc``: ``Main``
      - ``Main10``
    * - ``ENCODER_BITRATE``
      - An encoding bitrate in bit/s.
      - ``4000000``
      - ``8000000``
    * - ``FRAMERATE``
      - A frame rate for the output stream.
      - ``30/1``
      - ``60/1``
    * - ``IDR_PERIOD_FRAMES``
      - A period of I-frame insertion;
      - ``30``
      - ``60``
    * - ``METADATA_OUTPUT``
      - Where to dump metadata; one of: ``stdout``, ``logger``.
      - Unset
      - ``logger``
    * - ``SYNC_INPUT``
      - A flag indicating whether to show frames on sink synchronously (i.e. at the source rate).
      - ``True``
      - ``False``
    * - ``REALTIME``
      - A flag indicating whether to synchronise frames at realtime (i.e. using absolute timestamps); ignored when ``SYNC_INPUT=False``.
      - ``False``
      - ``True``
    * - ``SYNC_OFFSET_MS``
      - An offset in milliseconds to adjust the synchronisation. Tune this parameter to play video more smoothly. When ``REALTIME=False``, the offset is applied to the timestamp of the first frame; when ``REALTIME=True``, the offset is applied to the current time. Ignored when ``SYNC_INPUT=False``.
      - ``1000``
      - ``5000``
    * - ``SYNC_QUEUE_SIZE``
      - A size of queue for frames to be synchronised; ignored when ``SYNC_INPUT=False``. Tune this parameter according to the stream framerate and ``SYNC_OFFSET_MS``.
      - ``500``
      - ``1000``
    * - ``SOURCE_ID``
      - A filter to receive frames with a specific ``source_id`` only (at the start of the adapter, when no other streams are configured with the REST API). This parameter is ignored when ``SOURCE_IDS`` is specified.
      - Unset
      - ``test``
    * - ``SOURCE_IDS``
      - A filter to receive frames with specific ``source_id``-s only (at the start of the adapter, when no other streams are configured with the REST API).
      - Unset
      - ``test1,test2``
    * - ``MAX_RESOLUTION``
      - Maximum resolution of the incoming stream; if the resolution is greater than the allowed resolution, the video stream will terminate; you can override the max allowed resolution be setting width and height of frames.
      - ``3840x2152``
      - ``1920x1080``
    * - ``API_PORT``
      - A port for the stream control REST API.
      - ``13000``
      - ``12345``
    * - ``FAIL_ON_STREAM_ERROR``
      - A flag indicating whether to stop the adapter when a stream is failed.
      - ``True``
      - ``False``
    * - ``STATUS_POLL_INTERVAL_MS``
      - An interval in milliseconds to poll statuses of the streams.
      - ``1000``
      - ``500``

When ``DEV_MODE=True`` the stream is available at:

- RTSP: ``rtsp://<container-host>:554/stream/{source-id}``
- RTMP: ``rtmp://<container-host>:1935/stream/{source-id}``
- LL-HLS: ``http://<container-host>:888/stream/{source-id}``
- WebRTC: ``http://<container-host>:8889/stream/{source-id}``

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-always-on-rtsp \
        --gpus=all \
        --entrypoint python \
        -e SYNC_INPUT=True \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e SOURCE_ID=test \
        -e STUB_FILE_LOCATION=/path/to/stub_file/test.jpg \
        -e MAX_DELAY_MS=1000 \
        -e TRANSFER_MODE=scale-to-fit \
        -e RTSP_URI=rtsp://192.168.1.1 \
        -e RTSP_PROTOCOLS=tcp \
        -e RTSP_LATENCY_MS=100 \
        -e RTSP_KEEP_ALIVE=True \
        -e ENCODER_PROFILE=High \
        -e ENCODER_BITRATE=4000000 \
        -e FRAMERATE=30/1 \
        -e API_PORT=13000 \
        -p 13000:13000 \
        -v /path/to/stub_file/test.jpg:/path/to/stub_file/test.jpg:ro \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-deepstream:latest \
        -m adapters.ds.sinks.always_on_rtsp

Running the adapter with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py always-on-rtsp --source-id=test --stub-file-location=/path/to/stub_file/test.jpg rtsp://192.168.1.1

Stream control API
""""""""""""""""""

The Always-On RTSP Sink Adapter provides an API to control video streams. The API is available at ``http://<container-host>:<API_PORT>``. The OpenAPI documentation is available at ``http://<container-host>:<API_PORT>/docs``.

The API provides the following endpoints:

``PUT /streams/{source_id}``: Create a new stream and start it.

**Path parameters:**

- ``source_id`` (**required**): a source ID.

**Request body:**

- ``stub_file (string)``: a location of a stub image file; the image file must be in ``JPEG`` format;
- ``framerate (string)``: a frame rate for the output stream;
- ``idr_period (integer)``: a period of I-frame insertion;
- ``codec (string)``: an encoding codec; one of: ``h264``, ``hevc``;
- ``bitrate (integer)``: an encoding bitrate in bit/s;
- ``profile (string)``: an encoding profile. For ``h264`` one of: ``Baseline``, ``Main``, ``High``; for ``hevc`` one of: ``Main``, ``Main10``, ``FREXT``;
- ``max_delay_ms (integer)``: a maximum delay in milliseconds to wait after the last frame received before the stub image is displayed;
- ``latency_ms (integer)``: resulting RTSP stream buffer size in ms;
- ``transfer_mode (string)``: a transfer mode specification; one of: ``scale-to-fit``, ``crop-to-fit``; the parameter defines how the incoming video stream is mapped to the resulting stream;
- ``rtsp_keep_alive (boolean)``: whether to send RTSP keep alive packets;
- ``metadata_output (string)``: where to dump metadata; one of: ``stdout``, ``logger``;
- ``sync_input (boolean)``: a flag indicates whether to show frames on sink synchronously (i.e. at the source rate).

.. note::

    If any of the parameters is not specified, the value from the adapter parameters will be used.

**Response body:**

- all the fields from the request body;
- ``status (object)``: a status of the stream:

  - ``is_alive (boolean)``: a flag indicating whether the stream is alive;
  - ``exit_code (integer)``: an exit code of the stream in case of failure.

Examples:

.. code-block:: bash

    curl -X PUT 'http://localhost:13000/streams/test' \
        -H 'Content-Type: application/json' \
        -d '{
        "stub_file": "/stub_imgs/smpte100_640x360.jpeg",
        "framerate": "30/1",
        "idr_period": 15,
        "codec": "hevc",
        "bitrate": 4000000,
        "profile": "Main",
        "max_delay_ms": 1000,
        "latency_ms": 100,
        "transfer_mode": "scale-to-fit",
        "rtsp_keep_alive": true,
        "metadata_output": "stdout",
        "sync_input": false
    }'

.. code-block:: json

    {
      "stub_file": "/stub_imgs/smpte100_640x360.jpeg",
      "framerate": "30/1",
      "idr_period": 15,
      "bitrate": 4000000,
      "profile": "Main",
      "codec": "hevc",
      "max_delay_ms": 1000,
      "latency_ms": 100,
      "transfer_mode": "scale-to-fit",
      "rtsp_keep_alive": true,
      "metadata_output": "stdout",
      "sync_input": false,
      "status": {
        "is_alive": true,
        "exit_code": null
      }
    }

.. code-block:: bash

    curl -X PUT 'http://localhost:13000/streams/test2' \
        -H 'Content-Type: application/json' \
        -d '{}'

.. code-block:: json

    {
      "stub_file": "/stub_imgs/smpte100_1280x720.jpeg",
      "framerate": "20/1",
      "idr_period": 20,
      "bitrate": 4000000,
      "profile": "High",
      "codec": "h264",
      "max_delay_ms": 1000,
      "latency_ms": 100,
      "transfer_mode": "scale-to-fit",
      "rtsp_keep_alive": true,
      "metadata_output": null,
      "sync_input": false,
      "status": {
        "is_alive": true,
        "exit_code": null
      }
    }

``GET /streams/{source_id}``: get configuration of a stream.

**Path parameters:**

- ``source_id`` (**required**): a source ID.

**Query parameters:**

- ``format``: a response format; one of: ``json``, ``yaml``; the default value is ``json``.

Response body the same as for the ``PUT /streams/{source_id}`` request.

Example:

.. code-block:: bash

    curl 'http://localhost:13000/streams/test?format=json'

Response:

.. code-block:: json

    {
      "stub_file": "/stub_imgs/smpte100_640x360.jpeg",
      "framerate": "30/1",
      "idr_period": 30,
      "bitrate": 4000000,
      "profile": "Main",
      "codec": "hevc",
      "max_delay_ms": 1000,
      "latency_ms": 100,
      "transfer_mode": "scale-to-fit",
      "rtsp_keep_alive": true,
      "metadata_output": "stdout",
      "sync_input": false,
      "status": {
        "is_alive": true,
        "exit_code": null
      }
    }

``GET /streams``: list configurations of all streams.

**Query parameters:**

- ``format``: a response format; one of: ``json``, ``yaml``; the default value is ``json``.

Example:

.. code-block:: bash

    curl 'http://localhost:13000/streams?format=yaml'

Response:

.. code-block:: yaml

    test:
      stub_file: /stub_imgs/smpte100_640x360.jpeg
      framerate: 30/1
      idr_period: 30
      codec: hevc
      bitrate: 4000000
      profile: Main
      max_delay_ms: 1000
      latency_ms: 100
      transfer_mode: scale-to-fit
      rtsp_keep_alive: true
      metadata_output: stdout
      sync_input: false
      status:
        is_alive: true
        exit_code: null

``DELETE /streams/{source_id}``: stop and delete a stream.

**Path parameters:**

- ``source_id`` (**required**): a source ID.

Example:

.. code-block:: bash

    curl -X DELETE 'http://localhost:13000/streams/test'

Response:

.. code-block:: json

    "ok"

Kafka-Redis Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^^

The Kafka-Redis Sink Adapter sends video stream metadata to Kafka and frame content to Redis. Frame content location is encoded as ``<redis-host>:<redis-port>:<redis-db>/<redis-key>``. ``<redis-key>`` is in format ``REDIS_KEY_PREFIX:UUID`` where ``UUID`` is a unique identifier of the video frame. When Redis is not configured (i.e. ``REDIS_HOST`` is not set) the adapter will store frame content internally in the video frame.

**Parameters**:

- ``KAFKA_BROKERS`` (**required**): a comma-separated list of Kafka brokers;
- ``KAFKA_TOPIC`` (**required**): a Kafka topic to put messages to;
- ``KAFKA_FLUSH_INTERVAL``: a flush interval in seconds for Kafka producer; default is ``1``;
- ``KAFKA_FLUSH_TIMEOUT``: a flush timeout in seconds for Kafka producer; default is ``10``;
- ``KAFKA_CREATE_TOPIC``: a flag indicating whether to create a Kafka topic if it does not exist; default is ``False``;
- ``KAFKA_CREATE_TOPIC_NUM_PARTITIONS``: a number of partitions for a Kafka topic to create; default is ``1``;
- ``KAFKA_CREATE_TOPIC_REPLICATION_FACTOR``: a replication factor for a Kafka topic to create; default is ``1``;
- ``KAFKA_CREATE_TOPIC_CONFIG``: a json dict of a Kafka topic configuration, see `topic configs <https://kafka.apache.org/documentation.html#topicconfigs>`__ (e.g. ``{"retention.ms": 300000}``); default is ``{}``;
- ``REDIS_HOST``: a Redis host;
- ``REDIS_PORT``: a Redis port; default is ``6379``;
- ``REDIS_DB``: a Redis database; default is ``0``;
- ``REDIS_KEY_PREFIX``: a prefix for Redis keys; frame content is put to Redis with a key ``REDIS_KEY_PREFIX:UUID``; default is ``savant:frames``;
- ``REDIS_TTL_SECONDS``: a TTL for Redis keys; default is ``60``;
- ``QUEUE_SIZE``: a maximum amount of messages in the queue; default is ``50``;
- ``DEDUPLICATE``: when ``True`` and the frame content was not encoded by the module (i.e. the module works in pass-through mode) the adapter will only update TTL of the frame content in Redis; default is ``False``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-kafka-redis-test \
        --entrypoint python \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e KAFKA_BROKERS=kafka:9092 \
        -e KAFKA_TOPIC=kafka-redis-adapter-demo \
        -e KAFKA_CREATE_TOPIC=True \
        -e KAFKA_CREATE_TOPIC_NUM_PARTITIONS=4 \
        -e KAFKA_CREATE_TOPIC_REPLICATION_FACTOR=1 \
        -e 'KAFKA_CREATE_TOPIC_CONFIG={"retention.ms": 300000}' \
        -e REDIS_HOST=redis \
        -e REDIS_PORT=6379 \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-py:latest \
        -m adapters.python.sinks.kafka_redis

.. note::
    The adapter doesn't have ``ZMQ_TYPE``, ``ZMQ_BIND`` parameters.

Running the adapter with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py kafka-redis --brokers=kafka:9092 --topic=kafka-redis-adapter-demo --redis-host=redis

Multistream Kinesis Video Stream Sink Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Multistream Kinesis Video Stream Sink Adapter sends video frames to Kinesis Video Streams as video fragments.

.. note::

    The adapter requires frames to have absolute timestamps. This can be achieved by setting ``USE_ABSOLUTE_TIMESTAMPS=True`` on a source adapter.

.. list-table:: Parameters
    :header-rows: 1

    * - Parameter
      - Description
      - Default
      - Example
    * - ``AWS_REGION``
      - An AWS region.
      - Unset
      - ``us-west-2``
    * - ``AWS_ACCESS_KEY``
      - An AWS access key ID.
      - Unset
      - ``AKIAIOSFODNN7EXAMPLE``
    * - ``AWS_SECRET_KEY``
      - An AWS secret access key.
      - Unset
      - ``wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY``
    * - ``STREAM_NAME_PREFIX``
      - A prefix for the stream name. The stream name is generated as ``<STREAM_NAME_PREFIX><source_id>``.
      - Empty string
      - ``foo-``
    * - ``ALLOW_CREATE_STREAM``
      - A flag indicating whether to create a stream on KVS if it does not exist.
      - ``False``
      - ``True``
    * - ``KVSSDK_LOGLEVEL``
      - A log level for the KVS SDK.
      - ``INFO``
      - ``DEBUG``
    * - ``SOURCE_ID``
      - An optional filter to filter out frames with a specific ``source_id`` only.
      - Unset
      - ``rtsp-cam-street``
    * - ``SOURCE_ID_PREFIX``
      - An optional filter to filter out frames with a matching ``source_id`` prefix only.
      - Unset
      - ``usb-cam-``
    * - ``BUFFER_LOW_THRESHOLD``
      - A threshold in seconds when the KVS SDK slows down accepting new frames.
      - ``30``
      - ``25``
    * - ``BUFFER_HIGH_THRESHOLD``
      - A threshold in seconds when the KVS SDK stops accepting new frames.
      - ``40``
      - ``45``
    * - ``FPS_PERIOD_FRAMES``
      - A number of frames between FPS reports.
      - ``1000``
      - ``500``
    * - ``FPS_PERIOD_SECONDS``
      - A number of seconds between FPS reports.
      - Unset
      - ``10``

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name sink-kvs-test \
        --entrypoint /opt/savant/adapters/gst/sinks/multistream_kvs.py \
        -e ZMQ_ENDPOINT=sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc \
        -e AWS_REGION=us-west-2 \
        -e AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE \
        -e AWS_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
        -e STREAM_NAME_PREFIX=foo- \
        -e ALLOW_CREATE_STREAM=True \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

Running with the helper script:

.. code-block:: bash

    ./scripts/run_sink.py multistream-kvs \
        --aws-region=us-west-2 \
        --aws-access-key='AKIAIOSFODNN7EXAMPLE' \
        --aws-secret-key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'

.. note::

    The adapter doesn't have ``ZMQ_TYPE``, ``ZMQ_BIND`` parameters.

.. note::

    The adapter supports only ``h265`` and ``hevc`` codecs.


Bridge Adapters
---------------

Bridge adapters deliver data from other elements of the pipeline (e.g. a source adapter) to a module. Bridge adapters accept data from one ZeroMQ socket and send it to another ZeroMQ socket.

Currently, the following bridge adapters are available:

- Buffer adapter.

Bridge adapters accept the following common parameters:

- ``ZMQ_SINK_ENDPOINT``: a ZeroMQ socket for data input matching the one specified in module's output;  the endpoint coding scheme is ``<socket_type>+(bind|connect):<endpoint>``;
- ``ZMQ_SRC_ENDPOINT``: adapter's socket where it sends media stream; it must form a valid ZeroMQ pair with module's input socket; the endpoint coding scheme is ``<socket_type>+(bind|connect):<endpoint>``;

Buffer Bridge Adapter
^^^^^^^^^^^^^^^^^^^^^

The Buffer Bridge Adapter buffers messages from a source and sends them to a module. When the module is not able to accept the message, the adapter buffers it until the module is ready to accept it. When the buffer is full, the adapter drops the incoming message.

.. note::

    The endpoint in ``ZMQ_SRC_ENDPOINT`` can only be ``DEALER``.

**Parameters**:

- ``BUFFER_PATH`` (**required**): a path to a buffer;
- ``BUFFER_LEN``: a maximum amount of messages in the buffer; default is ``1000``;
- ``BUFFER_SERVICE_MESSAGES``: a buffer length for service messages (eg. EndOfStream, Shutdown); used when the main part of the buffer is full (``BUFFER_LEN``); default is ``100``;
- ``BUFFER_THRESHOLD_PERCENTAGE``: a threshold to mark the buffer not full; default is ``80``;
- ``IDLE_POLLING_PERIOD``: an interval between polling messages from the buffer when the buffer is empty, in seconds; default is ``0.005``;
- ``STATS_LOG_INTERVAL``: an interval between logging buffer statistics, in seconds; default is ``60``;
- ``METRICS_FRAME_PERIOD``: output FPS stats after every N frames; default is ``1000``;
- ``METRICS_TIME_PERIOD``: output FPS stats after every N seconds;
- ``METRICS_HISTORY``: how many last FPS stats to keep in the memory; default is ``100``;
- ``METRICS_PROVIDER``: a metrics provider name; only ``prometheus`` is supported;
- ``METRICS_PROVIDER_PARAMS``: a json dict of metrics provider parameters; default is ``{}``. The ``port`` in ``METRICS_PROVIDER_PARAMS`` is required when ``METRICS_PROVIDER`` is set to ``'prometheus'``. ``labels`` in ``METRICS_PROVIDER_PARAMS`` defines extra labels added to the metrics;
- ``MESSAGE_DUMP_ENABLED``: a flag indicating whether to dump messages to a file; default is ``False``;
- ``MESSAGE_DUMP_PATH``: a directory to dump message segment files; default is ``/tmp/buffer-adapter-dump``;
- ``MESSAGE_DUMP_SEGMENT_DURATION``: a duration of a message segment in seconds; default is ``60``.
- ``MESSAGE_DUMP_SEGMENT_TEMPLATE``: a template for message segment file names; default is ``dump-%Y-%m-%d-%H-%M-%S.msgpack``.

Running the adapter with Docker:

.. code-block:: bash

    docker run --rm -it --name bridge-buffer-test \
        --entrypoint python \
        -e ZMQ_SINK_ENDPOINT=sub+bind:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e ZMQ_SRC_ENDPOINT=dealer+bind:ipc:///tmp/zmq-sockets/buffered-video.ipc \
        -e BUFFER_PATH=/tmp/savant/buffer \
        -e BUFFER_LEN=1000 \
        -e BUFFER_SERVICE_MESSAGES=100 \
        -e BUFFER_THRESHOLD_PERCENTAGE=80 \
        -e IDLE_POLLING_PERIOD=0.005 \
        -e STATS_LOG_INTERVAL=60 \
        -e METRICS_FRAME_PERIOD=1000 \
        -e METRICS_TIME_PERIOD=10 \
        -e METRICS_HISTORY=100 \
        -e METRICS_PROVIDER=prometheus \
        -e METRICS_PROVIDER_PARAMS='{"port": 8000, "labels":{"adapter":"buffer"}}' \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        -v /tmp/savant/buffer:/tmp/savant/buffer \
        ghcr.io/insight-platform/savant-adapters-py:latest \
        -m adapters.python.bridges.buffer

Running the adapter with the helper script:

.. code-block:: bash

    ./scripts/run_bridge.py buffer --mount-buffer-path /tmp/savant/buffer

The adapter exports its internal state to Prometheus:

.. list-table:: Metrics
    :header-rows: 1

    * - Metric
      - Description
      - Type

    * - ``received_messages``
      - A total number of received messages.
      - Counter

    * - ``pushed_messages``
      - A total number of messages pushed in the disk buffer.
      - Counter

    * - ``dropped_messages``
      - A total number of dropped messages (exceeded the buffer limit).
      - Counter

    * - ``sent_messages``
      - A total number of downstream-sent messages.
      - Counter

    * - ``buffer_size``
      - A current buffer size in elements.
      - Gauge

    * - ``payload_size``
      - A current buffer size in bytes.
      - Gauge

    * - ``last_received_message``
      - A timestamp of the last received message.
      - Gauge

    * - ``last_pushed_message``
      - A timestamp of the last pushed message.
      - Gauge

    * - ``last_dropped_message``
      - A timestamp of the last dropped message.
      - Gauge

    * - ``last_sent_message``
      - A timestamp of the last sent message.
      - Gauge
