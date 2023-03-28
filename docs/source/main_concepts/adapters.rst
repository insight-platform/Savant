Adapters overview
=================

We call an adapter an independent process that either reads (source adapter) or writes (sink adapter) data from/to some location,
thus decoupling input/output operations from the main processing. All adapters are implemented as Docker images, and
Python scripts have been developed to simplify the process of running and using source and sink adapters.
The following section outlines the different parameters for each adapter and provides examples of how
to run them using the Python scripts and there are also direct docker run examples.
A savant module is able to interface with any number of both source and sink adapters at the same time by using ZeroMQ sockets.

.. note::

  In practice the number of parallel sources of a Savant module is capped. The limit is determined by the
  ``parameters.max_parallel_streams`` module config value and is 64 by default.

.. _src_adapter_source_id:

.. note::

    Successful demultiplexing of video streams from different sources inside the module relies on uniqueness of ``SOURCE_ID``
    parameter of each source.

    ``SOURCE_ID`` is an arbitrary string, the only requirement for ``SOURCE_ID`` values is that they should map one-to-one
    to module's source adapters.

    Savant source adapter requires ``SOURCE_ID`` to be correctly set by the user either through container environment variable
    or through CLI if using adapter start helper scripts.

    When writing custom source adapters user should take care to send messages with correct ``SOURCE_ID`` values in the messages
    (see :ref:`reference/avro:VideoFrame Schema`).

Default ZeroMQ source and sink configuration in module config

.. literalinclude:: ../../../savant/config/default.yml
  :language: YAML
  :lines: 74-

defines the following **default connection pattern**:

.. _default_connection_pattern:

- Module binds the source endpoint to a router socket.
- Source adapters are assumed to connect dealer sockets to the source endpoint.
- Module binds the sink endpoint to a publisher socket.
- Sink adapters are assumed to connect subscriber sockets to the sink endpoint.

:repo-link:`Savant` already includes several containerized :ref:`adapter_sources` and :ref:`adapter_sinks`
adapters that are ready-to-use with any Savant module.

Each Savant adapter has a number of parameters which can be set through environment variables.
Learn more about general ZMQ parameters that are required for running a Savant module in combination
with Savant adapters in :doc:`../getting_started/running` or
read below for specific Savant adapters descriptions and parameters.

.. _adapter_sources:

Sources
-------

These adapters should help the user with the most basic and widespread input data formats.

image
^^^^^

Image source adapter reads ``image/jpeg`` or ``image/png`` files from ``LOCATION``, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Image source adapter parameters. These parameters are set as environment variables in the docker run command:

- ``SOURCE_ID`` set unique identifier for the source adapter. This option is required.
- ``FRAMERATE`` is the desired framerate for the video stream formed from the input image files.
- ``SORT_BY_TIME`` is a flag that indicates whether files from ``LOCATION`` should be sorted by modification time (ascending order).
  By default it is **False**  and files are sorted lexicographically.
- ``READ_METADATA`` is a flag that indicates attempt to read the metadata of objects from the JSON file
  that has the identical name as the source file with json extension, and then send it to the module with frame.
  Default is **False**.
- ``OUT_ENDPOINT`` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- ``OUT_TYPE`` set adapter output ZeroMQ socket type. Default is DEALER.
- ``OUT_BIND`` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- ``SYNC`` send frames from source synchronously (i.e. with the frame rate set via the FRAMERATE parameter). Default is False.
- ``FPS_PERIOD_FRAMES`` set number of frames between FPS reports. Default is 1000.
- ``FPS_PERIOD_SECONDS`` set number of seconds between FPS reports. Default is None.
- ``FPS_OUTPUT`` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example

.. code-block:: bash

    docker run --rm -it --name source-pictures-files-test \
    --entrypoint /opt/app/adapters/gst/sources/media_files.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e LOCATION=/path/to/images \
    -e FILE_TYPE=picture \
    -e SORT_BY_TIME=False \
    -e READ_METADATA=False \
    -v /path/to/images:/path/to/images:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_source.py pictures --source-id test /path/to/images


videos
^^^^^^

Video source adapter reads ``video/*`` files from ``LOCATION``, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Video source adapter parameters. These parameters are set as environment variables in the docker run command:

- ``SOURCE_ID`` set unique identifier for the source adapter. This option is required.
- ``SORT_BY_TIME`` is a flag that indicates whether files from ``LOCATION`` should be sorted by modification time (ascending order).
  By default it is **False**  and files are sorted lexicographically.
- ``READ_METADATA`` is a flag that indicates attempt to read the metadata of objects from the JSON file
  that has the identical name as the source file with json extension, and then send it to the module with frame.
  Default is **False**.
- ``OUT_ENDPOINT`` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- ``OUT_TYPE`` set adapter output ZeroMQ socket type. Default is DEALER.
- ``OUT_BIND`` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- ``SYNC`` set send frames from source synchronously (i.e. at the source file rate). Default is False.
- ``FPS_PERIOD_FRAMES`` set number of frames between FPS reports. Default is 1000.
- ``FPS_PERIOD_SECONDS`` set number of seconds between FPS reports. Default is None.
- ``FPS_OUTPUT`` set path to the file where the FPS reports will be written. Default is ''.

Example

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/media_files.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e LOCATION=/path/to/data/test.mp4 \
    -e FILE_TYPE=video \
    -e SORT_BY_TIME=False \
    -e READ_METADATA=False \
    -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_source.py videos --source-id test /path/to/data/test.mp4

.. note::

  Resulting video stream framerate is fixed to be equal to the framerate of the first encountered video file,
  possibly overriding the framerate of the rest of input.

rtsp
^^^^

RTSP source adapter reads RTSP stream from rtsp uri.

RTSP source adapter parameters. These parameters are set as environment variables in the docker run command:

- ``RTSP_URI`` specifies the RTSP URI for rtsp_source adapter. This option is required.
- ``SOURCE_ID`` set unique identifier for the source adapter. This option is required.
- ``OUT_ENDPOINT`` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- ``OUT_TYPE`` set Adapter output ZeroMQ socket type. Default is DEALER.
- ``OUT_BIND`` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- ``FPS_PERIOD_FRAMES`` set number of frames between FPS reports. Default is 1000.
- ``FPS_PERIOD_SECONDS`` set number of seconds between FPS reports. Default is None.
- ``FPS_OUTPUT`` set path to the file where the FPS reports will be written. Default is ''.

Example

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/rtsp.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e RTSP_URI=rtsp://192.168.1.1 \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_source.py rtsp --source-id test rtsp://192.168.1.1

usb-cam
^^^^^^^

Usb-cam source adapter capture video from a v4l2 device specified in ``DEVICE`` parameter.

Usb-cam source adapter parameters. These parameters are set as environment variables in the docker run command:

- ``DEVICE`` is the device file of the USB camera (default value is /dev/video0).
- ``FRAMERATE`` is the desired framerate for the video stream formed from the captured video.
  Note that if input video framerate is not in accordance with ``FRAMERATE`` parameter value, results may be unexpected.
- ``SOURCE_ID`` set unique identifier for the source adapter. This option is required.
- ``OUT_ENDPOINT`` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- ``OUT_TYPE`` set Adapter output ZeroMQ socket type. Default is DEALER.
- ``OUT_BIND`` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- ``FPS_PERIOD_FRAMES`` set number of frames between FPS reports. Default is 1000.
- ``FPS_PERIOD_SECONDS`` set number of seconds between FPS reports. Default is None.
- ``FPS_OUTPUT`` set path to the file where the FPS reports will be written. Default is ''.

Example

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/rtsp.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e DEVICE=/dev/video1 \
    -e FRAMERATE=30/1 \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_source.py usb-cam --source-id test --framerate 30/1 /dev/video1

gige
^^^^

It is designed to take video streams from GigE cameras and send them into a module for processing.

Usb-cam source adapter parameters. These parameters are set as environment variables in the docker run command:

- ``WIDTH`` the width of the video stream, in pixels
- ``HEIGHT`` the height of the video stream, in pixels
- ``FRAMERATE`` the framerate of the video stream, in frames per second
- ``INPUT_CAPS`` the format of the video stream, in GStreamer caps format (e.g. "video/x-raw,format=RGB")
- ``PACKET_SIZE`` the packet size for GigEVision cameras, in bytes
- ``AUTO_PACKET_SIZE`` whether to negotiate the packet size automatically for GigEVision cameras
- ``EXPOSURE`` the exposure time for the camera, in microseconds
- ``EXPOSURE_AUTO`` the auto exposure mode for the camera, one of "off", "once", or "on"
- ``GAIN`` the gain for the camera, in decibels
- ``GAIN_AUTO`` the auto gain mode for the camera, one of "off", "once", or "on"
- ``FEATURES`` additional configuration parameters for the camera, as a space-separated list of feature assignations
- ``HOST_NETWORK`` whether to use the host network
- ``CAMERA_NAME`` the name of the camera, in the format specified in the command description. This parameter is optional, and if it is
- ``SOURCE_ID`` set unique identifier for the source adapter. This option is required.
- ``OUT_ENDPOINT`` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- ``OUT_TYPE`` set Adapter output ZeroMQ socket type. Default is DEALER.
- ``OUT_BIND`` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- ``FPS_PERIOD_FRAMES`` set number of frames between FPS reports. Default is 1000.
- ``FPS_PERIOD_SECONDS`` set number of seconds between FPS reports. Default is None.
- ``FPS_OUTPUT`` set path to the file where the FPS reports will be written. Default is ''.

Example

.. code-block:: bash

    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/gige_cam.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e CAMERA_NAME=test-camera\
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_source.py gige --source-id test test-camera


.. _adapter_sinks:

Sinks
-----

These adapters should help the user with the most basic and widespread output data formats.

meta-json
^^^^^^^^^

Meta-json sink adapter writes received messages as newline-delimited JSON streaming file to ``LOCATION``, which can be:

- Local path to a single file
- Local path with substitution patterns:
  - ``%source_id`` will insert ``SOURCE_ID`` value into resulting filename
  - ``%src_filename`` will insert source filename into resulting filename.

Meta-json sink adapter parameters. These parameters are set as environment variables in the docker run command:

- ``LOCATION`` the location of the file to write metadata to. Can be a plain location or a pattern.
    Allowed substitution parameters are %source_id and %src_filename.
- ``CHUNK_SIZE`` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate files with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new file starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- ``SKIP_FRAMES_WITHOUT_OBJECTS`` is a flag that indicates whether frames with 0 objects should be skipped in output. Default value is False.
- ``SOURCE_ID`` Optional filter to receive frames with a specific source ID only.
- ``SOURCE_ID_PREFIX`` Optional filter to receive frames with a source ID prefix only.
- ``IN-ENDPOINT`` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- ``IN_TYPE`` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- ``IN_BIND`` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/python/sinks/metadata_json.py \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-py:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_sink.py meta-json /path/to/output/%source_id-%src_filename

image-file
^^^^^^^^^^

Image file sink adapter writes received messages as separate image files and json files to a directory,
specified in ``DIR_LOCATION`` parameter.

Image file sink adapter parameters. These parameters are set as environment variables in the docker run command:

- ``DIR_LOCATION`` the location of the file to write metadata to. Can be a plain location or a pattern.
    Allowed substitution parameters are %source_id and %src_filename.
- ``CHUNK_SIZE`` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate folders with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new folder starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- ``SKIP_FRAMES_WITHOUT_OBJECTS`` is a flag that indicates whether frames with 0 objects should be skipped in output. Default value is False.
- ``SOURCE_ID`` Optional filter to receive frames with a specific source ID only.
- ``SOURCE_ID_PREFIX`` Optional filter to receive frames with a source ID prefix only.
- ``IN-ENDPOINT`` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- ``IN_TYPE`` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- ``IN_BIND`` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/python/sinks/image_files.py \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-py:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_sink.py image-files  /path/to/output/%source_id-%src_filename


video-file
^^^^^^^^^^^

Video file sink adapter writes received messages as video files to directory, specified in ``DIR_LOCATION`` parameter.

Video file sink adapter parameters. These parameters are set as environment variables in the docker run command:

- ``DIR_LOCATION`` the location of the file to write metadata to. Can be a plain location or a pattern.
    Allowed substitution parameters are %source_id and %src_filename.
- ``CHUNK_SIZE`` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate files with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new file starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- ``SOURCE_ID`` Optional filter to receive frames with a specific source ID only.
- ``SOURCE_ID_PREFIX`` Optional filter to receive frames with a source ID prefix only.
- ``IN-ENDPOINT`` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- ``IN_TYPE`` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- ``IN_BIND`` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example

.. code-block:: bash

    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/gst/sinks/video_files.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_sink.py video-files /path/to/output/%source_id-%src_filename

display
^^^^^^^

Display sink adapter sends received frames to a window output.

Display sink adapter parameters. These parameters are set as environment variables in the docker run command:

- ``CLOSING-DELAY`` he delay in seconds before closing the window after the video stream has finished.
- ``SYNC`` whether to show the frames on the sink synchronously with the source (i.e., at the source file rate).
- ``SOURCE_ID`` Optional filter to receive frames with a specific source ID only.
- ``SOURCE_ID_PREFIX`` Optional filter to receive frames with a source ID prefix only.
- ``IN-ENDPOINT`` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- ``IN_TYPE`` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- ``IN_BIND`` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example

.. code-block:: bash

    docker run --rm -it --name sink-display \
    --entrypoint /opt/app/adapters/ds/sinks/display.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DISPLAY \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -e CLOSING_DELAY=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/.docker.xauth:/tmp/.docker.xauth \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    --gpus=all savant-adapters-deepstream:latest


The same adapter can be run using an auxiliary python script

.. code-block:: python

    python3 scripts/run_sink.py display


always-on-rtsp
^^^^^^^^^^^^^^

Always-on RTSP sink adapter sends video stream from a specific source to an RTSP server.

Always-on RTSP sink adapter parameters. These parameters are set as environment variables in the docker run command:

- ``RTSP_URI``: The URI of the RTSP server, this parameter is required.
- ``STUB_FILE_LOCATION`` The location of the stub image file. Image file must be in JPEG format, this parameter is required.
  The stub image file is shown when there is no input stream.
- ``MAX_DELAY_MS`` Maximum delay for the last frame in milliseconds, default value is 1000.
- ``TRANSFER_MODE`` Transfer mode. One of: "scale-to-fit", "crop-to-fit", default value is "scale-to-fit".
- ``PROTOCOLS`` Allowed lower transport protocols, e.g. "tcp+udp-mcast+udp", default value is "tcp".
- ``LATENCY_MS`` Amount of ms to buffer RTSP stream, default value is 100.
- ``KEEP_ALIVE`` Send RTSP keep alive packets, disable for old incompatible server, default value is True.
- ``PROFILE`` H264 encoding profile. One of: "Baseline", "Main", "High", default value is "High".
- ``BITRATE`` H264 encoding bitrate, default value is 4000000.
- ``FRAMERATE`` Frame rate of the output stream, default value is "30/1".
- ``METADATA_OUTPUT`` Where to dump metadata (stdout or logger).
- ``SYNC`` Show frames on sink synchronously (i.e. at the source file rate). Inbound stream is not stable with this flag, try to avoid it. Default value is False.
- ``SOURCE_ID`` Optional filter to receive frames with a specific source ID only.
- ``SOURCE_ID_PREFIX`` Optional filter to receive frames with a source ID prefix only.
- ``IN-ENDPOINT`` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- ``IN_TYPE`` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- ``IN_BIND`` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example

.. code-block:: bash

    docker run --rm -it --name sink-always-on-rtsp \
    --entrypoint python \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
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
    -v /path/to/stub_file/test.jpg:/path/to/stub_file/test.jpg:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    --gpus=all \
    savant-adapters-deepstream:latest \
    -m adapters.ds.sinks.always_on_rtsp

The same adapter can be run using an auxiliary python script

.. code-block:: python

    python scripts/run_sink.py always-on-rtsp --source-id test --stub-file-location ../vlf-pipelines/data/bottle_defect_detector/test.jpg  rtsp://192.168.1.1



Custom adapters
---------------

A user is not limited by the adapters listed above, it's always possible to develop a custom one.
:repo-link:`Savant` requires the following from a custom source or sink adapter:

#. Adapters send and receive messages to/from Savant module through ZeroMQ sockets.
   See :ref:`default connection pattern <default_connection_pattern>`.
#. Messages are serialized with Apache Avro according to :ref:`reference/avro:Avro Schema Reference`.

   - See :ref:`note <src_adapter_source_id>` about ``SOURCE_ID``.

#. Custom source adapters have to ensure that:

   #. All the frames sent have the same parameters (codec, dimensions, framerate)
   #. Frames' timestamps are monotonic
   #. Message values pertaining to encoding are in accordance with actual encoded frame data
   #. Every time any of the conditions above is invalidated, EndOfStream message is sent by the source adapter
