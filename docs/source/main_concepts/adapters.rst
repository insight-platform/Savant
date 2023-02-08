Adapters overview
=================

We call an adapter an independent process that either reads (source adapter) or writes (sink adapter) data from/to some location,
thus decoupling input/output operations from the main processing.

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

- Module binds the source endpoint to a pull socket.
- Source adapters are assumed to connect push sockets to the source endpoint.
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

pictures
^^^^^^^^

Pictures source adapter reads ``image/jpeg`` or ``image/png`` files from ``LOCATION``, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Additional pictures source adapter parameters:

- ``FRAMERATE`` is the desired framerate for the video stream formed from the input image files.
- ``SORT_BY_TIME`` is a flag that indicates whether files from ``LOCATION`` should be sorted by modification time (ascending order).
  By default it is **False**  and files are sorted lexicographically.

videos
^^^^^^

Videos source adapter reads ``video/*`` files from ``LOCATION``, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Additional videos source adapter parameters:

- ``SORT_BY_TIME`` is a flag that indicates whether files from ``LOCATION`` should be sorted by modification time (ascending order).
  By default it is **False**  and files are sorted lexicographically.

.. note::

  Resulting video stream framerate is fixed to be equal to the framerate of the first encountered video file,
  possibly overriding the framerate of the rest of input.

rtsp
^^^^

RTSP source adapter reads RTSP stream from ``RTSP_URI``.

usb-cam
^^^^^^^

Usb-cam source adapter capture video from a v4l2 device specified in ``DEVICE`` parameter.

Additional usb-cam source adapter parameters:

- ``FRAMERATE`` is the desired framerate for the video stream formed from the captured video.
  Note that if input video framerate is not in accordance with ``FRAMERATE`` parameter value, results may be unexpected.

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
  - ``%src_filename`` will insert source filename into resulting filename

Additional meta-json sink adapter parameters:

- ``SKIP_FRAMES_WITHOUT_OBJECTS`` is a flag that indicates whether frames with 0 objects should be skipped in output
- ``CHUNK_SIZE`` specifies the number of frames that the sink will put in a single chunk. Value of 0 disables chunking.

image-files
^^^^^^^^^^^

Image-files sink adapter writes received messages as separate image files into directory, specified in ``DIR_LOCATION`` parameter.

Additional image-files sink adapter parameters:

- ``SKIP_FRAMES_WITHOUT_OBJECTS`` is a flag that indicates whether frames with 0 objects should be skipped in output
- ``CHUNK_SIZE`` specifies the number of frames that the sink will put in a single chunk. Value of 0 disables chunking.

video-files
^^^^^^^^^^^

Video-files sink adapter writes received messages as video files into directory, specified in ``DIR_LOCATION`` parameter.

Additional video-files sink adapter parameters:

- ``CHUNK_SIZE`` specifies the number of frames that the sink will put in a single chunk. Value of 0 disables chunking.

display
^^^^^^^

Display sink adapter sends received frames to a window output.

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
