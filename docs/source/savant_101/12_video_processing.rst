Video Processing Workflow
=========================

In Savant every frame passes certain processing stages which you have to understand. These stages are inspired by DeepStream's internals and there is no simple way to hack them in a different way.

Those stages are (see the image below):

- decoding (3);
- scaling to a common resolution (**mandatory**) (4);
- adding commonly-specified paddings (`optional`) (4);
- multiplexing (4-5);
- processing (6);
- drawing (`optional`) (6);
- de-multiplexing (7, 8);
- encoding (9).

.. image:: ../_static/img/0_streaming_model_detailed.png

Discuss them in details.

Decoding
--------

Nvidia supports very fast hardware-accelerated decoding for several video/image codecs. The dedicated NVDEC device performs the task at speed higher than 1000 FPS for FullHD. You must consider using Savant with those codecs which have hardware support. We currently support several source formats for the frames. The framework automatically understands the formats; you may simultaneously feed the module with streams in different codecs.

**Raw RGBA** is a slow representation as it requires extensive data transfers over PCI-E and network, leading to decreased performance.

Hardware-accelerated with NVDEC, preferred to be used:

- ``H264``: default;
- ``HEVC/H265``: preferred (performance, bandwidth);
- ``MJPEG``, ``JPEG``: when image streams or USB/CSI-cams are used.

Software-decoded (not recommended to use):

- ``PNG``: fallback for compatibility purposes.

Scaling to a Common Resolution
------------------------------

**The pipeline will scale all streams to the configured resolution.**

This is a crucial topic. The pipeline is always configured to run on common resolution. It means that every stream handled by a certain pipeline instance is always scaled to the common resolution configured for the pipeline instance, no matter what its input resolution was.

If you need different streams to be handled on different resolutions, you have to launch several pipelines configuring each pipeline to use a resolution acceptable for streams processed by that pipeline.

.. code-block:: yaml

    # base module parameters
    parameters:
      # pipeline processing frame parameters
      frame:
        width: 1280
        height: 720

Let us consider the following examples:

Case 1: No Output Footage
^^^^^^^^^^^^^^^^^^^^^^^^^

You have 10 cams of FullHD and 15 cams of HD resolution. You don't need the outgoing video at all, all your models are fine to work with HD resolution.

**Solution**: configure the pipeline to use HD resolution, send all streams to a single pipeline.

Case 2: Low-Res Output Footage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have 10 cams of FullHD and 15 cams of HD resolution. You need the outgoing video and HD resolition is acceptable, all your models are fine to work with HD resolution.

**Solution**: configure the pipeline to use HD resolution, send all streams to a single pipeline.

Case 3: Hi-Res Output Footage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have 10 cams of FullHD and 15 cams of HD resolution. You need the outgoing video in the same resolution as incoming.

**Solution**: configure two pipelines - the first to use Full-HD resolution, the second to use HD resolution. Point Full-HD cams to the Full-HD pipeline, HD cams to the HD pipeline.

Paddings
--------

Adding paddings is useful if you need spare space for utility purposes. E.g. you may use paddings to preprocess the image before passing it to the model. Another way to use paddings is to display utility content.

The paddings can either be preserved or removed at the output.

.. code-block:: yaml

    # base module parameters
    parameters:
      # pipeline processing frame parameters
      frame:
        width: 1280
        height: 720
        # Add paddings to the frame before processing
        padding:
          # Paddings are kept on the output frame
          keep: true
          left: 0
          right: 1280
          top: 0
          bottom: 0

.. note::

    If you specify ``parameters.frame.padding.keep == false``, the paddings are removed before frame encoding. The geometry for all objects are recalculated to conform new geometry.

Geometry Base
-------------

The ``geometry_base`` parameter specifies the value by which any geometry dimension of the frame (width, height, margin size) must be evenly divided. The default value is ``8``.

When the developer specifies the frame dimensions do not fit the ``geometry_base``, the pipeline will stop with an error. Thus, when defining ``frame.width``, ``frame.height``, and ``frame.padding.*`` every of them must be divisible by ``geometry_base``. The parameter is introduced to overcome unexpected behavior due to platform-specific hardware limitations when a non-standard resolution is used during image processing and encoding.

.. tip:: We do not recommend setting ``geometry_base`` parameter to the values other than ``8`` or ``4``.

.. code-block:: yaml

    # base module parameters
    parameters:
      # pipeline processing frame parameters
      frame:
        width: 1280
        height: 720
        # Base value for frame parameters. All frame parameters must be divisible by this value.
        # Default is 8.
        geometry_base: 8

Multiplexing
------------

All streams processed by a single module instance are grouped into batches before processing. Batch is a concept used to optimize the computations on Nvidia hardware. Savant is implemented to hide batching: developers typically work with a single frame, not a batch of frames.

.. code-block:: yaml

    # base module parameters
    parameters:
      ...
      batch_size: 1

Typically you may set ``batch_size`` equal to the maximum expected number of simultaneously processed streams. Find out more on :doc:`/advanced_topics/0_batching` in the advanced topics.

Processing
----------

At this step, the pipeline is implemented: model inference, tracking, and python functions work here. We will discuss the processing in detail in further sections. You cannot modify the frame resolution at this step. You can modify the frame content.

Drawing
-------

Certain objects during the processing step can receive labels specifying that they must be drawn on the frame (e.g. identifiers, class names, boxes, etc). Drawing is an optional step which can be performed.

.. code-block:: yaml

    # base module parameters
    parameters:
      ...
      draw_func: {}

The draw function may be overriden by the developer if the stock version cannot draw the information required:

.. code-block:: yaml

    parameters:
      ...
      draw_func:
        module: samples.peoplenet_detector.overlay
        class_name: Overlay
        kwargs:
          person_with_face_bbox_color: [0, 1, 0]
          person_no_face_bbox_color: [1, 0, 0]
          person_label_bg_color: [1, 0.9, 0.85]
          person_label_font_color: [0, 0, 0]
          bbox_border_width: 3
          overlay_height: 180
          logo_height: 120
          sprite_height: 120
          counters_height: 85
          counters_font_thickness: 5

.. note::

    To disable ``draw_func`` functionality, remove ``parameters.draw_func`` from the manifest completely.

Conditional Drawing
^^^^^^^^^^^^^^^^^^^

Savant supports a conditional drawing feature. It enables defining a special condition based on a frame tag which enables drawing. The motivation behind the feature is efficiency: often, you don't need to produce footage for all streams but only for certain streams under investigation. So you may implement a pyfunc which creates a tag for those streams.

To configure conditional drawing, add a subsection to ``draw_func`` as follows:

.. code-block:: yaml

    draw_func:
      condition:
        tag: <tagname, e.g. draw>


An example of conditional drawing can be found in a dedicated Savant `sample <https://github.com/insight-platform/Savant/tree/develop/samples/conditional_video_processing>`__.


De-Multiplexing
---------------

This step is automatically performed by the framework to turn batches into individual streams before passing the frames to stream encoders.

Encoding
--------

If the ``output_frame`` section is omitted, video frames will not be sent to sinks at all.

The framework supports several encoding schemes:

- RAW RGBA (not optimal, as it requires large transfers over PCI-E);
- JPEG (hardware ``nvjpegenc``, software ``jpegenc``);
- PNG (software ``pngenc``);
- H264 (hardware ``nvv4l2h264enc``, software ``x264enc``);
- HEVC/H265 (hardware ``nvv4l2h265enc``).

.. note::

    Hardware encoder for JPEG is available only on Nvidia Jetson. On dGPU JPEG encoder is CUDA-assisted when supported by the hardware.

We highly advise using hardware assisted codecs. The only caveat is to steer clear from GeForce GPUs in production as they have a limitation constraining simultaneous encoding to 3 streams. In case you are using GeForce, choose RAW RGBA.

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264

You can choose hardware or software encoder by setting ``encoder`` parameter to ``nvenc`` or ``software`` respectively:

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder: nvenc

When ``encoder`` parameter is specified and the framework doesn't find a suitable encoder, it will end with an error. When ``encoder`` parameter is omitted, the framework will try to use hardware encoder. When it fails, it will fall back to software encoder.

Every codec has its own configuration parameters related to a corresponding GStreamer plugin. Those parameters are defined in ``output_frame.encoder_params``:

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000
          iframeinterval: 10
          profile: High

Encoder Properties
------------------

Hardware H264 Encoder (NVENC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``bitrate``

   Sets the bitrate for the v4l2 encoder. Allowed range: ``0`` - ``4294967295``. The Default value is ``4000000``.

2. ``control-rate``

   Sets the control rate for the v4l2 encoder. The default value is ``1``.

   Options are:

   - ``0`` or ``variable_bitrate``;
   - ``1`` or ``constant_bitrate``;


3. ``extended-colorformat``

   Sets Extended ColorFormat pixel values ``0`` to ``255`` in VUI info. The default value is ``false``.

4. ``force-idr``

   Forces an IDR frame. The default value is ``false``.

5. ``force-intra``

   Forces an INTRA frame. The default value is ``false``.

6. ``iframeinterval``

   Encoding Intra Frame occurrence frequency. Range: ``0`` - ``4294967295``. The default value is ``30``.

7. ``preset-id``

   Sets CUVID Preset ID for the encoder. Range: ``1`` - ``7``. The default value is ``1``.

8. ``profile``

   Sets the profile for the v4l2 encoder. The default value is ``0`` (``Baseline``).

   Options are:

   - ``0``: ``Baseline``
   - ``2``: ``Main``
   - ``4``: ``High``
   - ``7``: ``High444``

9. ``tuning-info-id``

   Tuning Info Preset for the encoder. The default value is ``2``.

   Options are:

   - ``1``: ``HighQualityPreset``
   - ``2``: ``LowLatencyPreset``
   - ``3``: ``UltraLowLatencyPreset``
   - ``4``: ``LosslessPreset``


Software H264 Encoder
^^^^^^^^^^^^^^^^^^^^^

1. ``bitrate``

   Bitrate in kbit/sec. Range: ``1`` - ``2048000``. The default value is ``2048``.

2. ``key-int-max``

   Maximum distance between two key-frames (``0`` for automatic). Range: ``0`` - ``2147483647``. The default value is ``0``.

3. ``pass``

   Encoding pass/type. The default value is ``0`` (``cbr``)

   Options are:

   - ``0`` or ``cbr``: Constant Bitrate Encoding
   - ``4`` or ``quant``: Constant Quantizer
   - ``5`` or ``qual``: Constant Quality
   - ``17`` or ``pass1``: VBR Encoding - Pass 1
   - ``18`` or ``pass2``: VBR Encoding - Pass 2
   - ``19`` or ``pass3``: VBR Encoding - Pass 3

4. `speed-preset`

   Preset name for speed/quality tradeoff options (can affect decode compatibility - impose restrictions separately for your target decoder). The default value is ``6`` (or ``medium``).

   Options:

   - ``1`` or ``ultrafast``;
   - ``2`` or ``superfast``;
   - ``3`` or ``veryfast``;
   - ``4`` or ``faster``;
   - ``5`` or ``fast``;
   - ``6`` or ``medium``;
   - ``7`` or ``slow``;
   - ``8`` or ``slower``;
   - ``9`` or ``veryslow``;
   - ``10`` or ``placebo``;

5. `tune`

   Preset name for non-psychovisual tuning options. The default value is ``0x00000000`` or ``none``.

   Options:

   - ``0x00000000`` or ``none``
   - ``0x00000001`` or ``stillimage``: Still image
   - ``0x00000002`` or ``fastdecode``: Fast decode
   - ``0x00000004`` or ``zerolatency``: Zero latency

Hardware HEVC Codec (NVENC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``bitrate``

   Sets the bitrate for the v4l2 encoder. Range: ``0`` - ``4294967295``. The default value is ``4000000``.

2. ``control-rate``

   Sets the control rate for the v4l2 encoder. The default value is ``1`` or ``constant_bitrate``.

   Options are:

   - ``0`` or ``variable_bitrate``;
   - ``1`` or ``constant_bitrate``;

3. ``extended-colorformat``

   Sets extended color format pixel values ``0`` to ``255`` in VUI info. The default value is ``false``.

4. ``force-idr``

   Forces an IDR frame. The default value is ``false``.

5. ``force-intra``

   Forces an INTRA frame. The default value is ``false``.

6. ``iframeinterval``

   Encoding Intra Frame occurrence frequency. Range: ``0`` - ``4294967295``. The default value is ``30``.

7. ``preset-id``

   Sets CUVID Preset ID for Encoder. Range: ``1`` - ``7``. The default value is ``1``.

8. ``profile``

   Sets the profile for the v4l2 encoder. The default value is ``0`` or ``Main``.

   Options are:

   - ``0`` or ``Main``
   - ``1 `` or  ``Main10``

9. ``tuning-info-id``

   Tuning Info Preset for the encoder. The default value is ``2`` or ``LowLatencyPreset``.

   Options are:

   - ``1`` or ``HighQualityPreset``
   - ``2`` or ``LowLatencyPreset``
   - ``3`` or ``UltraLowLatencyPreset``
   - ``4`` or ``LosslessPreset``

JPEG Codec
^^^^^^^^^^

1. ``idct-method``

   The IDCT algorithm to use. The default value is ``1`` or ``ifast``.

   Options are:

   - ``0`` or ``islow``: slow but accurate integer algorithm
   - ``1`` or ``ifast``: faster, less accurate integer method
   - ``2`` or ``float``: floating-point, accurate, fast on fast HW

2. ``quality``

   Quality of encoding. Range: ``0`` - ``100``. The default value is ``85``.

PNG Сodec
^^^^^^^^^

1. ``compression-level``

   PNG compression level. Range: ``0`` - ``9``. The default value is ``6``.

Example:

  .. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000
          profile: 4

  .. code-block:: yaml

    parameters:
      output_frame:
        codec: jpeg
        encoder_params:
          quality: 90

To list all available properties run ``gst-inspect-1.0 <encoder-name>``. E.g. ``gst-inspect-1.0 nvv4l2h264enc``.

Conditional Encoding
--------------------

Savant 0.2.4 introduced a conditional encoding feature. It enables defining a special condition based on a frame tag, enabling encoding only certain streams. The motivation behind the feature is efficiency: often, you don't need to produce a resulting video for all streams but only for certain streams under investigation. So you may implement a pyfunc which creates a tag for those streams.

To configure conditional encoding, add a subsection to ``output_frame`` as follows:

.. code-block:: yaml

    output_frame:
      codec: h264
      encoder_params:
        iframeinterval: 25
      condition:
        tag: <tagname, e.g. encode>

An example of conditional drawing can be found in a dedicated Savant `sample <https://github.com/insight-platform/Savant/tree/develop/samples/conditional_video_processing>`__.
