Video Processing Workflow
=========================

In Savant every frame passes certain processing stages which you have to understand. These stages are inspired by DeepStream's internals and there is no simple way to hack them in a different way. Those stages are:

- decoding;
- scaling to a common resolution (mandatory);
- adding commonly-specified paddings (optional);
- multiplexing;
- processing;
- drawing (optional);
- de-multiplexing;
- encoding.

Let us consider them in details.

Decoding
--------

Nvidia supports very fast hardware-accelerated decoding for several video/image codecs. The dedicated NVDEC device performs the task at speed higher than 1000 FPS for FullHD. You must consider using Savant with those codecs which have hardware support. We currently support several source formats for the frames. The framework automatically understands the formats; you may simultaneously feed the module with streams in different codecs.

**Raw RGBA** is a slow representation as it requires extensive data transfers over PCI-E and network, leading to decreased performance.

Hardware-accelerated with NVDEC, preferred to be used:

- H264 - default;
- HEVC (H265) - preferred (performance, bandwidth);
- MJPEG, JPEG - when image streams or USB/CSI-cams are used.

Software-decoded (not recommended to use):

- PNG - made for compatibility purposes.

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

Adding Paddings
---------------

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

Multiplexing
------------

All streams processed by a single module instance are grouped into batches before processing. Batch is a concept used to optimize the computations on Nvidia hardware. Savant is implemented in such a way as to hide batching from the developer: you always operate with a single frame, not a batch of frames.

.. code-block:: yaml

    # base module parameters
    parameters:
      ...
      batch_size: 1

Set the batch size equal to the maximum expected number of simultaneously processed streams.

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

if ``draw_func`` section is omitted, drawing is not performed.

De-Multiplexing
---------------

This step is automatically performed by the framework to turn batches into individual streams before passing the frames to stream encoders.

Encoding
--------

If the ``output_frame`` section is omitted, video frames will not be sent to sinks at all.

The framework supports several encoding schemes:

- RAW RGBA (not optimal, as it requires large transfers over PCI-E);
- JPEG (software);
- PNG (software);
- H264 (hardware);
- HEVC (H265, hardware).

We highly advise using hardware NVENC-assisted codecs. The only caveat is to steer clear from GeForce GPUs in production as they have a limitation constraining simultaneous encoding to 3 streams. In case you are using GeForce, choose RAW RGBA.

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264

Every codec has its own configuration parameters related to a corresponding GStreamer plugin. Those parameters are defined in ``output_frame.encoder_params``:

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000
          iframeinterval: 10
          profile: High

Available properties are:

  - For **h264** codec

    1. `bitrate`

       Set bitrate for v4l2 encode. Unsigned Integer. Range: 0 - 4294967295. Default: 4000000

    2. `control-rate`

       Set control rate for v4l2 encode. Default: 1, "constant_bitrate"

       (0): variable_bitrate - GST_V4L2_VIDENC_VARIABLE_BITRATE

       (1): constant_bitrate - GST_V4L2_VIDENC_CONSTANT_BITRATE

    3. `extended-colorformat`

       Set Extended ColorFormat pixel values 0 to 255 in VUI info. Boolean. Default: false

    4. `force-idr`

       Force an IDR frame. Boolean. Default: false

    5. `force-intra`

       Force an INTRA frame. Boolean. Default: false

    6. `iframeinterval`

       Encoding Intra Frame occurance frequency. Unsigned Integer. Range: 0 - 4294967295. Default: 30

    7. `preset-id`

       Set CUVID Preset ID for Encoder. Unsigned Integer. Range: 1 - 7. Default: 1

    8. `profile`

       Set profile for v4l2 encode. Default: 0, "Baseline"

       (0): Baseline         - GST_V4L2_H264_VIDENC_BASELINE_PROFILE

       (2): Main             - GST_V4L2_H264_VIDENC_MAIN_PROFILE

       (4): High             - GST_V4L2_H264_VIDENC_HIGH_PROFILE

       (7): High444          - GST_V4L2_H264_VIDENC_HIGH_444_PREDICTIVE

    9. `tuning-info-id`

       Tuning Info Preset for encoder. Default: 2, "LowLatencyPreset"

       (1): HighQualityPreset - Tuning Preset for High Quality

       (2): LowLatencyPreset - Tuning Preset for Low Latency

       (3): UltraLowLatencyPreset - Tuning Preset for Low Latency

       (4): LosslessPreset   - Tuning Preset for Lossless

  - For **h265** codec

    1. `bitrate`

       Set bitrate for v4l2 encode. Unsigned Integer. Range: 0 - 4294967295. Default: 4000000

    2. `control-rate`

       Set control rate for v4l2 encode. Default: 1, "constant_bitrate"

       (0): variable_bitrate - GST_V4L2_VIDENC_VARIABLE_BITRATE

       (1): constant_bitrate - GST_V4L2_VIDENC_CONSTANT_BITRATE

    3. `extended-colorformat`

       Set Extended ColorFormat pixel values 0 to 255 in VUI info. Boolean. Default: false

    4. `force-idr`

       Force an IDR frame. Boolean. Default: false

    5. `force-intra`

       Force an INTRA frame. Boolean. Default: false

    6. `iframeinterval`

       Encoding Intra Frame occurance frequency. Unsigned Integer. Range: 0 - 4294967295. Default: 30

    7. `preset-id`

       Set CUVID Preset ID for Encoder. Unsigned Integer. Range: 1 - 7. Default: 1

    8. `profile`

       Set profile for v4l2 encode. Default: 0, "Main"

       (0): Main             - GST_V4L2_H265_VIDENC_MAIN_PROFILE

       (1): Main10           - GST_V4L2_H265_VIDENC_MAIN10_PROFILE

    9. `tuning-info-id`

       Tuning Info Preset for encoder. Default: 2, "LowLatencyPreset"

       (1): HighQualityPreset - Tuning Preset for High Quality

       (2): LowLatencyPreset - Tuning Preset for Low Latency

       (3): UltraLowLatencyPreset - Tuning Preset for Low Latency

       (4): LosslessPreset   - Tuning Preset for Lossless

  - For **jpeg** codec

    1. `idct-method`

       The IDCT algorithm to use. Default: 1, "ifast"

       (0): islow - Slow but accurate integer algorithm

       (1): ifast - Faster, less accurate integer method

       (2): float - Floating-point: accurate, fast on fast HW

    2. `quality`

       Quality of encoding. Integer. Range: 0 - 100. Default: 85

  - For **png** codec

    1. `compression-level`

       PNG compression level. Unsigned Integer. Range: 0 - 9. Default: 6

Example:

  .. code-block:: YAML

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000
          profile: 4

  .. code-block:: YAML

    parameters:
      output_frame:
        codec: jpeg
        encoder_params:
          quality: 90