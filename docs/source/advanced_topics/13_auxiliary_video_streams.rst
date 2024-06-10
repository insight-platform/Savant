Auxiliary Video Streams
-----------------------

There are situations, when you want to create a video stream which is artificial and does not match to any source video streams.

Examples of such streams are:

- Grid streams (e.g. 2x2 grid of videos);
- Video streams with augmentations (e.g. video with a sidebar or dashboard);
- Videos of other resolutions, e.g. you transcode incoming 4k to 1080p, 720p, 480p, etc.
- Super-resolution streams, e.g. you upscale incoming 720p to 1080p.
- Frame interpolation streams, e.g. you create a 60fps video from 30fps video.

Savant 0.4.1 introduces the concept of auxiliary streams. An auxiliary stream is a stream that is not directly associated with any source stream. It is created by the user in PyFunc and is sent directly to pipeline sink , bypassing downstream pipeline elements.

The sample showing how to work with auxiliary streams is available in the `examples/auxiliary_streams <https://github.com/insight-platform/Savant/tree/develop/samples/auxiliary_streams>`__ directory. The sample shows how to convert a video stream to multiple resolutions and send them to the pipeline sink.

.. note::

    Auxiliary streams deliver metadata provided by the user.

