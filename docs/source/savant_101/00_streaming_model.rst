Streaming Model
===============

In vanilla DeepStream, the sources and sinks are integral to the GStreamer pipeline because it's by design. However, such a design makes it difficult to create reliable applications in the real world.

There are reasons for that. The first one is low reliability. The source and sink are external entities that, being coupled into the pipeline, make it crash when they are failed or are no longer available. E.g., when the RTSP camera is unavailable, the corresponding RTSP Gstreamer source signals the pipeline to terminate.

The problem becomes more serious when multiple sources ingest data into a single pipeline - a natural case in the real world. You don't want to load multiple instances of the same AI models into the GPU because of RAM limitations and overall resource over utilization. So, following the natural Gstreamer approach, you have a muxer scheme with a high chance of failing if any source fails.

That's why you want to have sources decoupled from the pipeline - to increase the stability of the pipeline and avoid unnecessary reloads in case of source failure.

Another reason is dynamic source management which is a very difficult task when managed through the Gstreamer directly. You must implement the logic that attaches and detaches the sources and sinks when needed.

The third problem is connected to media formats. You have to reconfigure GStreamer pads setting proper capabilities when the data source changes the media format, e.g., switching from h.264 to HEVC codec. The simplest way to do that is to crash and recover, which causes significant unavailability time while AI models are compiled to TensorRT and loaded in GPU RAM. So, you want to avoid that as well.

The framework implements the handlers, which magically address all the mentioned problems without needing to manage them explicitly. It helps the developer process streams of anything without restarting the pipeline. The video files, sets of video files, image collections, network video streams, and raw video frames (USB, GigE) are all processed universally (and can be mixed) without reloading the pipeline to attach or detach the stream.

The framework virtualizes the stream concept by decoupling it from the real-life data source and takes care of a garbage collection for no longer available streams.

A savant module is capable to handle many video streams in parallel. The module accepts all those streams through a single multiplexing socket implemented with ZeroMQ. That socket enables injecting multiple isolated or already multiplexed streams into a module.

The module de-multiplexes those streams into an internal representation, where each stream is organized into a single queue. On the output side, the module multiplexes the streams back and sends them into the output ZeroMQ socket. You may find how it works on the following diagram:

.. image:: ../_static/img/0_streaming_model_mux_demux.png

Basically, the developer doesn't care about how to handle multiple streams. She must be aware that there are multiple streams, not a single one. Especially this is important if the code keeps the state of every stream, e.g., when counting people on a per-stream basis. To make this sort of state work properly, it must consider the stream's ``source_id``. The API allows the developer to retrieve per-stream information to address the situation.

Such a multiplexed/de-multiplexed model of operation is very beneficial because every Savant module is production-ready from scratch when is crafted properly. The module doesn't know whether it processes a file-based stream, a live stream, or just a bunch of images; it doesn't distinguish between them. The framework ensures that all those streams can be processed in the same way. More to say, mix of various streams can be processed at the same time by the same module.

Streams may appear and disappear dynamically, the framework handles such situations transparently to the developer, but providing handy callbacks if the developer wants to know when the stream is no longer exist.

To communicate with source and sink sockets Savant uses adapters: special modules that inject or receive streams from the module.
