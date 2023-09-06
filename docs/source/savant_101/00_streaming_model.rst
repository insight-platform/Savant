Streaming Model
===============

In vanilla DeepStream, the sources and sinks are by-design integrated to the GStreamer pipeline. Unfortunately, such a design makes it difficult to create reliable applications for the real business cases.

There are reasons for that. The first one is decreased reliability: the source and sink are external entities that, being coupled into the pipeline, will cause its outage when they fail. E.g., when the RTSP camera becomes unavailable, the corresponding GStreamer RTSP source will signal the pipeline to terminate.

The problem becomes more serious when multiple sources send data into a single pipeline: a natural case in the real world. Normally, you want avoid loading multiple instances of the same AI models into the GPU because of RAM limitations and resource over-utilization. So, following the natural GStreamer approach, you multiplex data streams into a single pipeline, making a program prone to crash when any source fails.

That's why you want to have sources decoupled from the pipeline: to increase the stability and avoid unnecessary reloads in case of source failure.

Another reason is dynamic source management which is a challenging task when managed through the GStreamer directly: to succeed, you must implement the cumbersome, quirky logic that handles attaches and detaches for the sources and sinks when needed.

The third problem is related to media formats. You need to reconfigure GStreamer elements setting proper capabilities when the data source changes the media format, e.g., switching from H.264 to the HEVC codec. The simplest way, again, is to reload the pipeline, which causes significant unavailability, while AI models are loaded in GPU. So, in real-world pipelines, you want to avoid it.

The framework implements the logic, which handles all the mentioned problems without needing to manage them explicitly. It enables processing streams of anything without restarting the pipeline: the video files, sets of video files, image collections, network video streams, and raw video frames (USB, GigE) are all processed universally (and can be mixed) without reloading the pipeline when the codec is changed, or a new stream is attached or detached.

The framework virtualizes the stream concept by decoupling it from a particular real-world data device and takes care of a garbage collection for no longer available streams.

A Savant module is capable to handle many video streams in parallel. The module accepts all those streams through a single ZeroMQ socket multiplexing them. That socket enables sending multiple isolated or already multiplexed streams into a module.

The module de-multiplexes those streams internally to the form where each stream is organized into a single queue. On the output side, the module multiplexes the streams back and sends them into the output ZeroMQ socket. You may find how it works on the following diagram:

.. image:: ../_static/img/0_streaming_model_mux_demux.png

Developers does not care about how to handle multiple streams. However, they must be aware that there are numerous streams, not one. Especially, It is important when the code maintains a per-stream state, e.g., when counting people on video. To ensure the state is handled properly, developers must consider using stream ``source_id``. The API allows developers to retrieve per-stream information to control the situation.

Such a multiplexed/de-multiplexed operation model is very beneficial because any well-made Savant module immediately becomes production-ready. The module does not know whether it processes a file-based stream, a live stream, or just a bunch of images; it does not distinguish between them. The framework ensures the processing for all those streams in the same way. More to say, a mix of various streams can be processed simultaneously by the same module.

Streams may appear and disappear dynamically: the framework handles such situations transparently to the developer, providing handy callbacks if they want to know when the stream no longer exists.

To communicate with source and sink sockets, Savant uses adapters: special modules that inject or receive streams from the module.
