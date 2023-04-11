Adapters
========

The adapter is a separate program executed in a Docker container. Adapters communicate with modules via ZeroMQ sockets: source adapters send data into a module, and sink adapters receive data from a module.

The decoupled nature of adapters provides better reliability because a failed data source affects the adapter operation, not a framework operation. As a result, the module is always available regardless of data sources.

The adapters can transfer the media through the network or locally. We have already implemented several handy adapters, and you can implement the required one if needed - the protocol is simple and utilizes standard open-source tools.

Savant Adapter Protocol
-----------------------

There is a protocol based on ZeroMQ and Apache Avro, which is used by adapters to communicate. It can be utilized to connect adapter with adapter, adapter with module, module with module, etc. The protocol is universal for sources and sinks. Simply speaking, the protocol allows transferring the following:

- video frame [optional];
- video stream information;
- per-frame tags;
- the hierarchy of objects related to the frame.

We are extending the protocol to support new framework features as they happen. The protocol is described in the `API <https://github.com/insight-platform/Savant/tree/develop/savant/api/avro-schemas>`_ section.

Communication Sockets
---------------------

Adapters and Modules can use various ZeroMQ socket pairs to establish the communication. The chosen type defines the possible topologies and quality of service. Currently the following pairs are supported:

- Dealer/Router - reliable, asynchronous pair with backpressure (default choice);
- Req/Rep - reliable, synchronous pair (paranoid choice);
- Pub/Sub - unreliable, real-time pair (default choice for strict real-time operation or broadcasting).

We haven't integrated adapters information to the current documentation yet, so please read a separate `document <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md>`_ describing on how to use them.

Supported Adapters
------------------

We provide adapters to address the everyday needs of users. The current list of adapters enables the implementation of many typical scenarios in real life. Every adapter can be used as an idea to implement a specific one required in your case.

Source Adapters
^^^^^^^^^^^^^^^

Currently, the following `source <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md#source-adapters>`_ adapters are available:

- Video Loop Adapter;
- Local video file;
- Local directory of video files;
- Local image file;
- Local directory of image files;
- Image URL;
- Video URL;
- RTSP stream;
- USB/CSI camera;
- GigE (Genicam) industrial cam.

There are basic `sink <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md#sink-adapters>`_ adapters implemented:

- Inference results are placed into JSON file stream;
- Resulting video overlay displayed on a screen (per source);
- MP4 file (per source);
- image directory (per source);
- Always-On RTSP Stream Sink.
