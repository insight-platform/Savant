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

The protocol is described in the `API <https://github.com/insight-platform/Savant/tree/develop/savant/api/avro-schemas>`_ section.

Communication Sockets
---------------------

The adapters and modules can use various ZeroMQ socket pairs to establish communication. The chosen type defines the possible topologies and quality of service. Currently, the following pairs are supported:

- Dealer/Router: reliable, asynchronous pair with backpressure (default choice);
- Req/Rep: reliable, synchronous pair (paranoid choice);
- Pub/Sub: unreliable, real-time pair (default choice for strict real-time operation or broadcasting).

We haven't integrated adapters information to the current documentation yet, so please read a separate `document <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md>`_ describing on how to use them.

Supported Adapters
------------------

We provide adapters to address the everyday needs of users. The current list of adapters enables the implementation of many typical scenarios in real life. Every adapter can be used as an idea to implement a specific one required in your case.

Source Adapters
^^^^^^^^^^^^^^^

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

Sink Adapters
^^^^^^^^^^^^^

There are basic `sink <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md#sink-adapters>`_ adapters implemented:

- Inference results are placed into JSON file stream;
- Resulting video overlay displayed on a screen (per source);
- MP4 file (per source);
- image directory (per source);
- Always-On RTSP Stream Sink.
