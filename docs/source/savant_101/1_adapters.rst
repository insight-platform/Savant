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
