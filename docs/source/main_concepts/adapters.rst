Adapters overview
=================

We call an adapter an independent process that either reads (source adapter) or writes (sink adapter) data from/to some location,
thus decoupling input/output operations from the main processing. All adapters are implemented as Docker images, and
Python scripts have been developed to simplify the process of running and using source and sink adapters.

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

:repo-link:`Savant` already includes several containerized `adapters <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md>`_ that are ready-to-use with any Savant module.

Each Savant adapter has a number of parameters which can be set through environment variables.
Learn more about general ZMQ parameters that are required for running a Savant module in combination
with Savant adapters in :doc:`../getting_started/2_running` or
read `adapters documentation <https://github.com/insight-platform/Savant/blob/develop/docs/adapters.md>`_ for specific Savant adapters descriptions and parameters.

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
