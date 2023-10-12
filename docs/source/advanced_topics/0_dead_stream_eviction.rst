Dead Streams Eviction
=====================

Savant automatically evicts dead streams. You know about such evictions by receiving corresponding events (via designated callbacks) in your Pyfuncs: normally, you do not need to manually deal with evictions. Instead, configuration parameters specified per source tune how frequently Savant checks the idle streams and how long the stream can be inactive before being considered dead.

The behavior is configured with the following parameters specified in the YAML configuration file per source:

* ``source_timeout`` (default: ``10`` seconds);
* ``source_eviction_interval`` (default: ``1`` second).

The first one specifies a period in seconds, defining how long  a stream can avoid sending frames to the pipeline before it is considered inactive;

The second specifies how often Savant checks for inactive streams and evicts them.

Depending on the nature of the data sources, you should tune the values to reflect real-life conditions and avoid sending EOS events to the functions when a stream is not expected to be dead.

Let us consider the situation when the pipeline is designed to process the data from RTSP cameras. In this case, you would like to decrease the ``source_timeout`` parameter to ``2-3`` seconds to reflect the always-availability nature of RTSP streams.

Another case is when you handle file fragments from S3. This case assumes that the fragment may be late. To address the situation, you may want to increase the ``source_timeout`` parameter to ``120`` seconds and ``source_eviction_interval`` to ``10`` seconds.

Take a look at the ``default.yaml`` for details:

.. literalinclude:: ../../../savant/config/default.yml
  :language: YAML
  :lines: 139-155

You can override only required parameters in your module YAML configuration file. Also, take a look at corresponding environment variables helping to configure the parameters without specifying them in the module config.
