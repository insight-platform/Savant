Frame filtering
===============

Source and sink elements in a Savant pipeline can include a filter function. This function is called for every frame, and can be used to filter out frames that should not be processed by the pipeline (ingress filter in a source element) or frames that should not be passed to the next pipeline (egress filter in a sink element).

This is useful, for example, in the case :ref:`savant_101/12_video_processing:Conditional Encoding` is defined and as a result the pipeline produces frames without video data. Egress filtering can be used to filter out these frames before they are passed to the next pipeline.

Frame filtering functions work similarly to :doc:`/savant_101/70_python`, with a couple of differences.

Filtering function config
-------------------------

To define an ingress filtering function, add an ``ingress_frame_filter`` node to the :py:class:`~savant.config.schema.SourceElement` definition, e.g.

.. code-block:: yaml

    source:
      element: zeromq_source_bin
      properties:
        socket: router+bind:ipc:///tmp/zmq-sockets/input-video.ipc
      ingress_frame_filter:
        module: path.to.module
        class_name: IngressFilter

.. note::

    Ingress filter can only be configured for ``zeromq_source_bin`` source.

Likewise for egress filtering, add an ``egress_frame_filter`` node to the :py:class:`~savant.config.schema.SinkElement` definition.

.. code-block:: yaml

    sink:
      - element: zeromq_sink
        properties:
          socket: pub+bind:ipc:///tmp/zmq-sockets/output-video.ipc
        egress_frame_filter:
          module: path.to.module
          class_name: EgressFilter

Defining the filtering functions is optional. By default, ingress filters out all frames without video, and egress is bypassed.

Filtering function code
-----------------------

Custom ingress and egress filters must implement :py:class:`~savant.base.frame_filter.BaseFrameFilter` interface. The interface defines one method:

.. code-block:: python

    def __call__(self, video_frame: VideoFrame) -> bool:
        """Return True if the frame should be processed, False otherwise."""

Default ingress filter provides an example of filter that drops frames without video data:

.. literalinclude:: ../../../savant/base/frame_filter.py
    :pyobject: DefaultIngressFilter
