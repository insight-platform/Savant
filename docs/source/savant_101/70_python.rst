Python Function Unit
====================

The Python Function Unit is used to include arbitrary custom Python code in the pipeline. To work with ``pyfunc``, custom code must be implemented as a successor to :py:class:`~savant.deepstream.NvDsPyFuncPlugin`, which exposes two methods: the first allows frames for each source to be handled separately, the second supports processing the frames for the whole batch.

Per-source processing, normally you want to use this method:

.. code-block:: python

    from savant.deepstream.pyfunc import NvDsPyFuncPlugin
    class CustomProcessing(NvDsPyFuncPlugin):
      def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Put custom frame processing code here."""

Per-batch processing:

.. code-block:: python

    from savant.deepstream.pyfunc import NvDsPyFuncPlugin
    class CustomProcessing(NvDsPyFuncPlugin):
      def process_buffer(self, buffer: Gst.Buffer):
        """Process Gst.Buffer, parse DS batch manually."""

In addition to the above methods, NvDsPyFuncPlugin also allows you to override several auxiliary methods with which the code can respond to various pipeline events.

The ``on_start`` method is called when the pipeline starts processing data, once for all frames and sources. By default, it is defined as follows:

.. code-block:: python

    def on_start(self) -> bool:
        """Do on plugin start."""
        return True

It is important to pay attention to the return value: if it returns ``False`` or ``None``, the pipeline will not start processing.

The ``on_stop`` method is similar to the ``on_start`` method: it is called once when the pipeline stops; defined by default like this:

.. code-block:: python

    def on_stop(self) -> bool:
        """Do on plugin stop."""
        return True

The ``on_src_event`` method is called for GStreamer events received on the src pad of the Python Function Unit and makes it possible to process them:

.. code-block:: python

    def on_src_event(self, event: Gst.Event):
        """Do on src event."""

It is an analogue of `GstBaseTransform.do_src_event() <https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=python#GstBaseTransformClass::src_event>`__, however, it does not replace the basic implementation, but is called before it. Therefore, the callback does not require any return value.

The ``on_sink_event`` method is similar to ``on_src_event``, but works for events on the sink pad. Also called just before the base GStreamer implementation of the `GstBaseTransform.do_sink_event() <https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=python#GstBaseTransformClass::sink_event>`__ method; is defined like this:

.. code-block:: python

    def on_sink_event(self, event: Gst.Event):
        """Add stream event callbacks."""

When overriding this method, it is important to note that, by default, the ``NvDsPyFuncPlugin.on_sink_event`` method handles the ``GST_NVEVENT_STREAM_EOS`` DeepStream event (gst-nvevent.h `documentation <https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/gst-nvevent_8h.html>`__), determining which source it refers to, and calls the ``on_source_eos`` method. Therefore, when implementing your version of the event handler on the sink pad, it is worth including a call to the parent class method in it.

The ``on_source_eos`` method is called for every ``GST_NVEVENT_STREAM_EOS`` event that arrives on the DeepStream sink pad. The purpose of this method is to handle the situation when a data stream corresponding to a particular source ends.

Source addressing is achieved by reading ``frame_meta.source_id``, which corresponds to the identifier of the source defined by an external system ingesting frames. The ``on_source_eos`` method can be used to release the state resources allocated for a particular source. For example, delete information about the tracks of this source.

.. code-block:: python

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""

The Python Function Unit is specified in the configuration by the ``pyfunc`` unit, specifying the required configuration parameters ``module`` and ``class_name``, where:

* ``module`` is a string indicating either the file system path to the user code file or a valid Python path to the code module.
* ``class_name`` is a string with the name of the class that performs the processing.

A Python path example:

.. code-block:: yaml

    - element: pyfunc
      module: samples.line_crossing.line_crossing
      class_name: LineCrossing

A filesystem path example:

.. code-block:: yaml

    - element: pyfunc
      module: /opt/app/samples/line_crossing/line_crossing.py
      class_name: LineCrossing

Also, the ``pyfunc`` unit configuration allows setting an arbitrary set of user parameters through the ``kwargs`` key:

.. code-block:: yaml

    - element: pyfunc
      module: /opt/app/samples/line_crossing/line_crossing.py
      class_name: LineCrossing
      kwargs:
        config_path: /opt/app/samples/line_crossing/line_crossing.yml

Parameters defined with ``kwargs`` are available as ``pyfunc`` class instance attributes.

