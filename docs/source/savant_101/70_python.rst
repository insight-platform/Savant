Python Function Unit
====================

The Python Function Unit is used to include arbitrary custom Python code in the pipeline. To work with ``pyfunc``, custom code must be implemented by specifying :py:class:`~savant.deepstream.NvDsPyFuncPlugin` as the parent class, which exposes two methods: the first allows frames for each source to be handled separately, the second supports processing the frames for the whole batch.

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

The ``on_event`` method is called for GStreamer events received on the sink pad of the Python Function Unit and makes it possible to process them:

.. code-block:: python

    def on_event(self, event: Gst.Event):
        """Do on event."""

It is an analogue of `GstBaseTransform.do_sink_event() <https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=python#GstBaseTransformClass::sink_event>`__, however, it does not replace the basic implementation, but is called before it. Therefore, the callback does not require any return value.

When overriding this method, it is important to note that, by default, the ``NvDsPyFuncPlugin.on_event`` method handles ``GST_NVEVENT_PAD_ADDED``, ``GST_NVEVENT_PAD_DELETED``, and ``GST_NVEVENT_STREAM_EOS`` DeepStream events (gst-nvevent.h `documentation <https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/gst-nvevent_8h.html>`__), determining which source it refers to, and calls ``on_source_add``, ``on_source_delete``, and ``on_source_eos`` callbacks respectively. Therefore, when implementing your version of the event handler on the sink pad, it is worth including a call to the parent class methods in it.

The ``on_source_add`` and ``on_source_delete`` methods are called for ``GST_NVEVENT_PAD_ADDED`` and ``GST_NVEVENT_PAD_DELETED`` events arriving at the PyFunc sink pad, respectively. The purpose of these methods is to handle the situation in which a source is added or a source is deleted. Methods are useful for state management, such as creating and releasing the state resource allocated to a particular source.

.. code-block:: python

    def on_source_add(self, source_id: str):
        """On source add callback."""

    def on_source_delete(self, source_id: str):
        """On source delete callback."""

The ``on_source_eos`` method is called for every ``GST_NVEVENT_STREAM_EOS`` event that arrives on the PyFunc sink pad. The purpose of this method is to handle the situation when a data stream corresponding to a particular source ends. The method can be used to free resources that are assigned to a specific source and are relevant within a single stream of the source. For example, delete information about the tracks of this stream.

.. code-block:: python

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""

Source addressing is achieved by reading ``frame_meta.source_id`` property, which corresponds to the identifier of the source defined by an external system ingesting frames.

The Python Function Unit is specified in the configuration by the ``pyfunc`` unit, specifying the required configuration parameters ``module`` and ``class_name``, where:

* ``module`` is a string indicating either the file system path to the user code file or a valid Python path to the code module.
* ``class_name`` is a string with the name of the class that performs the processing.

A Python path example:

.. code-block:: yaml

    - element: pyfunc
      module: samples.traffic_meter.line_crossing
      class_name: LineCrossing

A filesystem path example:

.. code-block:: yaml

    - element: pyfunc
      module: /opt/savant/samples/traffic_meter/line_crossing.py
      class_name: LineCrossing

Also, the ``pyfunc`` unit configuration allows setting an arbitrary set of user parameters through the ``kwargs`` key:

.. code-block:: yaml

    - element: pyfunc
      module: /opt/savant/samples/traffic_meter/line_crossing.py
      class_name: LineCrossing
      kwargs:
        config_path: /opt/savant/samples/traffic_meter/line_crossing.yml

Parameters defined with ``kwargs`` are available as ``pyfunc`` class instance attributes.

Grouping PyFuncs Into a Single Element
--------------------------------------

When a pipeline contains several Python Function Units that always run one
after another, each of them is normally instantiated as a standalone
GStreamer element, and the framework inserts a queue between neighboring
units. Every element boundary adds an extra thread and buffer hand-off, which
is pure overhead for short, strictly sequential PyFuncs.

The ``pygroup`` unit removes this overhead by *colocating* multiple PyFuncs in
a single GStreamer element. The colocated PyFuncs are executed sequentially on
every frame, in the listed order, within the same element — without queues in
between. Each colocated PyFunc keeps its own telemetry span, so per-stage
observability is preserved (see below).

A ``pygroup`` unit is declared with the ``elements`` key, which holds a list of
regular PyFunc definitions. Every sub-element uses the same ``module``,
``class_name``, and (optionally) ``kwargs`` keys as a standalone ``pyfunc``:

.. code-block:: yaml

    - element: pygroup
      name: my_group
      elements:
        - module: module.pyfunc_module_1
          class_name: PyFuncClass1
        - module: module.pyfunc_module_2
          class_name: PyFuncClass2
          kwargs:
            key: value

Each sub-element must implement :py:class:`~savant.deepstream.NvDsPyFuncPlugin`,
exactly like a standalone ``pyfunc``. Everything a regular PyFunc can do also
works inside a group: receiving its own ``kwargs``, reading and writing object
metadata, drawing on frames, and creating :doc:`auxiliary video streams
</advanced_topics/13_auxiliary_video_streams>`. Savant inserts the necessary
queues *around* the group automatically, just as it does for a standalone
``pyfunc``.

.. note::

   ``pygroup`` is intended for short chains of PyFuncs that are effectively
   serial and benefit from lower per-element overhead. Heavy PyFuncs that
   should run on their own threads, in parallel with the rest of the pipeline,
   are better kept as standalone ``pyfunc`` units.

Telemetry
^^^^^^^^^

Colocating PyFuncs does not merge them into an opaque block for tracing
purposes. For every frame, ``pygroup`` opens a ``process-frame`` span for the
group and wraps each colocated PyFunc in its own nested span named
``<module>.<class_name>``:

.. code-block:: text

    process-frame
    ├── module.pyfunc_module_1.PyFuncClass1
    └── module.pyfunc_module_2.PyFuncClass2

PyFunc code may open further nested spans inside its own span for finer-grained
profiling. As a result, you can still measure each stage independently in
Jaeger or any other OTLP backend. See :doc:`/advanced_topics/9_open_telemetry`
for details on tracing in Savant.

Development Server
^^^^^^^^^^^^^^^^^^

The ``pygroup`` unit honors the :doc:`Development Server
</advanced_topics/9_dev_server>`: when the module runs in dev mode, the
colocated PyFuncs are reloaded on source changes just like standalone
``pyfunc`` units.

Sample
^^^^^^

The `pygroup sample <https://github.com/insight-platform/Savant/tree/develop/samples/pygroup>`__
colocates two overlay PyFuncs that draw sequentially on the frame, each feeding
its own auxiliary stream, and ships a preconfigured Jaeger/OTLP setup so the
per-stage spans can be inspected out of the box.
