Defining a Custom ROI
=====================

By default, Savant creates the automatic default ROI, which covers all the frame space. Sometimes developers need to reconfigure the ROI globally or per frame. It can be done with custom ``pyfunc`` units.

Define Custom ROI
-----------------

TODO

Delete Default ROI
------------------

An example of how to delete the default ROI object from a frame is demonstrated in the Traffic Meter `demo <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`__. In the demo, when the lines are not configured for a source, the default ROI is removed from a frame.

.. literalinclude:: ../../../samples/traffic_meter/line_crossing.py
  :language: YAML
  :lines: 12-35

