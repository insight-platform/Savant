Nvidia Tracker Unit
===================

The Nvidia Tracker Unit is designed for use in the tracking pipeline implemented through the Nvidia ``Gst-nvtracker`` plugin, including the Nvidia NvMultiObjectTracker tracking library.

The Nvidia Tracker Unit is specified in the Savant module configuration with the ``nvtracker`` unit:

.. code-block:: yaml

    - element: nvtracker


The element does not have any Savant-specific configuration parameters. All possible parameters must be specified under the ``element.properties`` section and will be passed directly to the Gstreamer element. You can find a complete list of possible parameters at the Gst-nvtracker properties `link <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#gst-properties>`_.

The mandatory configuration parameter for nvtracker is ``properties.ll-lib-file``, which specifies the path to the tracking library. The configuration that uses the NvMultiObjectTracker library provided within Deepstream looks like the following:

.. code-block:: yaml

    - element: nvtracker
      properties:
        ll-lib-file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so


To fine-tune the tracker, you can pass a tracking configuration file to the library using the ``properties.ll-config-file parameter``:

.. code-block:: yaml

    - element: nvtracker
      properties:
        ll-lib-file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
        ll-config-file: ${oc.env:APP_PATH}/samples/line_crossing/config_tracker_NvDCF_perf.yml


In the example above, ``config_tracker_NvDCF_perf.yml`` specifies a configuration file for one of the tracking presets provided by Nvidia. This and other configuration files can be found in the DeepStream development image at ``/opt/nvidia/deepstream/deepstream-6.2/samples/configs/deepstream-app``. Nvidia provides four different approaches to tracking, which are described in detail in the Gst-nvtracker Tracker library `documentation <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#id9>`__.

The **IOU** tracker is the most lightweight, but suitable only for the simplest scenes. The configuration file is ``config_tracker_IOU.yml``.

**NvSORT** is also also a high performance with improved association due to the use of the Kalman filter. The configuration file is ``config_tracker_NvSORT.yml``.

**NvDeepSORT** is a visual tracker that uses a GPU-based ReID model to improve association in a complex environment. The configuration file is ``config_tracker_NvDeepSORT.yml``.

**NvDCF** is a flexible tracker based on discriminative correlation filter, wich also can use ReID. Three configuration presets are available:

* ``config_tracker_NvDCF_accuracy.yml``;
* ``config_tracker_NvDCF_perf.yml``;
* ``config_tracker_NvDCF_max_perf.yml``.

More information about parameters of Nvidia's object tracking library can be found in NvMultiObjectTracker `documentation <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#configuration-parameters>`__.

As an alternative to Nvidia's trackers, it is possible to implement custom object tracking, which is described in more detail in advanced topics.









