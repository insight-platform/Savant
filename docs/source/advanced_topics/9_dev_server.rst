DevServer Mode
--------------

DevServer is a special module execution mode enabling change detection in custom Python code and reloading those pieces automatically without the need for container restarts. It saves time significantly because code reloading happens instantly when the module processes the next frame after detecting the code change. Contrarily, manual restarts, described in the previous section, take many seconds because neural models must be loaded and initialized.

The mechanism currently has limitations:

- It affects only the Python code specified in the module manifest, so additional dependencies imported in the main code are not checked for changes.
- DevServer mode does not fit production use from the performance perspective; after the development is complete, disable the DevServer functionality.

.. warning::

    Changes in YAML and other resources are ignored and do not cause code hot reloading.

The DevServer functionality can be enabled in the manifest by setting the parameter:

.. code-block:: yaml

    parameters:
      dev_mode: True

In the `template <https://github.com/insight-platform/Savant/tree/develop/samples/template>`_ module manifest the functionality is enabled by default.
