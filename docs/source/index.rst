.. Savant documentation master file,
   should at least contain the root `toctree` directive.

Welcome to Savant documentation!
================================

Savant is a Python/C++ video analytics framework that helps building complex pipelines easily, leveraging optimized video processing and deep learning model inference capabilities of Gstreamer multimedia framework and NVIDIA Deepstream to provide fast performance.

With Savant deploying your models in an efficient pipeline can be as simple as writing a YAML configuration file, and custom model pre- and post-processing code integration is straightforward both for Python and C++.

Savant supports discrete NVIDIA GPU and NVIDIA Jetson platforms.

Start learning about Savant with :doc:`/getting_started/intro` and :doc:`/main_concepts/index`.

If any questions remain :doc:`/reference/api/index` might help to clear them up.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting_started/intro
   getting_started/installation
   getting_started/running
   getting_started/examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Main Concepts

   main_concepts/index
   main_concepts/module
   main_concepts/pipeline
   main_concepts/model_elements
   main_concepts/adapters

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference/api/index
   reference/avro


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
