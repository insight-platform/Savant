Configure The Documentation Environment
=======================================

The simplest way to work on the Savant documentation is to use a dockerized Sphinx environment. We provide instructions on how to build such a runtime. The GPU is not required to work on the documentation.

The Dockerized Environment
--------------------------

You need 20GB on local drive to build the environment properly.

Fetch Base Images For Nvidia DeepStream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend pulling Nvidia docker containers separately because the Nvidia registry is configured in such a way to kick the long-waiting pullers.

.. code-block:: bash

   docker pull nvcr.io/nvidia/deepstream:6.2-base
   docker pull nvcr.io/nvidia/deepstream:6.2-devel

Build The Dockerized Sphinx Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This operation must be executed only once. It will take a while to build the runtime.

.. code-block:: bash

   make build-docs

Compile The Documentation From Sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation, call the following command:

.. code-block:: bash

    make run-docs

Access The Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

open ``build/html/index.html`` in your browser.
