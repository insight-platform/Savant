Configure The Development Environment
=====================================

Ensure that you configured the host system according to :doc:`0_configure_prod_env`.

The Savant framework requires a complex environment, including GStreamer, Nvidia DeepStream, OpenCV CUDA, and other dependencies. Installing and configuring all the required packages is a daunting task. Therefore, we prepared ready-to-use docker images both for development and production usage. The image provides you with a fully configured environment in minutes.

Development Environment Problem
-------------------------------

When we develop the code, we expect IDE's assistance on available classes, their methods, properties, etc. The Savant codebase is in Docker, which makes a problem for the IDE because it cannot collect the database for symbols.

Normally, we use VENV to install packages, and IDE knows how to gather symbols from VENV; however, the Savant environment is too complex to install in VENV.

Fortunately, popular Python IDEs like PyCharm Professional and VSCode support setting up a development environment in a docker container: it gives the IDE information from where to collect available packages and their symbols.

Configuration Manuals
---------------------

We will guide how to configure PyCharm Professional and VSCode to use runtime environment in Docker. The GitHub repository contains a module template; we will use it to demonstrate configuring the development environment in PyCharm Professional and VSCode.

.. warning::

    PyCharm Community Edition doesn't support development in the Docker container environment, so it cannot be used easily. If you consider only free-of-charge IDEs, consider VSCode.

.. note::

    Recommended spare disk space is **20 GB**.


.. toctree::
   :maxdepth: 1

   1_pycharm
   1_vscode

