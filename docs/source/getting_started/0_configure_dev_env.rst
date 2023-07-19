Configure The Development Environment
=====================================

The Savant framework requires a complex environment, including GStreamer, Nvidia DeepStream, OpenCV CUDA, and other dependencies. Installing and configuring all the required packages is a daunting task. Therefore, we prepared ready-to-use docker images both for development and production usage. The image provides you with a fully configured environment in minutes.

Popular Python IDEs like PyCharm Professional and VSCode support the feature to set up a development environment inside a  docker container: it gives the best opportunity to configure the dev runtime for Savant quickly. The Savant repository includes a module template for a quick start; we will use it to show you how to set up the development environment in popular IDEs.

.. note:: Unfortunately, the PyCharm Community edition doesn't support the environment on Docker, so you cannot use it to reproduce the presented instructions.

.. note:: The recommended spare space in a filesystem where docker images are stored is **20 GB**.

.. toctree::
   :maxdepth: 1
   :caption: Prepare Environments

   0_configure_dev_env_pycharm
   0_configure_dev_env_vscode

