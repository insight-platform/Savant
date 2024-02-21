Configure The Production Environment
====================================

The section discusses how to configure the host system for Savant modules execution. Savant modules and adapters are run in the docker environment. This document describes how to configure the dockerized runtime.

General Information
-------------------

.. warning::

   **Jetson Nano 1st Gen:** Savant does not support 1st-gen Jetson Nano due to the outdated software stack supported by the device.

.. warning::

    We do not support Windows as a host OS for Savant. WSL2 is not supported by DeepStream and is not recommended for use with Savant. If you are using Windows, you do it on your own risk. We cannot help with problems and issues.

:repo-link:`Savant` runs on top of DeepStream ecosystem, therefore, it requires DeepStream dependencies to be satisfied:

* Docker with Compose, Nvidia GPU drivers R525 (data center hardware), 530+ professional and desktop hardware;
* Docker with Compose, Jetpack 5.1.2 GA on Jetson AGX Xavier/ NX, Orin Family.

You can look for detailed environment setup instructions in the Nvidia `Quickstart Guide <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#quickstart-guide>`_.

The recommended spare space in a filesystem where docker images are stored is **15 GB**.

Nvidia Jetson
-------------

An Nvidia Jetson device is ready to run Savant after setup.

Ubuntu 22.04
------------

At the current moment, we provide the instruction on how to configure Ubuntu 22.04 runtime. If you use another operation system, adapt the instructions to your OS.

Update Packages and Install Basic Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git git-lfs curl -y

Install Docker
^^^^^^^^^^^^^^

.. code-block:: bash

   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

Install Nvidia Drivers
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt install --no-install-recommends nvidia-driver-530
   sudo reboot

Install Nvidia Container Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

Test The Runtime Works Properly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

