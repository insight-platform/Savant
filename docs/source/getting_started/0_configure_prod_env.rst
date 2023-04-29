Configure The Production Environment
====================================

Savant modules and adapters are executed in the docker environment. This document describes how to configure the dockerized runtime.

General Compatibility Notes
---------------------------

**Savant does not support 1st-gen Jetson Nano due to the outdated software stack supported by the device.**

:repo-link:`Savant` runs on top of DeepStream ecosystem, therefore, it requires DeepStream's dependencies to be satisfied:

* Docker with Compose, R525+ display driver for dGPU platform;
* Docker with Compose, Jetpack 5.1 GA on Jetson AGX Xavier/ NX, Orin Family.

You can look for detailed environment setup instructions in the Nvidia `Quickstart Guide <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#quickstart-guide>`_.

Ubuntu 22.04
------------

At the current moment, we provide the instruction on how to configure Ubuntu 22.04 runtime. If you have other operation system than Ubuntu, please adapt the instructions to conform your OS.

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

   sudo apt install --no-install-recommends nvidia-driver-525
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

   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
