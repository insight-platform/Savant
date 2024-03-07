Configure The Runtime Environment
=================================

The section discusses how to configure the host system for Savant modules execution. Savant modules and adapters are run in the docker environment. This document describes how to configure the dockerized runtime.

General Requirements
--------------------

:repo-link:`Savant` runs on top of DeepStream ecosystem, therefore, it requires DeepStream to be supported by the host system. The section observes the supported configurations. The recommended spare space in a filesystem where docker images are stored is **15 GB**.

Data Center, Professional And Desktop Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Docker with Compose, Nvidia GPU drivers R525 (data center hardware), 530+ professional and desktop hardware;

Edge Hardware
^^^^^^^^^^^^^

Docker with Compose, Jetpack 6.0 DP on Jetson AGX Orin, Orin NX, Orin Nano.

Previous Savant Versions
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Device
      - Support Notes
      - JetPack Requirements
    * - Jetson Nano 1st gen, TX1, TX2
      - Not supported
      - Not supported
    * - Jetson Xavier Family
      - Savant 0.2.x
      - Jetpack 5.1.2, 5.1.3
    * - Jetson Orin Family
      - Savant 0.2.x
      - Jetpack 5.1.2, 5.1.3

Nvidia Jetson Setup
-------------------

An Nvidia Jetson device is almost ready to run Savant after setup. You only need to install **Compose** plugin for Docker. Follow the official `guide <https://docs.docker.com/compose/install/linux/>`_ to install it.

Ubuntu 22.04 Setup
------------------

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

   sudo apt install --no-install-recommends nvidia-driver-535
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

Test The Nvidia Container Runtime Works Properly (X86 only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

Test Docker Ecosystem Works Properly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are on X86 + Nvidia dGPU:

.. code-block:: bash

   sudo docker compose -f samples/opencv_cuda_bg_remover_mog2/docker-compose.x86.yml up
   # press Ctrl+C to stop the container

If you are on Jetson:

.. code-block:: bash

   sudo docker compose -f samples/opencv_cuda_bg_remover_mog2/docker-compose.l4t.yml up
   # press Ctrl+C to stop the container

Check that streaming works properly:

.. code-block:: bash

   ffplay rtsp://127.0.0.1:554/stream/road-traffic-processed

You must see the video stream as demonstrated in the following Youtube video:

.. youtube:: P9w-WS6HLew

Disable SUDO for Docker
^^^^^^^^^^^^^^^^^^^^^^^

We often assume that Docker is available without ``sudo``, for simplicity you can add your user into the ``docker`` group to avoid using ``sudo``.

.. code-block:: bash

   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
