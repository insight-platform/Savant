Installing Savant
=================

The recommended way to run Savant currently does not require any installation steps.

Savant modules are run in the framework or module docker containers (look for details in the :doc:`next section <running>`).

Requirements
------------

Nvidia container runtime
^^^^^^^^^^^^^^^^^^^^^^^^

Requirements for running docker containers with NVIDIA hardware support are described in NVIDIA Container Toolkit
`Installation Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

Platform and OS compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Savant builds on top of Deepstream ecosystem, therefore, it currently requires

* R515.65.01 display driver for dGPU platform
* Jetpack 5.0.2 GA on Jetson AGX Xavier / NX / Orin
* Jetpack 4.6.1 GA on Jetson Nano

You can look for detailed environment setup instructions in the Nvidia `Quickstart Guide <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#quickstart-guide>`_.
