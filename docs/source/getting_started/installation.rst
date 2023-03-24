Installing Savant
=================

The recommended way to run :repo-link:`Savant` currently does not require any installation steps.

Savant modules are run in the framework or module docker containers (look for details in the :doc:`next section <running>`).

Requirements
------------

Nvidia container runtime
^^^^^^^^^^^^^^^^^^^^^^^^

Requirements for running docker containers with NVIDIA hardware support are described in NVIDIA Container Toolkit
`Installation Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.

Platform and OS compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:repo-link:`Savant` builds on top of Deepstream ecosystem, therefore, it currently requires

* R525+ display driver for dGPU platform
* Jetpack 5.1 GA on Jetson AGX Xavier / NX / Orin

You can look for detailed environment setup instructions in the Nvidia `Quickstart Guide <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#quickstart-guide>`_.
