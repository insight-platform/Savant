Hardware
========

Savant **requires** Nvidia dGPU or a modern Nvidia Jetson board to run.

The goal of Savant is to demonstrate the same behavior on all Nvidia Hardware: desktop, professional, data center, and edge. Our idea is to provide the framework helping developers create pipelines that work on every device, in other words, portable. However, various devices have certain limitations which must be addressed by a developer. Let us enumerate several of them:

1. **CPU performance**. Savant can work on both ``X86_64`` and ``AARCH64`` (Jetson Family). Nevertheless, ARM CPUs are less capable from the performance perspective, so CPU-bound pipelines suffer from CPU deficiency.

2. **RAM Capacity and architecture**. Savant takes care of RAM architecture, so usually, a developer doesn't care about whether the pipeline uses the unified memory of Jetson or dedicated RAM of Nvidia GPU, but great in-memory demand may cause situations when the device cannot allocate enough memory to handle the data, and Savant cannot track that.

   Also, systems with discrete GPUs and Jetson devices have different memory performance characteristics, which is important when frames are modified intensively. Savant provides two models: the first is based on memory mapping, and the second is based on the OpenCV CUDA library. Depending on the device, a developer should consider both to find which works better in a specific situation.

3. **Device limitations**. Nvidia may limit or exclude certain features from a device. Savant may detect situations when a developer tries to request functions absent on the device, but the framework doesn't guarantee that the pipeline will work without reconfiguring on such a device. In other words, the module may need other properties for certain devices.

   Let us discuss the most frequently met situation related to NVENC (Nvidia Encoder). Depending on the hardware NVENC may:

   * be limited by the number of simultaneously encoded streams (GeForce);
   * be absent at all (Tesla A100, H100, Nvidia Jetson Nano New);
   * have no limitations (other accelerators and boards).

   A developer must remember that they cannot use H264/HEVC-encoding if they use A100/H100 or that GeForce can be used for development when working with a single encoded stream, but it doesn't fit production needs. Savant works on all of those devices, but the module configuration may require changes.

Data Center GPUs
----------------

Savant works on any data center GPU starting from the Turing family, like Nvidia V100, A100, H100, T4, A2, A10, A30, etc.

Certain cards have a better hardware balance than others: e.g., when the pipeline is expected to encode video actively with high quality, you may find that certain GPUs like NVIDIA RTX 6000 Ada Generation or Nvidia L16 have more NVENC devices onboard than others, which may be beneficial.

When the pipeline doesn't generate resulting video footage, you may consider A100/H100 as the most efficient GPUs available.

.. note::

    Please take a look at the `hardware support matrix <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>`__ to find possible hardware limitations before deploying the hardware.

Professional GPUs
-----------------

Savant works on professional GPUs starting from the Turing family. Turing cards like RTX 4000/5000/6000/8000, and their Ampere descendants are supported.

.. note::

    Please take a look at the `hardware support matrix <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>`__ to find possible hardware limitations before deploying the hardware.

Desktop GPUs
------------

Savant works on GeForce Turing (RTX20), Ampere (RTX30, RTX40). Keep in mind that all consumer GPUs have NVENC limitations, so they can be used for development purposes, but not in production if NVENC is planned to be used at scale.

Edge Devices
------------

Savant doesn't work on the 1st gen Jetson Nano because Nvidia doesn't `support <https://www.reddit.com/r/JetsonNano/comments/wz034x/nvidia_abandones_jetson_nano/>`__ it in the recent software. If you need to use it, you cannot benefit from Savant. Savant is known to support:

- Jetson NX (regularly tested);
- Jetson AGX (regularly tested);
- Jetson Orin Family (Nano, NX, AGX).

Keep in mind that, unfortunately, Jetson Orin Nano doesn't have NVENC hardware.

.. note::

    If you have a positive or negative experience with Savant on certain hardware, please share; we will update the document accordingly or fix the bugs encountered. We don't have all the possible hardware in our lab, so knowing the situation is crucial to ensure that Savant supports the whole range of hardware.
