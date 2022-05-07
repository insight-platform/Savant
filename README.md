# Savant

Savant is an open-source, high-level toolkit for implementing real-time, streaming, highly efficient multimedia AI applications on the Nvidia stack. It makes it possible to develop the inference pipelines which utilize the best Nvidia approaches for both data center and edge accelerators with a very moderate amount of code in Python.

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing high-performance real-time streaming AI applications that run several times faster than conventional AI applications executed within the conventional runtimes like PyTorch, TensorFlow and similar. 

That is possible due to specially designed architecture, which utilizes the best Nvidia accelerator features, including hardware encoding and decoding for video streams, moving the frames thru inference blocks mostly in GPU RAM without excessive data transfers between CPU and GPU RAM. It also stands on a highly optimized low-level software stack that optimizes inference operations to get the best of the hardware used.
