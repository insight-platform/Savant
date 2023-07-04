Running a Savant module
=======================

Pieces of Savant framework are normally run in a docker containers. We provide docker compose files for examples we have developed so far. To get an idea how to run a Savant module, let us take a look at the docker compose from one of the Savant examples:

.. literalinclude:: ../../../samples/nvidia_car_classification/docker-compose.x86.yml
  :language: YAML

The file introduces several services:

- ``module``:  the Savant module that runs video analytics on the video streams;
- ``video-loop-source``: the adapter that ingests looped video stream from a file to the module;
- ``always-on-sink``: the adapter that receives the results of video analytics and represents them as RTSP-stream.

In the following sections, we dive into the details of modules and explain what the adapters are and how do they communicate with modules.
