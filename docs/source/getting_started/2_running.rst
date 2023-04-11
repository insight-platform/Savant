Running a Savant module
=======================

Every piece of Savant framework is executed in a docker container normally. We provide such compose files for evey example we have developed so far. To get idea how to run a Savant module, let us take a look at the docker compose from one of the Savant examples:

.. literalinclude:: ../../../samples/nvidia_car_classification/docker-compose.x86.yml
  :language: YAML

The file introduces several services:

- ``module``:  Savant module that runs video analytics on the video streams;
- ``rtsp-source``: adapter that provides video streams from RTSP cameras to the module;
- ``always-on-sink``: adapter that receives the results of video analytics and represents them in the form of RTSP-stream.

In the next sections, we dive into the details of each service. We will explain what are the adapters and how do they communicate with modules.
