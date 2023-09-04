Running Savant Module
=====================

Savant is a cloud-ready solution. You can easily run it in K8s, Docker and other containerized runtimes.

All pieces of Savant framework are normally run in docker containers. We provide docker compose files for all examples we have developed so far.

To get an idea how to run a Savant module, let us take a look at the docker compose from one of the Savant examples:

.. literalinclude:: ../../../samples/nvidia_car_classification/docker-compose.x86.yml
  :language: YAML

The file introduces several services:

- ``module``:  the Savant module that runs video analytics on the video streams;
- ``video-loop-source``: the adapter that ingests looped video stream from a file to the module;
- ``always-on-sink``: the adapter that receives the results of video analytics and represents them as RTSP or HLS stream.

In the following sections, we dive into the details of modules and explain what the adapters are and how do they communicate with modules.

The Savant module image provides a healthcheck to indicate when the module is ready to receive video frames. You can use it to start a source adapter only when the module is ready to receive frames by adding ``depends_on`` section to the source adapter service in the docker compose file:

.. code-block:: yaml

    depends_on:
      module:
        condition: service_healthy
