Running a Savant module
=======================

Any Savant module is run in a docker container, either the base Savant container or a specific module container with additional dependencies (see :ref:`main_concepts/module:module overview`).

By default module takes input from a source adapter container and outputs to a sink adapter container, relying on ZMQ for connection.

Connecting containers through ZMQ sources/sinks requires specifying the following environment variables inside containers:

- Source adapter container:

    - ``SOURCE_ID`` is a required env var that specifies an identifier for this source
    - ``ZMQ_ENDPOINT`` is a required env var that specifies the ZMQ socket endpoint for output (module input)
    - ``ZMQ_TYPE`` env var specifies the ZMQ socket type, ``REQ`` by default
    - ``ZMQ_BIND`` env var specifies whether the output ZMQ socket type should be bound or connected to, ``False`` by default
    - ``LOCATION`` is a required env var for ``pictures`` and ``videos`` sources that specifies the directory with the source data
    - ``RTSP_URI`` is a required env var for ``rtsp`` source that specifies the rtsp uri with the source data
    - ``DEVICE`` is a required env var for ``usb_cam`` source that specifies the device with the source data

- Module container:

    - ``ZMQ_SRC_ENDPOINT`` is a required env var that specifies the input ZMQ socket endpoint, corresponds to source adapter ``ZMQ_ENDPOINT``
    - ``ZMQ_SRC_TYPE`` env var specifies the input ZMQ socket type, ``REP`` by default
    - ``ZMQ_SRC_BIND`` env var specifies whether the input ZMQ socket type should be bound or connected to, ``True`` by default
    - ``ZMQ_SINK_ENDPOINT`` is a required env var that specifies the output ZMQ socket endpoint, corresponds to sink adapter ``ZMQ_ENDPOINT``
    - ``ZMQ_SINK_TYPE`` env var specifies the output ZMQ socket type, ``PUB`` by default
    - ``ZMQ_SINK_BIND`` env var specifies whether the output ZMQ socket type should be bound or connected to, ``True`` by default
    - ``OUTPUT_FRAME`` env var specifies whether to include frames in module output, and not just metadata; it is required for ``image_files`` and ``video_files`` sinks to produce valid output. Set this var to a JSON string with attribute ``codec`` equal to one of ``h264``, ``hevc``, ``png``, ``jpeg`` or ``raw-rgba`` and optional ``encoder_params``. For example

        .. code-block:: JSON

            {"codec": "jpeg"}
            {"codec": "h264", "encoder_params": {"bitrate": 4000000}}
            {"codec": "hevc"}

- Sink adapter container:

    - ``ZMQ_ENDPOINT`` is a required env var that specifies the ZMQ socket endpoint for input (module output)
    - ``ZMQ_TYPE`` env var specifies the ZMQ socket type, ``SUB`` by default
    - ``ZMQ_BIND`` env var specifies whether the output ZMQ socket type should be bound or connected to, ``False`` by default
    - ``LOCATION`` is a required env var for ``json`` sink that specifies the output json files or files
    - ``DIR_LOCATION`` is a required env var for ``image_files`` and ``video_files`` sink that specifies the directory for output files inside the container

.. note::

    Description of ZMQ related parameters is available in ZMQ documentation:

    - ``ENDPOINT`` is described on `zmq-connect <http://api.zeromq.org/2-1:zmq-connect>`_ page
    - ``TYPE`` is described on `zmq-socket <http://api.zeromq.org/2-1:zmq-socket>`_ page
    - ``BIND`` is described on `Socket API <https://zeromq.org/socket-api/#bind-vs-connect>`_ page

.. note::

    Module and sink adapters are connected through PUB-SUB pattern by default. Since the publisher does not wait for subscribers to connect, take care to start a module or any source adapters only after the sink adapter becomes ready to receive incoming messages.

Additionally, ZMQ connection might require the following ``docker run`` parameters:

- In case ipc transport is used for ZMQ connection, ipc file should be shared between containers with ``-v`` mount flag.
- In case tcp transport is used for ZMQ connection, appropriate ports must be published, or host network mode must be used.

Module container mounts
-----------------------

Since running a module involves accessing module configuration file and model files, these files must be made available inside the module container, for example, using docker bind mounts.

The paths for mounts inside the container are determined by the module configuration:

- ``parameters.model_path`` defines the directory with the model files. It can be set through ``MODEL_PATH`` env var and equals ``/models`` by default.
- ``parameters.download_path`` defines the directory used as download target for remote files (for example, model files on remote storage). It can be set through ``DOWNLOAD_PATH`` env var and equals ``/downloads`` by default.

.. note::

    Module expects files related to each pipeline element to be placed in their own subdirectory in the model and download directories. Meaning that if your pipeline contains an element with the name ``detector``, module will look for detector model files inside the ``/opt/app/models/detector/`` directory.

Module container mounts are also useful as a local cache for runtime module artifacts, for example, model TRT engine files for nvinfer elements, or checksum files for model elements with configured remote files.

Example run commands
--------------------

For convenience there are python run scripts that generate appropriate docker run commands.
These scripts can be found in the ``scripts`` directory in the :repo-link:`Savant` repository.

Use helper script to run

- video source adapter container

.. code-block:: shell

    python scripts/run_source.py videos <path_to_input_video_dir> --source-id my-src-id

- module container, peoplenet sample

.. code-block:: shell

    python scripts/run_module.py samples/peoplenet_detector/module.yml

- video sink adapter container

.. code-block:: shell

    python scripts/run_sink.py image-files <path_to_output_image_dir>
