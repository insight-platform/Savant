.. _source_stall_detection:

Source Stall Detection
----------------------

A live source can stop delivering frames while its connection stays up: a camera that keeps the RTSP session alive but sends nothing, or a stream that degrades to a trickle. ``FFMPEG_TIMEOUT_MS`` covers the case where the source never connects, but a connected-and-silent feed keeps the adapter running forever with nothing flowing through it.

The RTSP source adapter can detect this in-process. It inserts a ``stall_detector`` element that passes buffers through unchanged, watches how long ago the last buffer arrived, and reports the result in a heartbeat file that the container health check probes. Optionally it also terminates the adapter so that the container supervisor restarts it.

This is the in-process complement to the :doc:`16_pipeline_watchdog`. The watchdog is a separate service that monitors the buffer adapter in front of the pipeline over HTTP and restarts containers through the Docker API; the stall detector runs inside the source adapter, needs no webserver, and recovers only itself.

Enabling
^^^^^^^^

Detection is enabled when either ``STALL_ACTION`` is set to something other than ``none`` or ``PIPELINE_HEALTH_FILEPATH`` is set, so observability and recovery are independent:

.. code-block:: bash

    docker run --rm -it --name source-rtsp-test \
        --entrypoint /opt/savant/adapters/gst/sources/rtsp.sh \
        -e ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc \
        -e SOURCE_ID=test \
        -e RTSP_URI=rtsp://192.168.1.1 \
        -e STALL_ACTION=message \
        -e STALL_MAX_IDLE_SECONDS=30 \
        -v /tmp/zmq-sockets:/tmp/zmq-sockets \
        ghcr.io/insight-platform/savant-adapters-gstreamer:latest

The ``STALL_*`` environment variables are documented with the other :ref:`RTSP source adapter parameters <ffmpeg_rtsp_source_adapter>`.

**Live sources only.** The detector treats silence as a fault, so it must not be used with a source that is legitimately quiet. A file that has been read to the end is quiet by design, which rules out the media files, video loop and multi-stream adapters permanently.

The FFmpeg source and and the GigE camera adapter adapter both are also eligible, neither is wired yet.

Statuses
^^^^^^^^

Every evaluation produces one of five statuses, written as the first line of the heartbeat file:

.. list-table::
    :header-rows: 1

    * - Status
      - Meaning
      - Action

    * - ``starting``
      - The warmup period has not finished, so no judgement is made.
      - none

    * - ``running``
      - Buffers are arriving within the configured limits.
      - none

    * - ``blocked``
      - A downstream push has been outstanding longer than ``STALL_MAX_IDLE_SECONDS``. The source is fine; the streaming thread is parked inside the downstream push.
      - **none, ever**

    * - ``ended``
      - The stream ended with ``EOS``, so there is nothing left to judge. The adapter normally exits on ``EOS``; the status exists so that one which does not exit reads ``unhealthy`` instead of looking like a stream that is still starting up.
      - none

    * - ``stalled``
      - The source went quiet while nothing downstream was holding the streaming thread.
      - ``STALL_ACTION``

The ``blocked`` status is the reason the detector is worth having. A plain frame counter cannot tell upstream starvation from downstream backpressure: when the ZMQ consumer stops reading, or the downstream module restarts, ``zeromq_sink`` blocks and the adapter sees zero frames per second, which looks exactly like a dead camera. Restarting the adapter then tears down a perfectly good RTSP session while the actual fault is elsewhere. The detector times each downstream push, so it reports ``blocked``, logs a warning, and **never** fires an action. Only a genuinely quiet source is ``stalled``.

Detection criteria
^^^^^^^^^^^^^^^^^^

A stream is ``stalled`` when either criterion trips:

- **Idle timeout.** No buffer has arrived for longer than ``STALL_MAX_IDLE_SECONDS``. This is the primary criterion and is always active.
- **Rate floor.** The frame rate over the last ``STALL_WINDOW_SECONDS`` is below ``STALL_MIN_FPS``. This catches a stream that degrades to a trickle rather than going silent. It is **disabled by default**: ``STALL_MIN_FPS=0`` means the idle timeout is the only criterion. The rate arm stays quiet until the collected samples cover a full window, so it cannot fire on a partially filled window.

Neither criterion is evaluated during ``STALL_WARMUP``, counted from the first time the pipeline reaches ``PLAYING``. Later ``PAUSED`` / ``PLAYING`` transitions do not extend the warmup.

Actions
^^^^^^^

``STALL_ACTION`` selects what happens on the first ``stalled`` verdict. The action fires once per stall, not once per evaluation: it re-arms when the stream recovers, so a second stall is reported again.

``none``
    Only the heartbeat file records the stall. Use this for observability without recovery: the Docker health check will report the container ``unhealthy``, and nothing else changes.

``message``
    Logs a warning and posts a warning message on the GStreamer bus, in addition to the heartbeat file. The pipeline keeps running.

``fail``
    Everything ``message`` does, escalated to a bus error. The error ends the process with a non-zero exit status, which is what makes a container supervisor restart the adapter. ZMQ teardown still runs, but no ``EOS`` reaches the consumer; the sink's ``EOS_ON_START`` (on by default) announces the stream boundary on the next run instead. If the posted error does not end the process within ten seconds, the detector exits the process itself as a backstop — that path skips the ZMQ teardown, which is why it is the fallback and not the primary route.

**``fail`` is compose-only.** It needs something to restart the container, and both ``restart: unless-stopped`` and ``restart: on-failure`` work, since the error path exits non-zero:

.. code-block:: yaml

    services:
      source-rtsp:
        image: ghcr.io/insight-platform/savant-adapters-gstreamer:latest
        entrypoint: /opt/savant/adapters/gst/sources/rtsp.sh
        restart: unless-stopped
        environment:
          - ZMQ_ENDPOINT=dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc
          - SOURCE_ID=test
          - RTSP_URI=rtsp://192.168.1.1
          - STALL_ACTION=fail
          - STALL_MAX_IDLE_SECONDS=30

``scripts/run_source.py`` rejects ``--stall-action=fail`` for this reason: it runs ``docker run --rm`` with no restart policy, so terminating there would remove the container permanently.

The exit status does not distinguish a stall from any other pipeline error. The stall warning, the error text and the heartbeat file all name the cause, so use the logs for diagnosis and do not key anything off the exit code.

Health check
^^^^^^^^^^^^

The detector rewrites its heartbeat file every few seconds — at least every five, whatever ``STALL_CHECK_INTERVAL`` says, so a fresh modification time proves the detector is alive and the body carries the current status and measurements:

.. code-block::

    running
    fps=24.97
    frames=124850
    seconds-since-last-frame=0.04
    probe=test
    pid=42

The GStreamer adapter image ships a ``HEALTHCHECK`` that probes this file with ``adapters/shared/pipeline_healthcheck.sh``. The probe fails closed: the container is healthy only if the file is fresh and the first line is exactly ``running`` or ``starting``. Anything else — ``blocked``, ``ended``, ``stalled``, a stale modification time, an empty or truncated file, a malformed ``PIPELINE_HEALTH_MAX_AGE`` — is ``unhealthy``. A stale file is its own signal: it catches a detector thread that died or an element that never started.

If the file does not exist, the probe passes. Its absence at a fixed path is how the probe knows the feature is not enabled in that container, which is why every other adapter built from the same image — every sink adapter included — stays healthy.

Note that Docker's ``unhealthy`` state restarts nothing by itself; it only gates ``depends_on: condition: service_healthy`` at startup and surfaces in ``docker ps``. The health check is the observability half of this feature, ``STALL_ACTION=fail`` the recovery half.

The heartbeat file is meant for one detector each. Two adapters sharing a path are last-writer-wins, where a healthy instance masks a stalled one — the ``pid`` line in the body is there to make that visible. Give each adapter its own ``PIPELINE_HEALTH_FILEPATH`` if they share a volume.
