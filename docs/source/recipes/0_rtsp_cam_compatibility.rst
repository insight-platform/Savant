RTSP Cam Compatibility
======================

There are many reasons why an RTSP camera might work incorrectly, starting from incorrect protocol implementation and ending with network issues and latency. To address the possible problems, use information from the manual.

Network Test
------------

A poor network connection is often a reason for RTSP problems. RTSP is a protocol designed to deliver video to the viewer in real-time, with the lowest latency. Thus, when the underlying network doesn't perform predictably, it suffers dramatically.

You can run simple test to find that network is able to deliver traffic from the cam properly.

The time, traffic travels from the device to the receiver, is not a real problem. The variation and loss in time are real problems. To estimate it, use the ping command:

.. code-block:: bash

    ping -qc50 -s 1460 10.10.12.1
    PING 10.10.12.1 (10.10.12.1) 1460(1488) bytes of data.

    --- 10.10.12.1 ping statistics ---
    50 packets transmitted, 50 received, 0% packet loss, time 49075ms
    rtt min/avg/max/mdev = 1.398/3.695/28.755/3.926 ms

In the above-displayed listing, you can see that the loss is 0% which is good. Jitter is pretty stable, never exceeding a reasonable value of 20-30 ms, which should be normal when serving a video on 30 FPS. If you experience large **mdev**, it may cause problems.

FFplay Playback Evaluation
---------------------------

Use the ``ffplay`` command to test video visually:

.. code-block:: bash

    ffplay 'rtsp://your.server/stream'

If you experience **periodic** packet losses like:

.. code-block::

    [rtsp @ 0x7f7ba8000cc0] max delay reached. need to consume packet
    [rtsp @ 0x7f7ba8000cc0] RTP: missed 5 packets
    [rtsp @ 0x7f7ba8000cc0] max delay reached. need to consume packet
    [rtsp @ 0x7f7ba8000cc0] RTP: missed 5 packets
    [rtsp @ 0x7f7ba8000cc0] max delay reached. need to consume packet
    [rtsp @ 0x7f7ba8000cc0] RTP: missed 3 packets

or

.. code-block::

    [h264 @ 0x7f7ba8c315c0] concealing 11875 DC, 11875 AC, 11875 MV errors in I frame

or see artifacts on the screen, you need to fix your network connection and camera properties. It is unlikely the stream will be processed normally.

The Savant RTSP Source Adapter Playback Evaluation
--------------------------------------------------

Visit `RTSP Compatibility Test Sample <https://github.com/insight-platform/Savant/tree/develop/samples/rtsp_cam_compatibility_test>`__ and launch it for your cam to ensure that there are no problems with the decoding chain, including NVDEC.
