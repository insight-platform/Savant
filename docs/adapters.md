# Adapters Manual

Savant provides several source and sink adapters. Adapters use an open protocol 
based on Avro and ZeroMQ. Everyone can develop their own adapter or modify one of the provided. 
We welcome contributions to adapters functionality. You can find more information 
in the [documentation](https://insight-platform.github.io/Savant/main_concepts/adapters.html).

Adapters run in prepared docker containers. The repository contains scripts that allow you to launch the container more conveniently. Below are descriptions of all adapters provided in the repository and examples of how to run them.

## Source adapters

### Image file source adapter 

Image file source adapter reads `image/jpeg` and `image/png` files from `LOCATION`, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Image file source adapter parameters are set trough environment variables:

- `SOURCE_ID` set unique identifier for the source adapter. This option is required.
- `FRAMERATE` is the desired framerate for the video stream formed from the input image files.
- `SORT_BY_TIME` is a flag that indicates whether files from `LOCATION` should be sorted by modification time (ascending order).
  By default, it is **False**  and files are sorted lexicographically.
- `READ_METADATA` is a flag that indicates attempt to read the metadata of objects from the JSON file
  that has the identical name as the source file with json extension, and then send it to the module with frame.
  Default is **False**.
- `OUT_ENDPOINT` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- `OUT_TYPE` set adapter output ZeroMQ socket type. Default is DEALER.
- `OUT_BIND` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- `SYNC` send frames from source synchronously (i.e. with the frame rate set via the FRAMERATE parameter). Default is False.
- `FPS_PERIOD_FRAMES` set number of frames between FPS reports. Default is 1000.
- `FPS_PERIOD_SECONDS` set number of seconds between FPS reports. Default is None.
- `FPS_OUTPUT` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example

``bash
    docker run --rm -it --name source-pictures-files-test \
    --entrypoint /opt/app/adapters/gst/sources/media_files.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e LOCATION=/path/to/images \
    -e FILE_TYPE=picture \
    -e SORT_BY_TIME=False \
    -e READ_METADATA=False \
    -v /path/to/images:/path/to/images:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
``

The same adapter can be run using a script
``bash
    python3 scripts/run_source.py pictures --source-id test /path/to/images
``

### Video file source adapter

Video file source adapter reads `video/*` files from `LOCATION`, which can be:

- Local path to a single file
- Local path to a directory with one or more files (not necessarily with the same encoding)
- HTTP URL to a single file

Video file source adapter parameters are set trough environment variables in the docker run command:

- `SOURCE_ID` set unique identifier for the source adapter. This option is required.
- `SORT_BY_TIME` is a flag that indicates whether files from `LOCATION` should be sorted by modification time (ascending order).
  By default, it is **False**  and files are sorted lexicographically.
- `READ_METADATA` is a flag that indicates attempt to read the metadata of objects from the JSON file
  that has the identical name as the source file with json extension, and then send it to the module with frame.
  Default is **False**.
- `OUT_ENDPOINT` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- `OUT_TYPE` set Adapter output ZeroMQ socket type. Default is DEALER.
- `OUT_BIND` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- `SYNC` set send frames from source synchronously (i.e. at the source file rate). Default is False.
- `FPS_PERIOD_FRAMES` set number of frames between FPS reports. Default is 1000.
- `FPS_PERIOD_SECONDS` set number of seconds between FPS reports. Default is None.
- `FPS_OUTPUT` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example

``bash
    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/media_files.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e LOCATION=/path/to/data/test.mp4 \
    -e FILE_TYPE=video \
    -e SORT_BY_TIME=False \
    -e READ_METADATA=False \
    -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
``

The same adapter can be run using a script
``bash
    python3 scripts/run_source.py videos --source-id test /path/to/data/test.mp4
``

*Note: Resulting video stream framerate is fixed to be equal to the framerate of the first encountered video file, possibly overriding the framerate of the rest of input.

### RTSP source adapter

RTSP source adapter reads RTSP stream from `RTSP_URI`.

RTSP source adapter parameters are set as environment variables in the docker run command:

- `RTSP_URI` specifies the RTSP URI for rtsp_source adapter. This option is required.
- `SOURCE_ID` set unique identifier for the source adapter. This option is required.
- `OUT_ENDPOINT` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- `OUT_TYPE` set adapter output ZeroMQ socket type. Default is DEALER.
- `OUT_BIND` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- `FPS_PERIOD_FRAMES` set number of frames between FPS reports. Default is 1000.
- `FPS_PERIOD_SECONDS` set number of seconds between FPS reports. Default is None.
- `FPS_OUTPUT` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example
```bash
    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/rtsp.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e RTSP_URI=rtsp://192.168.1.1 \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
```

The same adapter can be run using a script
```bash
    python3 scripts/run_source.py rtsp --source-id test rtsp://192.168.1.1
```

### Usb-cam source adapter

Usb-cam source adapter capture video from a v4l2 device specified in `DEVICE` parameter.

Usb-cam source adapter parameters are set as environment variables in the docker run command:

- `DEVICE` is the device file of the USB camera (default value is /dev/video0).
- `FRAMERATE` is the desired framerate for the video stream formed from the captured video.
  Note that if input video framerate is not in accordance with `FRAMERATE` parameter value, results may be unexpected.
- `SOURCE_ID` set unique identifier for the source adapter. This option is required.
- `OUT_ENDPOINT` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- `OUT_TYPE` set Adapter output ZeroMQ socket type. Default is DEALER.
- `OUT_BIND` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- `FPS_PERIOD_FRAMES` set number of frames between FPS reports. Default is 1000.
- `FPS_PERIOD_SECONDS` set number of seconds between FPS reports. Default is None.
- `FPS_OUTPUT` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example
```bash
    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/rtsp.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e DEVICE=/dev/video1 \
    -e FRAMERATE=30/1 \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
```

The same adapter can be run using a script
```bash
    python3 scripts/run_source.py usb-cam --source-id test --framerate 30/1 /dev/video1
```

### GigE source adapter

It is designed to take video streams from GigE cameras.

Usb-cam source adapter parameters are set as environment variables in the docker run command:

- `WIDTH` the width of the video stream, in pixels
- `HEIGHT` the height of the video stream, in pixels
- `FRAMERATE` the framerate of the video stream, in frames per second
- `INPUT_CAPS` the format of the video stream, in GStreamer caps format (e.g. "video/x-raw,format=RGB")
- `PACKET_SIZE` the packet size for GigEVision cameras, in bytes
- `AUTO_PACKET_SIZE` whether to negotiate the packet size automatically for GigEVision cameras
- `EXPOSURE` the exposure time for the camera, in microseconds
- `EXPOSURE_AUTO` the auto exposure mode for the camera, one of "off", "once", or "on"
- `GAIN` the gain for the camera, in decibels
- `GAIN_AUTO` the auto gain mode for the camera, one of "off", "once", or "on"
- `FEATURES` additional configuration parameters for the camera, as a space-separated list of feature assignations
- `HOST_NETWORK` whether to use the host network
- `CAMERA_NAME` the name of the camera, in the format specified in the command description. This parameter is optional, and if it is
- `SOURCE_ID` set unique identifier for the source adapter. This option is required.
- `OUT_ENDPOINT` set adapter output (should be equal to module input) ZeroMQ socket endpoint.
  Default is ipc:///tmp/zmq-sockets/input-video.ipc.
- `OUT_TYPE` set Adapter output ZeroMQ socket type. Default is DEALER.
- `OUT_BIND` set adapter output ZeroMQ socket bind/connect mode (bind if True). Default is False.
- `FPS_PERIOD_FRAMES` set number of frames between FPS reports. Default is 1000.
- `FPS_PERIOD_SECONDS` set number of seconds between FPS reports. Default is None.
- `FPS_OUTPUT` set path to the file where the FPS reports will be written. Default is 'stdout'.

Example
```bash
    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/gige_cam.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e CAMERA_NAME=test-camera\
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
```

The same adapter can be run using an auxiliary python script
```bash
    python3 scripts/run_source.py gige --source-id test test-camera
```


## Sink adapters

These adapters should help the user with the most basic and widespread output data formats.

### Metadata Sink Adapter

Meta-json sink adapter writes received messages as newline-delimited JSON streaming file to a `LOCATION`, which can be:

- Local path to a single file
- Local path with substitution patterns:
  - `%source_id` will insert `SOURCE_ID` value into resulting filename
  - `%src_filename` will insert source filename into resulting filename.

Meta-json sink adapter parameters are set as environment variables in the docker run command:

- `CHUNK_SIZE` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate files with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new file starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- `SKIP_FRAMES_WITHOUT_OBJECTS` is a flag that indicates whether frames with 0 objects should be skipped in output. Default value is False.
- `SOURCE_ID` Optional filter to receive frames with a specific source ID only.
- `SOURCE_ID_PREFIX` Optional filter to receive frames with a source ID prefix only.
- `IN-ENDPOINT` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- `IN_TYPE` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- `IN_BIND` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.

Example
```bash
    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/python/sinks/metadata_json.py \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-py:latest
```

The same adapter can be run using a script
```bash
    python3 scripts/run_sink.py meta-json /path/to/output/%source_id-%src_filename
```

### Image file sink adapter

Image file sink adapter writes received messages as separate image files and json files to a directory,
specified in `DIR_LOCATION` parameter.

Image file sink adapter parameters are set as environment variables in the docker run command:

- `DIR_LOCATION` the location of the file to write metadata to. Can be a plain location or a pattern.
    Allowed substitution parameters are %source_id and %src_filename.
- `CHUNK_SIZE` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate folders with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new folder starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- `SKIP_FRAMES_WITHOUT_OBJECTS` is a flag that indicates whether frames with 0 objects should be skipped in output. Default value is False.
- `SOURCE_ID` Optional filter to receive frames with a specific source ID only.
- `SOURCE_ID_PREFIX` Optional filter to receive frames with a source ID prefix only.
- `IN-ENDPOINT` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- `IN_TYPE` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- `IN_BIND` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.

Example
```bash
    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/python/sinks/image_files.py \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-py:latest
```

The same adapter can be run using a script

```bash
    python3 scripts/run_sink.py image-files  /path/to/output/%source_id-%src_filename
```

### Video file sink adapter

Video file sink adapter writes received messages as video files to a directory, specified in `DIR_LOCATION` parameter.

Video file sink adapter parameters are set as environment variables in the docker run command:

- `DIR_LOCATION` the location of the file to write metadata to. Can be a plain location or a pattern.
    Allowed substitution parameters are %source_id and %src_filename.
- `CHUNK_SIZE` Chunk size in frames. The whole stream of incoming frames with meta-data is split into separate
    parts and written to separate files with consecutive numbering. If a message about the end of the data stream
    (generated by the module or source-adapter) comes, the recording to a new file starts,
    even if there are less than the specified number of frames. Default value is 10000. A value of 0 disables chunking
    within one continuous stream of frames by source_id.
- `SOURCE_ID` Optional filter to receive frames with a specific source ID only.
- `SOURCE_ID_PREFIX` Optional filter to receive frames with a source ID prefix only.
- `IN-ENDPOINT` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- `IN_TYPE` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- `IN_BIND` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.

Example
```bash
    docker run --rm -it --name sink-meta-json \
    --entrypoint /opt/app/adapters/gst/sinks/video_files.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DIR_LOCATION=/path/to/output/%source_id-%src_filename \
    -e SKIP_FRAMES_WITHOUT_OBJECTS=False \
    -e CHUNK_SIZE=0 \
    -v /path/to/output/:/path/to/output/ \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    savant-adapters-gstreamer:latest
```

The same adapter can be run using a script
```bash
    python3 scripts/run_sink.py video-files /path/to/output/%source_id-%src_filename
```

### Display sink adapter

Display sink adapter sends received frames to a window output.

Display sink adapter parameters are set as environment variables in the docker run command:

- `CLOSING-DELAY` he delay in seconds before closing the window after the video stream has finished.
- `SYNC` whether to show the frames on the sink synchronously with the source (i.e., at the source file rate).
- `SOURCE_ID` Optional filter to receive frames with a specific source ID only.
- `SOURCE_ID_PREFIX` Optional filter to receive frames with a source ID prefix only.
- `IN-ENDPOINT` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- `IN_TYPE` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- `IN_BIND` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.

Example
```bash
    xhost +
    docker run --rm -it --name sink-display \
    --entrypoint /opt/app/adapters/ds/sinks/display.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e DISPLAY \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -e CLOSING_DELAY=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/.docker.xauth:/tmp/.docker.xauth \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    --gpus=all savant-adapters-deepstream:latest
```

The same adapter can be run using a script
```bash
    python3 scripts/run_sink.py display
```

### Always-On RTSP sink adapter

Always-on RTSP sink adapter sends video stream from a specific source to an RTSP server.

Always-on RTSP sink adapter parameters are set as environment variables in the docker run command:

- `RTSP_URI`: The URI of the RTSP server, this parameter is required.
- `STUB_FILE_LOCATION` The location of the stub image file. Image file must be in JPEG format, this parameter is required.
  The stub image file is shown when there is no input stream.
- `MAX_DELAY_MS` Maximum delay for the last frame in milliseconds, default value is 1000.
- `TRANSFER_MODE` Transfer mode. One of: "scale-to-fit", "crop-to-fit", default value is "scale-to-fit".
- `PROTOCOLS` Allowed lower transport protocols, e.g. "tcp+udp-mcast+udp", default value is "tcp".
- `LATENCY_MS` Amount of ms to buffer RTSP stream, default value is 100.
- `KEEP_ALIVE` Send RTSP keep alive packets, disable for old incompatible server, default value is True.
- `PROFILE` H264 encoding profile. One of: "Baseline", "Main", "High", default value is "High".
- `BITRATE` H264 encoding bitrate, default value is 4000000.
- `FRAMERATE` Frame rate of the output stream, default value is "30/1".
- `METADATA_OUTPUT` Where to dump metadata (stdout or logger).
- `SYNC` Show frames on sink synchronously (i.e. at the source file rate). Inbound stream is not stable with this flag, try to avoid it. Default value is False.
- `SOURCE_ID` Optional filter to receive frames with a specific source ID only.
- `SOURCE_ID_PREFIX` Optional filter to receive frames with a source ID prefix only.
- `IN-ENDPOINT` ZeroMQ socket endpoint for the adapter's input, i.e., the module output. Default value is 'ipc:///tmp/zmq-sockets/output-video.ipc'.
- `IN_TYPE` ZeroMQ socket type for the adapter's input. Default value is 'SUB'.
- `IN_BIND` Specifies whether the adapter's input should be bound or connected to the specified endpoint.
    If True, the input is bound; otherwise, it's connected. Default value is False.


Example
```bash
    docker run --rm -it --name sink-always-on-rtsp \
    --entrypoint python \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/output-video.ipc \
    -e ZMQ_TYPE=SUB \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e STUB_FILE_LOCATION=/path/to/stub_file/test.jpg \
    -e MAX_DELAY_MS=1000 \
    -e TRANSFER_MODE=scale-to-fit \
    -e RTSP_URI=rtsp://192.168.1.1 \
    -e RTSP_PROTOCOLS=tcp \
    -e RTSP_LATENCY_MS=100 \
    -e RTSP_KEEP_ALIVE=True \
    -e ENCODER_PROFILE=High \
    -e ENCODER_BITRATE=4000000 \
    -e FRAMERATE=30/1 \
    -v /path/to/stub_file/test.jpg:/path/to/stub_file/test.jpg:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    --gpus=all \
    savant-adapters-deepstream:latest \
    -m adapters.ds.sinks.always_on_rtsp
```

The same adapter can be run using a script
```bash
    python scripts/run_sink.py always-on-rtsp --source-id test --stub-file-location ../vlf-pipelines/data/bottle_defect_detector/test.jpg  rtsp://192.168.1.1
```
