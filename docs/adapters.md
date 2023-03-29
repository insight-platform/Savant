# Adapters Manual

The adapter is an independent process that either reads (source adapter) or writes (sink adapter) data from/to some location, thus decoupling input/output operations from the main processing.

Savant provides several ready-to-use source and sink adapters. All adapters are implemented as Docker images, and Python helper scripts have been developed to simplify the process of running and using source and sink adapters. However, scripts are optional so that you can launch adapters directly as docker commands.

The framework can communicate with both source and sink adapters via a protocol implemented with ZeroMQ sockets Apache AVRO. Below you will find descriptions for all adapters provided and examples of how to run them.

## Source Adapters

### The Image File Source Adapter

The image file source adapter reads `image/jpeg` or `image/png` files from `LOCATION`, which can be:

- a local path to a single file;
- a local path to a directory with files (not necessarily in the same encoding);
- HTTP URL to a single file.

The adapter is useful for development purposes. It also can be used to process image streams efficiently in production.

The adapter parameters are set with environment variables:

- `LOCATION` - image file(s) location or URL;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `FRAMERATE` - desired framerate for the video stream formed from the input image files (if sync mode is chosen);
- `SORT_BY_TIME` - flag indicates whether the files from `LOCATION` are sorted by modification time (ascending order); by default, it is `False` and the files are sorted lexicographically.
- `READ_METADATA` - flag indicates the need to read and send the object's metadata from a JSON file that has the identical name as the source file; default is `False`.
- `OUT_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; default is `ipc:///tmp/zmq-sockets/input-video.ipc`.
- `OUT_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `OUT_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`.
- `SYNC` - flag indicates the need to send frames from source synchronously (i.e. with the frame rate set via the `FRAMERATE` parameter); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; default is `stdout`.

Example:

```bash
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
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py pictures --source-id=test /path/to/images
```

## The Video File Source Adapter

The video file source adapter reads `video/*` files from `LOCATION`, which can be:

- a local path to a single file;
- a local path to a directory with one or more files;
- HTTP URL to a single file;

The adapter parameters are set with environment variables:

- `LOCATION` - video file(s) location or URL;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `SORT_BY_TIME` - flag indicates whether files from `LOCATION` are sorted by modification time (ascending order); by default, it is `False` and files are sorted lexicographically;
- `READ_METADATA` - flag indicates the need to read the object's metadata from a JSON file that has the identical name as the source file; default is `False`;
- `OUT_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; default is `ipc:///tmp/zmq-sockets/input-video.ipc`;
- `OUT_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `OUT_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `SYNC` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; default is `stdout`.

## Example

```bash
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
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py videos --source-id=test /path/to/data/test.mp4
```

**Note**: Resulting video stream framerate will be equal to the framerate of the first encountered video file, possibly overriding the framerate of the rest of input files.

### The RTSP Source Adapter

The RTSP source adapter reads RTSP stream from specified `RTSP_URI`.

The adapter parameters are set with environment variables:
- `RTSP_URI` - RTSP URI of the stream; this option is required;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `OUT_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; default is `ipc:///tmp/zmq-sockets/input-video.ipc`;
- `OUT_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `OUT_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; default is `stdout`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py rtsp --source-id=test rtsp://192.168.1.1
```

### The USB Cam Source Adapter

The USB cam source adapter captures video from a V4L2 device specified in `DEVICE` parameter.

The adapter parameters are set with environment variables:
- `DEVICE` - USB camera device; default value is `/dev/video0`;
- `FRAMERATE` - desired framerate for the video stream captured from the device; note that if the input device does not support specified video framerate, results may be unexpected;
- `SOURCE_ID` - unique identifier for the source adapter; this option is required;
- `OUT_ENDPOINT` - adapter output (should be equal to module input) ZeroMQ socket endpoint; default is `ipc:///tmp/zmq-sockets/input-video.ipc`;
- `OUT_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `OUT_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - Number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; Default is `stdout`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py usb-cam --source-id=test --framerate=30/1 /dev/video1
```

### The GigE Source Adapter

The adapter is designed to take video streams from GigE cameras.

The adapter parameters are set with environment variables:

- `WIDTH` - the width of the video frame, in pixels;
- `HEIGHT` - the height of the video frame, in pixels;
- `FRAMERATE` - the framerate of the video stream, in frames per second;
- `INPUT_CAPS` - the format of the video stream, in GStreamer caps format (e.g. "video/x-raw,format=RGB");
- `PACKET_SIZE` - the packet size for GigEVision cameras, in bytes;
- `AUTO_PACKET_SIZE` - whether to negotiate the packet size automatically for GigEVision cameras;
- `EXPOSURE` - the exposure time for the camera, in microseconds;
- `EXPOSURE_AUTO` - the auto exposure mode for the camera, one of `off`, `once`, or `on`;
- `GAIN` - the gain for the camera, in decibels;
- `GAIN_AUTO` - the auto gain mode for the camera, one of `off`, `once`, or `on`;
- `FEATURES` - additional configuration parameters for the camera, as a space-separated list of features;
- `HOST_NETWORK` - host network to use;
- `CAMERA_NAME` - name of the camera, in the format specified in the command description; 
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `OUT_ENDPOINT` - adapter output (should be equal to module input) ZeroMQ socket endpoint; default is `ipc:///tmp/zmq-sockets/input-video.ipc`;
- `OUT_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `OUT_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - Number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; Default is `stdout`.

Example:

```bash
    docker run --rm -it --name source-video-files-test \
    --entrypoint /opt/app/adapters/gst/sources/gige_cam.sh \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e CAMERA_NAME=test-camera \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py gige --source-id=test test-camera
```


## Sink Adapters

These adapters should help the user with the most basic and widespread output data formats.

### The JSON Meta Sink Adapter

The JSON meta sink adapter writes received messages as newline-delimited JSON streaming file to a `LOCATION`, which can be:

- Local path to a single file;
- Local path with substitution patterns:
  - `%source_id` will insert `SOURCE_ID` value into resulting filename;
  - `%src_filename` will insert source filename into resulting filename.

The adapter parameters are set with environment variables:

- `LOCATION` - output metadata file path;
- `CHUNK_SIZE` - chunk size in frames; the whole stream of incoming frames with meta-data is split into separate parts and written to separate files with consecutive numbering; the default value is `10000`, a value of `0` disables chunking resulting to a continuous stream of frames by `source_id`;
- `SKIP_FRAMES_WITHOUT_OBJECTS` - flag that indicates whether frames with `0` objects should be ignored in output; the default value is `False`;
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `IN-ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; the default value is `ipc:///tmp/zmq-sockets/output-video.ipc`;
- `IN_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `IN_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-py:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py meta-json /path/to/output/%source_id-%src_filename
```

### The Image File Sink Adapter

The image file sink adapter extends the JSON meta adapter by writing image files along with meta JSON files to a specified in `DIR_LOCATION` parameter directory.

The adapter parameters are set trough environment variables:

- `DIR_LOCATION` - location to write files to; can be a plain location or a pattern; allowed substitution parameters are `%source_id` and `%src_filename`;
- `CHUNK_SIZE` - chunk size in frames; the whole stream of incoming frames with meta data is split into separate parts and written to separate folders with consecutive numbering; default value is `10000`. A value of `0` disables chunking within one continuous stream of frames by `source_id`;
- `SKIP_FRAMES_WITHOUT_OBJECTS` - flag that indicates whether frames with `0` objects should be ignored in output; the default value is `False`;
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `IN-ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; the default value is `ipc:///tmp/zmq-sockets/output-video.ipc`;
- `IN_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `IN_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-py:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py image-files /path/to/output/%source_id-%src_filename
```

### The Video File Sink Adapter

The video file sink adapter extends the JSON meta adapter by writing video files along with meta JSON files to a specified in `DIR_LOCATION` parameter directory.

The adapter parameters are set with environment variables:

- `DIR_LOCATION` - location to write video and metadata to; can be a plain location or a pattern; allowed substitution parameters are `%source_id` and `%src_filename`;
- `CHUNK_SIZE` - chunk size in frames; the whole stream of incoming frames with meta-data is split into separate parts and written to separate files with consecutive numbering; the default value is `10000`, a value of `0` disables chunking resulting to a continuous stream of frames by `source_id`;
- `SKIP_FRAMES_WITHOUT_OBJECTS` - flag that indicates whether frames with `0` objects should be ignored in output; the default value is `False`;
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `IN-ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; the default value is `ipc:///tmp/zmq-sockets/output-video.ipc`;
- `IN_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `IN_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py video-files /path/to/output/%source_id-%src_filename
```

### The Display Sink Adapter

The display sink adapter opens player windows where displays every processed stream:

The adapter parameters are set with environment variables:

- `CLOSING-DELAY` - delay in seconds before closing the window after the video stream has finished;
- `SYNC` - flag indicates whether to show the frames on the sink synchronously with the source (i.e., at the source file rate);
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `IN-ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; the default value is `ipc:///tmp/zmq-sockets/output-video.ipc`;
- `IN_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `IN_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

Example:

```bash
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
    --gpus=all \
    ghcr.io/insight-platform/savant-adapters-deepstream:0.2.0-6.2
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py display
```

### The Always-On RTSP Sink Adapter

The Always-on RTSP sink adapter casts the video stream from a specific source to an RTSP server.

The adapter parameters are set with environment variables:

- `RTSP_URI` - URI of the RTSP server, this parameter is required;
- `STUB_FILE_LOCATION` - location of the stub image file; the image file must be in JPEG format, this parameter is required; the stub image file is shown when there is no input data;
- `MAX_DELAY_MS` - maximum delay for the last frame in milliseconds, default value is `1000`; after the delay the stub image will be displayed;
- `TRANSFER_MODE` - transfer mode specification; one of: `scale-to-fit`, `crop-to-fit`; the default value is "scale-to-fit";
- `PROTOCOLS` - allowed lower transport protocols, e.g. `tcp+udp-mcast+udp`; the default value is `tcp`;
- `LATENCY_MS` - amount of ms to buffer for the RTSP stream, default value is `100`;
- `KEEP_ALIVE` - whether to send RTSP keep alive packets, disable for old incompatible server, default value is `True`;
- `PROFILE` - H264 encoding profile; one of: `Baseline`, `Main`, `High`; the default value is `High`;
- `BITRATE` - H264 encoding bitrate; the default value is `4000000`;
- `FRAMERATE` - frame rate for the output stream; the default value is `30/1`;
- `METADATA_OUTPUT` - where to dump metadata (stdout or logger);
- `SYNC` - flag indicates whether to show frames on sink synchronously (i.e. at the source rate); the streaming may be not stable with this flag, try to avoid it; the default value is `False`;
- `SOURCE_ID` - optional filter to receive frames with a specific source ID only;
- `IN-ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; the default value is `ipc:///tmp/zmq-sockets/output-video.ipc`;
- `IN_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `IN_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

Example:

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
    ghcr.io/insight-platform/savant-adapters-deepstream:0.2.0-6.2 \
    -m adapters.ds.sinks.always_on_rtsp
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py always-on-rtsp --source-id=test --stub-file-location=/path/to/stub_file/test.jpg rtsp://192.168.1.1
```
