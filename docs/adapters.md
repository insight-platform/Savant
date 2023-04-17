# Adapters Manual

The adapter is an independent process that either reads (source adapter) or writes (sink adapter) data from/to some location, thus decoupling input/output operations from the main processing. Adapters are implemented as Docker images, and Python helper scripts have been developed to simplify the process of running and using source and sink adapters. However, scripts are optional so that you can launch adapters directly as docker commands.

The framework can communicate with both source and sink adapters via a protocol implemented with ZeroMQ sockets and Apache AVRO. Below you will find descriptions for all adapters provided and examples of how to run them.

- [Socket Types](#socket-types)
  - [The Rule of Thumb](#the-rule-of-thumb)
  - [Typical Patterns](#typical-patterns)
    - [Data-Center Patterns](#data-center-patterns)
    - [Edge Patterns](#edge-patterns)
  - [DEALER/ROUTER](#dealer/router)
  - [REQ/REP](#req/rep)
  - [PUB/SUB Explanation](#pub/sub-explanation)
- [Source Adapters](#source-adapters)
  - [The Image File Source Adapter](#the-image-file-source-adapter)
  - [The Video File Source Adapter](#the-video-file-source-adapter)
  - [The Video Loop File Source Adapter](#the-video-loop-file-source-adapter)
  - [The RTSP Source Adapter](#the-rtsp-source-adapter)
  - [The USB Cam Source Adapter](#the-usb-cam-source-adapter)
  - [The GigE Source Adapter](#the-gige-source-adapter)
- [Sink Adapters](#sink-adapters)
  - [The JSON Meta Sink Adapter](#the-json-meta-sink-adapter)
  - [The Image File Sink Adapter](#the-image-file-sink-adapter)
  - [The Video File Sink Adapter](#the-video-file-sink-adapter)
  - [The Display Sink Adapter](#the-display-sink-adapter)
  - [The Always-On RTSP Sink Adapter](#the-always-on-rtsp-sink-adapter)


## Socket Types

Savant supports three kinds of sockets for communications with the framework:

- DEALER/ROUTER - safe asynchronous socket pair, supporting multiple publishers, single subscriber;
- REQ/REP - safe synchronous socket pair, supporting multiple publishers, single subscriber;
- PUB/SUB - unsafe socket pair, supporting single publisher, multiple subscribers.

You have to carefully decide what socket pair to use when building the pipeline based on your needs and socket features.

The sockets can be in either `bind` or `connect` modes. If the socket is configured as `bind` it listens the address, 
if it is configured as `connect` it connects to the address. 

There are two URL schemes supported:
- Unix domain sockets;
- TCP sockets.

Read more about ZeroMQ socket pairs on ZeroMQ [website](https://zeromq.org/socket-api/).

You usually want to use combinations, which are marked with Green color:

![image](https://user-images.githubusercontent.com/15047882/228468880-878857cc-2262-498c-a647-51e669a45b4b.png)

Socket type and bind/connect mode can be embedded into an endpoint (`[<socket_type>+(bind|connect):]<endpoint>`). E.g.

```bash
ZMQ_ENDPOINT="dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc"
# is equal to
ZMQ_ENDPOINT="ipc:///tmp/zmq-sockets/input-video.ipc"
ZMQ_TYPE="DEALER"
ZMQ_BIND="False"
```

```bash
ZMQ_ENDPOINT="pub+bind:tcp://1.1.1.1:3333"
# is equal to
ZMQ_ENDPOINT="tcp://1.1.1.1:3333"
ZMQ_TYPE="PUB"
ZMQ_BIND="True"
```

### The Rule of Thumb

1. Try to use the framework in **bind** mode, and adapters in **connect** mode first; change only if it does not work for you.
2. The part which delivers multiplexed stream usually has the **bind** type; the part which handles a single (non-multiplexed) stream usually has the **connect** type.
3. Use the `PUB/SUB` pair only when the pipeline or adapter is capable to handle the traffic in real-time.

### Typical Patterns

There are typical patterns widely used, try to start from them when designing pipelines.

- `D` - dealer;
- `R` - router;
- `P` - publisher (PUB);
- `S` - subscriber (SUB).

The pairs are explained after the patterns section in detail.

#### Data-Center Patterns

Data-center patterns are designed to reliably process video streams with increased latency in situations when the pipeline is overwhelmed with data. 0MQ socket pairs used in data-center patterns are `DEALER/ROUTER` (default recommended) or `REQ/REP`. These pairs implement a backpressure mechanism which causes the processing to be delayed when watermarks are reached.

![Savant socket pairs (10)](https://user-images.githubusercontent.com/15047882/228738612-43251848-c6fe-478d-98ae-fdca7192696d.png)

The first one is a typical scenario when the adapter reads multiplexed streams from an external queue system (like Kafka) and passes them to the framework instance. The framework, in turn, transfers analytics results (and video) to the adapter, which places the results into a database or another queue system.

The second is typical when adapters are used to aggregate data from multiple streams (e.g. RTSP cams) into the framework instance. The right side of the pipeline stays the same as in the previous case.

#### Edge Patterns

Edge patterns often aim to provide real-time operations for data sources with the lowest latency possible. To implement that, you may utilize the `PUB/SUB` socket pair because it drops the packets that the `SUB` part cannot process in a timely manner. This mechanism works absolutely great when used with streams delivering `MJPEG`, `RAW`, `JPEG`, `PNG`, and other independent video frames. Using the pattern with video-encoded streams is troublesome because drops cause video corruption.

![Savant socket pairs (11)](https://user-images.githubusercontent.com/15047882/228739197-fe5289b8-ff2e-47ea-95e8-39eea1adaeb2.png)

The first pattern may be used when neither adapters nor framework must be frozen because of the sink stalling. The second pattern is beneficial when the sink guarantees the processing, and you do not concern that it can be overwhelmed, causing the framework pipeline to stall too.

### DEALER/ROUTER

This is a recommended pair to use when you don't need to duplicate the messages to multiple subscribers or can implement such duplication programmatically. It is a reliable socket pair: the `DEALER` will block if the `ROUTER`'s queue is full.

**Source/CONNECT-to-Framework/BIND communication**. This is a typical scheme of communication.

![Savant socket pairs (4)](https://user-images.githubusercontent.com/15047882/228478831-af9032c7-50e2-4f6a-84d4-9583b609dd96.png)

**Framework/CONNECT-to-Sink/BIND communication**. This is a normal pattern, when you have the sink adapter communicating with the external system like Kafka and wish to send data from multiple framework instances.

![Savant socket pairs (5)](https://user-images.githubusercontent.com/15047882/228480218-b222776c-baa2-4342-8c1b-0133e256bd40.png)

**Source/BIND-to-Framework/CONNECT communication**. This is an exotic pattern. Although, It may do the job when you handle raw frames or isolated image streams and don't care about per-stream order. In this scheme, the source will distribute data berween connected frameworks according to LRU strategy, so it is impossible to use the scheme when you work with video.

![Savant socket pairs (6)](https://user-images.githubusercontent.com/15047882/228480906-b74ca06a-3f48-4a8b-b4cc-18bad4fc2565.png)

**Framework/BIND-to-Sink/CONNECT communication**. This is a valid pattern, when sinks communicating with an external system are slow or require multiple operations and the order of data appending is not critical.

![Savant socket pairs (7)](https://user-images.githubusercontent.com/15047882/228481469-c11c8d53-9244-4dfb-8071-c042197b716a.png)

### REQ/REP

The `REQ/REP` pair works the same way as the `DEALER/ROUTER` except that the `REQ` part receives replies from the REP part every time the REP part handles the message. It can be useful to modify the injecting pace on the `REQ` part. This is a generally recommended pair to use when you don't need multiple subscribers or can implement such duplication somehow. It is reliable socket pair: the `REQ` sends the next frame only when received the response previously sent from `REP`.

### PUB/SUB Explanation

The `PUB/SUB` is convenient to use when you need to handle the same data by multiple subscribers. Another use case for `PUB/SUB` is when you are processing the real-time data: when excessive elements are silently dropped if the pipeline or adapter is unable to handle the traffic burst.

**Source/BIND-to-Framework/CONNECT communication**. The source is initialized as a server (bind), the framework connects to it as a client. This scheme is typically can be used when the source already delivers multiple streams or the frameworks handles a single stream provided by the source. In this scenario the source can duplicate the same stream to multiple frameworks simultaneously.

![Pub/Sub for Source-Framework communication](https://user-images.githubusercontent.com/15047882/228461503-0e93cd62-986d-43b2-b309-5f905b6f873a.png)

**Framework/BIND-to-Sink/CONNECT communication**. This is a typical schene which can be used widely. The framework as a server can stream results to multiple sink adapters. Every such adapter can filter out only required information.

![PubSub for Framework-Sink communication](https://user-images.githubusercontent.com/15047882/228462824-a615e635-b795-44a9-8680-072b53936a5e.png)

**Source/CONNECT-to-Framework/BIND communication**. This is a typical when the framework handles multiple streams. The framework binds to a socket and clients connect to that socket.

![Savant socket pairs (2)](https://user-images.githubusercontent.com/15047882/228470309-e6dd2746-457f-409c-8eb0-514870dda011.png)

**Framework/CONNECT-to-Sink/BIND communication**. This is not a typical but a legal scheme. The sink handles multiple outputs from frameworks to deliver them some storage, e.g. Kafka or ClickHouse.

![Savant socket pairs (3)](https://user-images.githubusercontent.com/15047882/228470669-47e76e52-a8a4-4055-869a-a2ce405b4319.png)

E.g.:

- you want to pass frames from a single camera to two different pipelines;
- you want to pass resulting video analytics to two different adapters (e.g. RTSP streaming and somewhere else).

The `PUB/SUB` is not a reliable communication pair, which means that if the subscriber is slow the frames will be dropped; the `PUB` part never blocks. To overcome that the adapter must handle incoming frames in a sophisticated way (e.g. using internal queueing).

Generally we recommend using the `PUB/SUB` in the following scenarios:
- you work with raw frames from CAM (MJPEG, RGB, etc.) and if the processing is slow you can afford dropping frames;
- you implemented the adapter in the way to read frames from the socket fast and know how to queue them internally.

Antipattern example: passing video files to the framework with no `SYNC` mode set.

Pattern example: Always-On RTSP Sink Adapter when multiple streams are cast.


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
- `EOS_ON_FILE_END` - flag indicates whether to send EOS message at the end of each file; default is `False`;
- `ZMQ_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`.
- `SYNC_OUTPUT` - flag indicates the need to send frames from source synchronously (i.e. with the frame rate set via the `FRAMERATE` parameter); default is `False`;
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

### The Video File Source Adapter

The video file source adapter reads `video/*` files from `LOCATION`, which can be:

- a local path to a single file;
- a local path to a directory with one or more files;
- HTTP URL to a single file;

The adapter parameters are set with environment variables:

- `LOCATION` - video file(s) location or URL;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `SORT_BY_TIME` - flag indicates whether files from `LOCATION` are sorted by modification time (ascending order); by default, it is `False` and files are sorted lexicographically;
- `READ_METADATA` - flag indicates the need to read the object's metadata from a JSON file that has the identical name as the source file; default is `False`;
- `EOS_ON_FILE_END` - flag indicates whether to send EOS message at the end of each file; default is `True`;
- `ZMQ_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `SYNC_OUTPUT` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is `False`;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; default is `stdout`.

Example:

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

### The Video Loop File Source Adapter

The video loop file source adapter reads a `video/*` file from `LOCATION` and loop it, which can be:

- a local path to a single file;
- HTTP URL to a single file;

The adapter parameters are set with environment variables:

- `LOCATION` - video file location or URL;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `EOS_ON_LOOP_END` - flag indicates whether to send EOS message at the end of each loop; default is `False`;
- `READ_METADATA` - flag indicates the need to read the object's metadata from a JSON file that has the identical name as the source file; default is `False`;
- `ZMQ_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `SYNC_OUTPUT` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is `False`;
- `DOWNLOAD_PATH` - target directory to download files from remote storage in the first loop and reuse it in the next loops;
- `LOSS_RATE` - probability to drop the frames;
- `FPS_PERIOD_FRAMES` - number of frames between FPS reports; default is `1000`;
- `FPS_PERIOD_SECONDS` - number of seconds between FPS reports; default is `None`;
- `FPS_OUTPUT` - path to the file where the FPS reports will be written; default is `stdout`;
- `MEASURE_FPS_PER_LOOP` - flag indicates whether to report FPS at the end of each loop; by default is `False`.

Example:

```bash
docker run --rm -it --name source-video-loop-test \
    --entrypoint /opt/app/adapters/gst/sources/video_loop.sh \
    -e SYNC_OUTPUT=False \
    -e ZMQ_ENDPOINT=ipc:///tmp/zmq-sockets/input-video.ipc \
    -e ZMQ_TYPE=DEALER \
    -e ZMQ_BIND=False \
    -e SOURCE_ID=test \
    -e LOCATION=/path/to/data/test.mp4 \
    -e READ_METADATA=False \
    -e DOWNLOAD_PATH=/tmp/video-loop-source-downloads \
    -v /path/to/data/test.mp4:/path/to/data/test.mp4:ro \
    -v /tmp/zmq-sockets:/tmp/zmq-sockets \
    -v /tmp/video-loop-source-downloads:/tmp/video-loop-source-downloads \
    ghcr.io/insight-platform/savant-adapters-gstreamer:0.2.0
```

The same adapter can be run using a script:

```bash
    ./scripts/run_source.py video-loop --source-id=test /path/to/data/test.mp4
```

### The RTSP Source Adapter

The RTSP source adapter reads RTSP stream from specified `RTSP_URI`.

The adapter parameters are set with environment variables:
- `RTSP_URI` - RTSP URI of the stream; this option is required;
- `SOURCE_ID` - unique identifier for the source stream; this option is required;
- `ZMQ_ENDPOINT` - adapter output (should be equal to the configured framework input) ZeroMQ socket endpoint; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
- `SYNC_OUTPUT` - flag indicates the need to send frames from source synchronously (i.e. at the source file rate); default is `False`;
- `SYNC_DELAY` - delay in seconds before sending frames; useful when the source has B-frames to avoid sending frames in batches; default is `0`;
- `CALCULATE_DTS` - flag indicates whether the adapter should calculate DTS for frames; set this flag when the source has B-frames; default is `False`;
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
- `ZMQ_ENDPOINT` - adapter output (should be equal to module input) ZeroMQ socket endpoint; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
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

The adapter is designed to take video streams from GigE/Genicam industrial cameras. It passes the frames captured from the camera to the framework without encoding (https://github.com/insight-platform/Savant/issues/18) which may introduce significant network payload. We recommend using it locally with the framework deployed at the same host.

The adapter parameters are set with environment variables:

- `WIDTH` - the width of the video frame, in pixels;
- `HEIGHT` - the height of the video frame, in pixels;
- `FRAMERATE` - the framerate of the video stream, in frames per second;
- `INPUT_CAPS` - the format of the video stream, in GStreamer caps format (e.g. `video/x-raw,format=RGB`);
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
- `ZMQ_ENDPOINT` - adapter output (should be equal to module input) ZeroMQ socket endpoint; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - adapter output ZeroMQ socket type; default is `DEALER`, also can be set to `PUB` or `REQ` as well;
- `ZMQ_BIND` - adapter output ZeroMQ socket bind/connect mode (the bind mode is when set to `True`); default is `False`;
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
- `ZMQ_ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `ZMQ_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

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
- `CHUNK_SIZE` - chunk size in frames; the whole stream of incoming frames with metadata is split into separate parts and written to separate folders with consecutive numbering; default value is `10000`. A value of `0` disables chunking within one continuous stream of frames by `source_id`;
- `SKIP_FRAMES_WITHOUT_OBJECTS` - flag that indicates whether frames with `0` objects should be ignored in output; the default value is `False`;
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `ZMQ_ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `ZMQ_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

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
- `ZMQ_ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `ZMQ_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

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

The Display Sink Adapter is a debugging adapter designed for development purposes. To use this adapter, you need a working X server and monitor. The adapter is intended for use with synchronous streams, so for optimal performance, the data source on the adapter side should use the `SYNC=True` mode. The adapter also allows you to specify the `SYNC` flag, but it is better to configure this on the source side also.

The adapter parameters are set with environment variables:

- `CLOSING_DELAY` - delay in seconds before closing the window after the video stream has finished, the default value is 0;
- `SYNC_OUTPUT` - flag indicates whether to show the frames on the sink synchronously with the source (i.e., at the source file rate); if you are intending to use `SYNC` processing, consider `DEALER/ROUTER` or `REQ/REP` sockets, because `PUB/SUB` may drop packets when queues are overflown; 
- `SOURCE_ID` - optional filter to filter out frames with a specific source ID only;
- `SOURCE_ID_PREFIX` - optional filter to filter out frames with a source ID prefix only;
- `ZMQ_ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `ZMQ_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

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

The Always-On RTSP Adapter is designed for low-latency streaming of a single RTSP stream. The adapter accepts only one input stream, so if you plan to stream multiple streams from the framework, you should use a `PUB` socket type on the framework side and a `SUB` socket type on the adapter side. However, if the framework serves a single stream, you can use either `REQ/REP` or `DEALER/ROUTER` pairs.

This adapter is implemented using the DeepStream SDK and performs hardware re-encoding of streams to ensure stable streaming even when the data source stops streaming. In this case, the adapter will continue to stream a static image until the source resumes sending data.

The adapter parameters are set with environment variables:

- `RTSP_URI` - URI of the RTSP server, this parameter is required when `DEV_MODE=False`;
- `DEV_MODE` - use embedded MediaMTX to publish RTSP stream; default value is `False`;
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
- `SYNC_OUTPUT` - flag indicates whether to show frames on sink synchronously (i.e. at the source rate); the streaming may be not stable with this flag, try to avoid it; the default value is `False`;
- `SOURCE_ID` - optional filter to receive frames with a specific source ID only;
- `ZMQ_ENDPOINT` - ZeroMQ socket endpoint for the adapter's input, i.e., the framework output; schema: `[<socket_type>+(bind|connect):]<endpoint>`;
- `ZMQ_TYPE` - ZeroMQ socket type for the adapter's input; the default value is `SUB`, can also be set to `ROUTER` or `REP`;
- `ZMQ_BIND` - flag specifies whether the adapter's input should be bound or connected to the specified endpoint; If `True`, the input is bound; otherwise, it's connected; the default value is `False`.

**Note**: When `DEV_MODE=False` the stream is available at:
- RTSP - `rtsp://<container-host>:554/stream`;
- RTMP - `rtmp://<container-host>:1935/stream`;
- HLS - `http://<container-host>:888/stream`;
- WebRTC - `http://<container-host>:8889/stream`.

Example:

```bash
    docker run --rm -it --name sink-always-on-rtsp \
    --gpus=all \
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
    ghcr.io/insight-platform/savant-adapters-deepstream:0.2.0-6.2 \
    -m adapters.ds.sinks.always_on_rtsp
```

The same adapter can be run using a script:

```bash
    ./scripts/run_sink.py always-on-rtsp --source-id=test --stub-file-location=/path/to/stub_file/test.jpg rtsp://192.168.1.1
```
