# Conditional Video Processing

A simple pipeline that demonstrates conditional video processing, drawing on frames and encoding. The first element of the pipeline is the pyfunc [ConditionalSkipProcessing](conditional_video_processing.py). Pyfunc checks if the source should be processed by checking the value of the corresponding parameter (the source name) in Etcd. If not, it removes the primary object and therefore disables downstream inference. The secondary element is the [Nvidia PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) model. The model is used to detect persons in the video. When the model detects person, the pyfunc [ConditionalVideoProcessing](conditional_video_processing.py) adds tags `draw` and `encode` to the frame. DrawFunc draws on frame only when tag `draw` is present and encoder encodes frame only when tag `encode` is present. On Always-On RTSP sink you can see the video constantly switching between original and stub frames.

Preview:

![](assets/conditional-video-processing.webp)

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/conditional_video_processing
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Source processing control

By default, the pipeline processes any source. Etcd is used to control the processing. By changing the value of the key `savant/sources/{source-id}` in Etcd you can enable or disable processing of the corresponding source.

To enable/disable source processing it is convenient to use the script:
```bash
./source-switch.sh on
# or
./source-switch.sh off
```

## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/conditional_video_processing.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/conditional_video_processing.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/conditional_video_processing/run_perf.sh
```
