# Conditional Video Processing

A simple pipeline that demonstrates conditional drawing on frames and encoding. The pipeline uses standard [Nvidia PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) model to detect persons in the video. When the model detects person, the pyfunc [ConditionalVideoProcessing](conditional_video_processing.py) adds tags `draw` and `encode` to the frame. DrawFunc draws on frame only when tag `draw` is present and encoder encodes frame only when tag `encode` is present. On Always-On RTSP sink you can see the video constantly switching between original and stub frames.

Preview:

![](assets/conditional-video-processing.webp)

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/conditional_video_processing
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/peoplenet.html

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
