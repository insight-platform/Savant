# Removing the background on a video frame using MOG2 from OpenCV CUDA

A simple pipeline using MOG2 from OpenCV to remove the background from the frame. 
The background removal acceleration is made by using OpenCV CUDA. 
This allows very fast and efficient background removal.

The left part represents the reference video. 
The right part represents video without background.

Preview:

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- capacity processing: directory of files (FPS).
- use of computer vision methods from OpenCV

Demonstrated adapters:
- RTSP source adapter;
- video file source adapter;
- Always-ON RTSP sink adapter;


**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/background_remover
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/peoplenet.html

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:8554/road-traffic-processed' in your player
# or visit 'http://127.0.0.1:8888/road-traffic-processed/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```


To measure performance:

Download the file to your local folder. For example, create a data folder and download the video into it

```bash
mkdir data && curl -o data/road_traffic.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/road_traffic.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/app/samples \
-v `pwd`/data:/data:ro \
ghcr.io/insight-platform/savant-deepstream:0.2.0-6.2-samples \
samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
```

or for Jetson

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/app/samples \
-v `pwd`/data:/data:ro \
ghcr.io/insight-platform/savant-deepstream-l4t:0.2.0-6.2-samples \
samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
```