# YOLOv8 Instance Segmentation

A simple pipeline using a [YOLOv8 instance segmentation model](https://docs.ultralytics.com/tasks/segment/) to identify the people in a frame and to segment them from the rest of the frame.

Preview:

TBD

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- instance segmentation;

Demonstrated adapters:
- RTSP source adapter;
- video file source adapter;
- Always-ON RTSP sink adapter;
- Video/Metadata sink adapter.


Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/yolov8_seg

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


## Performance Measurement

Download the video file to your local folder. For example, create a data folder 
and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/suffle_dance.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/suffle_dance.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/yolov8_seg/run_perf.sh 
```
