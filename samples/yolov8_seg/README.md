# YOLOv8 Instance Segmentation

A simple pipeline using a [YOLOv8 instance segmentation model](https://docs.ultralytics.com/tasks/segment/) to identify the people in a frame and to segment them from the rest of the frame.

We created an ONNX version of the YOLOv8m-seg model using a script from the original repository (see [Export section](https://docs.ultralytics.com/tasks/segment/#export)). To process the model output, we wrote [converter](module/converter.py). The segmentation model is a complex model in Savant terms. The model produces objects defined by bounding boxes and corresponding masks. To render the bounding boxes and masks we used cv2.cuda.GpuMat, the implementation code is in the [overlay](module/overlay.py).

Preview:

![](assets/shuffle_dance.webp)


Tested on platforms:
- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

## Run the demo

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

mkdir -p data && curl -o data/shuffle_dance.mp4 \
https://eu-central-1.linodeobjects.com/savant-data/demo/shuffle_dance.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/yolov8_seg/run_perf.sh 
```
