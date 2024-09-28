# YOLOv8 Instance Segmentation

**NB**: The demo optionally uses **YOLOV8** model which takes up to **10-15 minutes** to compile to TensorRT engine. The first launch may take a decent time.

A simple pipeline using a [YOLOv8 instance segmentation model](https://docs.ultralytics.com/tasks/segment/) to identify the people in a frame and to segment them from the rest of the frame.

We created an ONNX version of the YOLOv8m-seg model using a script from the original repository (see [Export section](https://docs.ultralytics.com/tasks/segment/#export)). 

To process the model output, we wrote two converters. The default [converter](module/converter.py) processes the model output tensors on the host using `NumPy` and `Numba`. The [gpu_converter](module/gpu_converter.py) processes the model output tensors on the device (GPU) using `CuPy`. You can select one of them in the module [config](module/module.yml) or by setting the `MODEL_OUTPUT_CONVERTER` env variable. By default, the demo runs with `gpu_converter` on the dGPU and uses the default `converter` on Jetson. Using `gpu_converter` can double the performance in some cases, depending on the combination of GPU and CPU, and in any case redistribute the load to the GPU, freeing up CPU resources.

The segmentation model is a complex model in Savant terms. The model produces objects defined by bounding boxes and corresponding masks. To render the bounding boxes and masks we used cv2.cuda.GpuMat, the implementation code is in the [overlay](module/overlay.py).

Preview:

![](assets/shuffle_dance.webp)

Tested on platforms:

- Nvidia Turing
- Nvidia Jetson Orin family

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Build Engines

The demo uses models that are compiled into TensorRT engines the first time the demo is run. This takes time. Optionally, you can prepare the engines before running the demo by using the command:

```bash
# you are expected to be in Savant/ directory

./scripts/run_module.py --build-engines samples/yolov8_seg/module/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/yolov8_seg/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/yolov8_seg/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/shuffle_dance.mp4 \
https://eu-central-1.linodeobjects.com/savant-data/demo/shuffle_dance.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/yolov8_seg/run_perf.sh 
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
