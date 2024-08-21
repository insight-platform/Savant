# NanoSAM

The demo pipeline demonstrates how to utilize [NanoSAM model](https://github.com/NVIDIA-AI-IOT/nanosam/tree/main) to segment objects within a frame. Four dots are positioned on the frame to identify and separate objects. 
The dots appear in black, and each object is assigned a unique color (green, red, blue, yellow) with a gradient.
The gradient is constructed from a series of masks produced by the model, with each mask being wider than the preceding one in relation to the point's position so that small objects close to the point will have a brighter color.

In addition, the pipeline demonstrates how to use a pyfunc element to handle cases where a model has custom inputs.

Preview:

![](assets/sam_4_masks.webp)

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

./scripts/run_module.py --build-engines samples/nanosam/module/demo.yml
```

## Run Demo

Mask decoder engine has custom inputs which are not supported by the default engine builder. 
You need to do the [previous step](#build-engines) and then build a mask decoder engine with the following command:

```bash
# you are expected to be in Savant/ directory

# if x86
docker run --rm \
  --gpus=all \
  -v "$(pwd)/cache/models/nanosam:/opt/nanosam" \
  --entrypoint bash ghcr.io/insight-platform/savant-deepstream-extra \
  -c "/usr/src/tensorrt/bin/trtexec --onnx=/opt/nanosam/image_encoder/mobile_sam_mask_decoder.onnx \
    --saveEngine=/opt/nanosam/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10"
    
# if Jetson
docker run --rm \
  --runtime=nvidia \
  -v "$(pwd)/cache/models/nanosam:/opt/nanosam" \
  --entrypoint bash ghcr.io/insight-platform/savant-deepstream-l4t-extra \
  -c "/usr/src/tensorrt/bin/trtexec --onnx=/opt/nanosam/image_encoder/mobile_sam_mask_decoder.onnx \
    --saveEngine=/opt/nanosam/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10"
```

Then you can run the demo:

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/nanosam/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/nanosam/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
