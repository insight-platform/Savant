# RT-DETR R50 Demo

The sample shows how RT-DETR model can be used in a Savant module.

The detector model was prepared in the ONNX format using instructions from [DeepStream-Yolo repo](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/RTDETR.md).

Weights used: `v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth`  from the [RT-DETR releases](https://github.com/lyuwenyu/storage/releases).

Tested on platforms:

- Nvidia Turing;
- Nvidia Jetson Orin Nano.

Demonstrated operational modes:

- real-time processing: RTSP streams.

Demonstrated adapters:

- RTSP source adapter;
- Always-ON RTSP sink adapter.

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

./samples/rtdetr/build_engines.sh
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/rtdetr/docker-compose.x86.yml up

# open 'rtsp://127.0.0.1:554/stream/leeds' in your player
# or visit 'http://127.0.0.1:888/stream/leeds/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
