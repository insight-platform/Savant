# RT-DETR R50 Demo

The sample shows how RT-DETR model can be used in a Savant module.

The detector is used as a plain ONNX model: bounding boxes are parsed by the Savant YOLO output converter, so no custom nvinfer parsing library and no custom module image are needed.

The ONNX model shipped with the sample was exported from the [RT-DETR PyTorch implementation](https://github.com/lyuwenyu/RT-DETR) with [export_rtdetr_pytorch.py](./export/export_rtdetr_pytorch.py), see [Export the ONNX Model](#export-the-onnx-model) to reproduce it.

Tested on platforms:

- Nvidia Turing, Ampere
- Nvidia Jetson Orin family

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

## Export the ONNX Model

The sample runs on a model exported from the RT-DETR repo. The steps below reproduce the exact ONNX file the sample downloads.

1. Check out the RT-DETR repo at the commit the export was tested with and download the weights:

```bash
git clone https://github.com/lyuwenyu/RT-DETR.git
cd RT-DETR
git checkout 0d0a4f5
cd rtdetr_pytorch
wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

2. Create a virtualenv and install the pinned export environment ([requirements.txt](requirements.txt) from this sample):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r /path/to/Savant/samples/rtdetr/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

3. Copy the export script from this sample into `RT-DETR/rtdetr_pytorch` and run it:

```bash
cp /path/to/Savant/samples/rtdetr/export/export_rtdetr_pytorch.py .
python3 export_rtdetr_pytorch.py \
  -w rtdetr_r50vd_6x_coco_from_paddle.pth \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  --dynamic --simplify
```

The resulting `rtdetr_r50vd_6x_coco_from_paddle.onnx` has a single input `input` of shape `batch x 3 x 640 x 640` and a single output `output` of shape `batch x 84 x 300`, where the first 4 rows are `cxcywh` boxes in network input pixels and the remaining 80 rows are per-class scores. This is the layout `savant.converter.yolo.TensorToBBoxConverter` expects, which is why `num_detected_classes: 80` in [module.yml](module.yml) is necessary: the converter dispatches on `shape[0] == num_detected_classes + 4`.

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

# if Jetson
docker compose -f samples/rtdetr/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/leeds' in your player
# or visit 'http://127.0.0.1:888/stream/leeds/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
