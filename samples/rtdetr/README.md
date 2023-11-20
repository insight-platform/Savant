# RT-DETR R50 Demo

The sample shows how RT-DETR model can be used in a Savant module.

The detector model was prepared in the ONNX format using instructions from [DeepStream-Yolo repo](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/RTDETR.md).

Weights used: `v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth`  from the [RT-DETR releases](https://github.com/lyuwenyu/storage/releases).

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/rtdetr
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up
```
