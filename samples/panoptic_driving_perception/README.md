# Panoptic Driving Perception by PyTorch inference

The app shows how to use the torch hub and make PyTorch inference in Savant using reference PyTorch model. Also, it shows how to interact with image in GPU memory. The [YOLOP](https://github.com/hustvl/YOLOP) model is used for object detection and semantic segmentation.

Preview:

![](assets/panoptic_driving_perception.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- Video loop source adapter;
- Always-ON RTSP sink adapter.

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
cd samples/panoptic_driving_perception

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
# Prepare docker image with torchvision
docker buildx build --target builder -f ./docker/Dockerfile.l4t -t savant_torch_build .
docker run -it --rm --runtime nvidia -e MAX_JOBS=1 -v `pwd`/torchvision:/torchvision --entrypoint /bin/bash savant_torch_build /opt/torchvision/build_torchvision.sh
docker buildx build --target savant_torch -f ./docker/Dockerfile.l4t -t panoptic_driving_perception-module .

# Run the demo
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

