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
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/panoptic-driving-perception' in your player
# or visit 'http://127.0.0.1:888/stream/panoptic-driving-perception/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/panoptic_driving_perception.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/panoptic_driving_perception.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/panoptic_driving_perception/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
