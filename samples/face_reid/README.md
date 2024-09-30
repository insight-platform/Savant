# Facial ReID

**NB**: The demo uses **YOLOV8-Face** model which takes up to **30-40 minutes** to compile to TensorRT engine. The first launch takes an enormous time.

The sample demonstrates how to use [YOLOV8-Face](https://github.com/akanametov/yolov8-face) face detector with landmarks and [Adaface](https://github.com/mk-minchul/AdaFace) face recognition model to build a facial ReID pipeline that can be utilized, for example, in doorbell security systems.

Preview:

![](assets/face-reid-loop.webp)

The sample is split into two parts: Index Builder and Demo modules.

Tested on platforms:

- Nvidia Turing
- Nvidia Jetson Orin family

## Index Builder

Index builder module loads images from [gallery](./assets/gallery), detects faces and facial landmarks, performs face preprocessing and facial recognition model inference. The resulting feature vectors are added into [hnswlib](https://github.com/nmslib/hnswlib) index, and the index (along with cropped face images from gallery) is saved on disk in the `index_files` directory.

Note that the Index Builder pipeline processes all images in a single resolution with 16:9 aspect ratio, but the gallery is sent to the pipeline through the Client SDK which can take care of padding the image to a target aspect ratio. As such it's safe to add new images of any aspect ratio to the gallery. The gallery images are expected to contain 1 face each and to be named according to scheme `<person_name>_<img_n>.jpeg`.

## Demo

Demo module loads previously generated gallery index file and cropped face images, runs face detection and recognition on a sample video stream, displaying face matches on a padding to the right of the main frame.

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

./scripts/run_module.py --build-engines samples/face_reid/src/module.yml
```

## Run Index Builder

Note, there is a bug in the nvv4l2decoder on the Jetson platform so the example currently does not work correctly on that platform. See https://github.com/insight-platform/Savant/issues/314

To build the face reid index, start the `index` docker compose profile and wait for the services to complete building the index.

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/face_reid/docker-compose.x86.yml --profile index up

# if Jetson
# currently not supported
docker compose -f samples/face_reid/docker-compose.l4t.yml --profile index up
```

The first launch can take several minutes as the `index-builder-pipeline` needs to convert ONNX models into TRT format. Successful module start is indicated by a log messages like

```
INFO ... > The pipeline is initialized and ready to process data. Initialization took ...
INFO ... > Setting module status to ModuleStatus.RUNNING
```

After that the `index-builder-client` container will be started automatically.

The `index-builder-client` service runs the [index_builder_client.py](./src/index_builder_client.py) script, which loads the images from the [gallery](./assets/gallery), resizes them to the pipeline frame size while preserving content aspect ratio, sends them to the `index-builder-pipeline` and uses the received results to create `index_files/index.bin` index file and the cropped face images in the `index_files/processed_gallery` dir.

After the services complete, the containers shut down automatically. Check that the `index_files` directory is created and `index.bin` file and `processed_gallery` image directory is written into it.

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/face_reid/docker-compose.x86.yml --profile demo up

# if Jetson
docker compose -f samples/face_reid/docker-compose.l4t.yml --profile demo up

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Performance Measurement

Run the Index Builder according to instructions [above](#index-builder).

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data
curl -o data/jumanji_cast.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/jumanji_cast.mp4
```

Run the performance benchmark with the following command:

```bash
./samples/face_reid/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
