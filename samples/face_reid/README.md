# Facial ReID

**NB**: The demo uses **YOLOV5-Face** model which takes up to **30-40 minutes** to compile to TensorRT engine. The first launch takes an enormous time.

The sample demonstrates how to use [Yolov5face](https://github.com/deepcam-cn/yolov5-face) face detector with landmarks and [Adaface](https://github.com/mk-minchul/AdaFace) face recognition model to build a facial ReID pipeline that can be utilized, for example, in doorbell security systems.

Preview:

![](assets/face-reid-loop.webp)

The sample is split into two parts: Index Builder and Demo modules.

## Index Builder

Index builder module loads images from [gallery](./assets/gallery), detects faces and facial landmarks, performs face preprocessing and facial recognition model inference. The resulting feature vectors are added into [hnswlib](https://github.com/nmslib/hnswlib) index, and the index (along with cropped face images from gallery) is saved on disk in the `index_files` directory.

Note, when adding new gallery images it is important to make sure that they are as close to 16:9 aspect ratio as possible. The reason being that Index Builder pipeline processes all images in a single resolution with 16:9 aspect ratio, and resizing may introduce image warping that will negatively affect both detector and ReID models' performance.

## Demo

Demo module loads previously generated gallery index file and cropped face images, runs face detection and recognition on a sample video stream, displaying face matches on a padding to the right of the main frame.

## Run

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

### Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/face_reid
git lfs pull
../../utils/check-environment-compatible
```

### Index Builder

Note, there is a bug in the nvv4l2decoder on the Jetson platform so the example currently does not work correctly on that platform. See https://github.com/insight-platform/Savant/issues/314

To build the face reid index, start the `index-builder` container and wait for it to complete initialization.

```bash
# if x86
docker compose -f docker-compose.x86.yml up index-builder

# if Jetson
# currently not supported
docker compose -f docker-compose.l4t.yml up index-builder
```

First startup can take several minutes as the module needs to convert ONNX models into TRT format. Successful module start is indicated by a log message like

```
INFO ... > Pipeline starting ended after 0:00:04.306222
```

Next, run the `pictures-source` container

```bash
# if x86
docker compose -f docker-compose.x86.yml up pictures-source

# if Jetson
# currently not supported
docker compose -f docker-compose.l4t.yml up pictures-source
```

Index Builder module does not stop automatically and requires manual exit. It is safe to do so once the `face_reid-pictures-source` container exits and `face_reid-index-builder` container logs the message following messages (assuming the default gallery image set)

```
nvstreammux: Successfully handled EOS for source_id=9
...
INFO ... > Face processed, index file refreshed
INFO ... > Resources for source gallery has been released.
```

Check that the `index_files` directory is created and `index.bin` file and `processed_gallery` image directory is written into it.

### Demo

```bash
# if x86
docker compose -f docker-compose.x86.yml --profile demo up

# if Jetson
docker compose -f docker-compose.l4t.yml --profile demo up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Performance measurement

Run the Index Builder according to instuctions [above](#index-builder).

Download the video file to your local folder. For example, create a data folder
and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/jumanji_cast.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/jumanji_cast.mp4
```

Build the module image

```bash
# if x86
docker compose -f docker-compose.x86.yml --profile demo build

# if Jetson
docker compose -f docker-compose.l4t.yml --profile demo build
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/face_reid/run_perf.sh
```
