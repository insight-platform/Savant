# Facial ReID

**NB**: The demo uses **YOLOV5-Face** model which takes up to **30-40 minutes** to compile to TensorRT engine. The first launch takes an enormous time.

The sample demonstrates how to use [Yolov5face](https://github.com/deepcam-cn/yolov5-face) face detector with landmarks and [Adaface](https://github.com/mk-minchul/AdaFace) face recognition model to build a facial ReID pipeline that can be utilized, for example, in doorbell security systems.

Preview:

![](assets/face-reid-loop.webp)

The sample is split into two parts: Index Builder and Demo modules.

## Index Builder

Index builder module loads images from [gallery](./assets/gallery), detects faces and facial landmarks, performs face preprocessing and facial recognition model inference. The resulting feature vectors are added into [hnswlib](https://github.com/nmslib/hnswlib) index, and the index (along with cropped face images from gallery) is saved on disk in the `index_files` directory.

Note that the Index Builder pipeline processes all images in a single resolution with 16:9 aspect ratio, but the gallery is sent to the pipeline through the Client SDK which can take care of padding the image to a target aspect ratio. As such it's safe to add new images of any aspect ratio to the gallery. The gallery images are expected to contain 1 face each and to be named according to scheme `<person_name>_<img_n>.jpeg`.

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

To build the face reid index, start the `index` docker compose profile and wait for the services to complete building the index.

```bash
# if x86
docker compose -f docker-compose.x86.yml --profile index up

# if Jetson
# currently not supported
docker compose -f docker-compose.l4t.yml --profile index up
```

First startup can take several minutes as the `index-builder-pipeline` needs to convert ONNX models into TRT format. Successful module start is indicated by a log messages like

```
INFO ... > The pipeline is initialized and ready to process data. Initialization took ...
INFO ... > Setting module status to ModuleStatus.RUNNING
```

After that the `index-builder-client` container will be started automatically.

The `index-builder-client` service runs the [index_builder_client.py](./src/index_builder_client.py) script, which loads the images from the [gallery](./assets/gallery), pads them to 16:9 aspect ratio if necessary, sends them to the `index-builder-pipeline` and uses the received results to create `index_files/index.bin` index file and the cropped face images in the `index_files/processed_gallery` dir.

After the services complete, the containers shut down automatically. Check that the `index_files` directory is created and `index.bin` file and `processed_gallery` image directory is written into it.

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

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data
curl -o data/jumanji_cast.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/jumanji_cast.mp4
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
