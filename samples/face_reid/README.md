# Facial ReID

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

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/face_reid.html
```

### Index Builder

```bash
# if x86
docker compose -f docker-compose.x86.yml --profile index up

# if Jetson
docker compose -f docker-compose.l4t.yml --profile index up

# Ctrl+C to stop running the compose bundle
```

Index Builder module will not stop automatically and will require manual exit. It is safe to do so once the `face_reid-index-builder` container begins repeatedly logging messages like

```
...No Sources found at the input of muxer. Waiting for sources.
```

Check that the `index_files` directory is created and `index.bin` file and `processed_gallery` image directory were written into it.

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

