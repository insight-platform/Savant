# Faces detection, tracking and age-gender recognition (YoloV5face, Nvidia Tracker, Age-Gender model)

A pipeline that uses [Yolov5face](https://github.com/deepcam-cn/yolov5-face) model to detect faces and 5 face landmarks (eyes, nose, mouth). Landmarks are used 
to calculate the faces orientation and preprocess face images for age/gender model.  
Age/gender model estimate age and gender for each face and add this information to 
the object metadata. The pipeline uses Nvidia Tracker to track face.

Preview:

![](assets/age-gender-recognition-loop.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- capacity processing: directory of files (FPS).
- image preprocessing for model input;

Demonstrated adapters:
- RTSP source adapter;
- video file source adapter;
- Always-ON RTSP sink adapter;
- Video/Metadata sink adapter.


**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/age_gender_recognition
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/age_gender_recognition.html

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```


## Performance Measurement

Download the video file to your local folder. For example, create a data folder 
and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/elon_musk.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/elon_musk.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/savant/samples \
-v `pwd`/data:/data:ro \
-v `pwd`/downloads:/downloads \
-v `pwd`/models/age_gender_recognition:/models \
ghcr.io/insight-platform/savant-deepstream:latest \
samples/age_gender_recognition/module_performance.yml
```

or for Jetson

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/savant/samples \
-v `pwd`/data:/data:ro \
-v `pwd`/downloads:/downloads \
-v `pwd`/models/age_gender_recognition:/models \
ghcr.io/insight-platform/savant-deepstream-l4t:latest \
samples/age_gender_recognition/module_performance.yml
```