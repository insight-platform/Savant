# People detection, tracking and face blurring (PeopleNet, Nvidia Tracker, OpenCV CUDA)

A pipeline that uses [Yolov5face](https://github.com/deepcam-cn/yolov5-face) model to detect faces. 
The model detected faces and 5 landmarks (eyes, nose, mouth). Landmarks are used 
to calculate the face orientation and preprocessing face image for age/gender model.  
Age/gender model estimate age and gender for each face and add this information to 
the object metadata. The pipeline uses Nvidia Tracker to track face.

Preview:

![](assets/peoplenet-blur-demo-loop.webp)

YouTube Video:

[![Watch the video](https://img.youtube.com/vi/YCvT3XbiSik/default.jpg)](https://youtu.be/YCvT3XbiSik)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Ampere.

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
