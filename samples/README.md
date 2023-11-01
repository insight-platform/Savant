# Savant Demos, Samples and Education Materials

On this page, you can find practical examples, how-tos, publications, and other non-formal documentation links that can
help you dive into Savant.

- [ML/AI Examples](#mlai-examples)
- [Utility And Coding Examples](#utility-and-coding-examples)

## ML/AI Examples

To help novice users to dive into Savant, we have prepared several ready-to-use examples, which one can download, build
and launch. In addition, every sample includes all you need to run it: the source code, models, build file, sample data,
and a short README.md which covers the purpose and gives a brief explanation. Some samples are also accompanied by the
how-to guides published on the Medium portal, where one can study them step-by-step.

### People Detecting, Tracking and Anonymizing (Peoplenet, Nvidia Tracker, OpenCV-CUDA)

Sample Location: [peoplenet_detector](./peoplenet_detector)

Preview:

![](peoplenet_detector/assets/peoplenet-blur-demo-loop-400.webp)

### Car Detection and Classification (Nvidia detectors and classifiers, Nvidia tracker)

Sample Location: [nvidia_car_classification](./nvidia_car_classification)

Preview:

![](nvidia_car_classification/assets/nvidia-car-classification-loop-400.webp)

### Traffic Meter

Sample Location: [traffic_meter](./traffic_meter)

Preview:

![](traffic_meter/assets/traffic-meter-loop-400.webp)

### Intersection Traffic Meter

Sample Location: [intersection_traffic_meter](./intersection_traffic_meter)

Preview:

![](intersection_traffic_meter/assets/intersection-traffic-meter-loop-400.webp)

### Area Object Counting

Sample Location: [area_object_counting](./area_object_counting/)

![](area_object_counting/assets/area-object-counting-loop-400.webp)

### Age Gender Recognition

Sample Location: [age gender recognition](./age_gender_recognition)

Preview:

![](age_gender_recognition/assets/age-gender-recognition-loop-400.webp)

### YOLOv8 Instance Segmentation

Sample Location: [yolov8_seg](./yolov8_seg)

Preview:

![](yolov8_seg/assets/shuffle_dance-400.webp)

### Facial ReID

Sample location: [face_reid](./face_reid)

Preview:

![](face_reid/assets/face-reid-loop-400.webp)

### License Plate Recognition

Sample location: [LPR](./license_plate_recognition)

Preview:

![](license_plate_recognition/assets/license-plate-recognition-400.webp)

### Keypoint Detection 

Sample location: [keypoint_detection](./keypoint_detection)

Preview:

![](keypoint_detection/assets/shuffle_dance-400.webp)

### Super Resolution 

Sample location: [super_resolution](./super_resolution)

Preview:

![](super_resolution/assets/shuffle_dance_360p_1080p_small.webp)

## Utility And Coding Examples

### OpenCV CUDA MOG2 Background Segmentation Demo

Sample Location: [opencv_cuda_bg_remover_mog2](./opencv_cuda_bg_remover_mog2)

Preview:

![](opencv_cuda_bg_remover_mog2/assets/opencv_cuda_bg_remover_mog2-800.webp)

### Conditional Video Processing

Sample location: [conditional_video_processing](./conditional_video_processing)

Preview:

![](conditional_video_processing/assets/conditional-video-processing_400.webp)

### Multiple RTSP Streams Demo

A simple pipeline demonstrates how multiplexed processing works in Savant. In the demo, two RTSP streams are ingested in the module and processed with the PeopleNet model.

Sample Location: [multiple_rtsp](./multiple_rtsp)

### Testing RTSP Camera Compatibility

A very primitive source-sink pipeline testing that RTSP cam normally processed by NVDEC and thus, is compatible with Savant.

Location: [rtsp_cam_compatibility_test](./rtsp_cam_compatibility_test)

### Multiple GigE Vision Cameras Demo

A simple pipeline demonstrates how GigE Vision Source Adapter works in Savant. In the demo video from one GigE Vision camera is passed as raw-rgba frames, and another one is passed as HEVC-encoded frames. Both streams are passed to Always-On-RTSP sinks.

Sample Location: [multiple_gige](./multiple_gige)

### Source Adapter With JSON Metadata

A demo showing how to inject ground truth metadata into frames and use them to estimate the model performance.

Sample location: [source_adapter_with_json_metadata](./source_adapter_with_json_metadata)

### OpenTelemetry Sample

A sample demonstrating the use of OpenTelemetry in Savant.

Sample location: [telemetry](./telemetry)

### Kafka-Redis Adapters Demo

A sample demonstrating the use of Kafka-Redis adapters in Savant.

Sample location: [kafka_redis_adapter](./kafka_redis_adapter)

### Pass-through Processing

A sample demonstrating the use of pass-through mode in Savant.

Sample location: [pass_through_processing](./pass_through_processing)
