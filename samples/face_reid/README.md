# Facial ReID

The sample demonstrates how to use [Yolov5face](https://github.com/deepcam-cn/yolov5-face) face detector with landmarks and [Adaface](https://github.com/mk-minchul/AdaFace) face recognition model to build a facial ReID pipeline that can be utilized, for example, in doorbell security systems.

Preview:

![](assets/face-reid-loop.webp)

The sample is split into two parts: Index Builder and Demo modules.

## Index Builder

Index builder module loads images from [gallery](./assets/gallery), detects faces and facial landmarks, performs face preprocessing and facial recognition model inference. The resulting feature vectors are added into [hnswlib](https://github.com/nmslib/hnswlib) index, and the index (along with cropped face images from gallery) is saved on disk in the [index_files](./index_files) directory by default.

Note, when adding new gallery images it is important to make sure that they are as close to 16:9 aspect ratio as possible. The reason being that Index Builder pipeline processes all images in a single resolution with 16:9 aspect ratio, and resizing may introduce deformities that will negatively affect both detector and reid models' performance.

## Demo

Demo module loads previously generated gallery index file and cropped face images, runs face detection and recognition on a sample video stream, displaying face matches on a padding to the right of the main frame.

## Run

## Performance measurement

