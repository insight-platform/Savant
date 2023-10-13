# Performance measurements

## Hardware

| Name      | CPU                                | GPU              | RAM, Gi |
|-----------|------------------------------------|------------------|---------|
| A4000     | AMD Ryzen 7 3700X 8-Core Processor | NVIDIA RTX A4000 | 62      |
| Jetson NX | MODE 20W, 6 CORE, Clocks On        |                  | 8       |

## Pipeline FPS

### age_gender_recognition

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 174.10 | 36.34     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 171.82 | 35.73     |

### conditional_video_processing

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [#244](https://github.com/insight-platform/Savant/issues/244)                    | 327    | 64        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 325.64 | 64.90     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 315.22 | 63.87     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 288.91 | 61.60     |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues)        | 311.62 | 61.46     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 263.44 | 59.86     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 263.89 | 63.52     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 257.63 | 62.77     |

### face_reid

| Savant ver.                                                               | A4000  | Jetson NX |
|---------------------------------------------------------------------------|--------|-----------|
| [#142](https://github.com/insight-platform/Savant/issues/142)             | 124    | 26        |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues) | 121.11 | 25.4      |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues) | 118.79 | 25.99     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues) | 127.37 | 27.61     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues) | 127.23 | 28.71     |

### nvidia_car_classification

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 155    | 42        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 161.61 | 45.43     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 157.49 | 44.16     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 156.72 | 43.24     |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues)        | 156.73 | 42.44     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 151.44 | 42.97     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 149.66 | 41.11     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 150.07 | 38.76     |

### opencv_cuda_bg_remover_mog2

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 675    | 65        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 689.91 | 87.73     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 669.15 | 93.14     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 671.76 | 92.41     |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues)        | 671.34 | 92.24     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 608.54 | 91.69     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 607.48 | 90.92     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 606.74 | 95.05     |

### opencv_cuda_bg_remover_mog2 (multi-stream)

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [#372](https://github.com/insight-platform/Savant/issues/372) (no queues)        | 510.82 | 91.62     |
| [#372](https://github.com/insight-platform/Savant/issues/372) (buffer length 10) | 510.97 | 93.20     |

### peoplenet_detector

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 125    | 30        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 118.68 | 28.97     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 116.35 | 29.31     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 114.51 | 29.46     |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues)        | 117.22 | 29.27     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 116.43 | 28.05     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 116.61 | 28.54     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 116.44 | 26.03     |

### traffic_meter (yolov8m)

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 132    | 23        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 140.43 | 23.62     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 137.62 | 26.59     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 123.98 | 25.61     |
| [#341](https://github.com/insight-platform/Savant/issues/341) (no queues)        | 135.19 | 24.67     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 136.49 | 24.40     |
| [#407](https://github.com/insight-platform/Savant/issues/407) (no queues)        | 136.16 | 24.66     |
| [#456](https://github.com/insight-platform/Savant/issues/456) (no queues)        | 136.80 | 23.29     |


### intersection_traffic_meter (yolov8m)

| Savant ver.                                                   | A4000  | Jetson NX |
|---------------------------------------------------------------|--------|-----------|
| [#356](https://github.com/insight-platform/Savant/issues/356) | 93.03   | 21.22     |

### yolov8_seg

Note: `yolov8_seg` always has a buffer length of 10. `BUFFER_QUEUES` env doesn't affect it.

| Savant ver.                                                   | A4000 | Jetson NX |
|---------------------------------------------------------------|-------|-----------|
| [#131](https://github.com/insight-platform/Savant/issues/131) | 45.78 | 14.84     |
| [#334](https://github.com/insight-platform/Savant/issues/334) | 44.33 | 14.82     |
| [#341](https://github.com/insight-platform/Savant/issues/341) | 45.21 | 14.02     |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 44.34 | 13.07     |
| [#407](https://github.com/insight-platform/Savant/issues/407) | 67.73 | 21.57     |
| [#456](https://github.com/insight-platform/Savant/issues/456) | 68.48 | 21.71     |

### license_plate_recognition

| Savant ver.                                                   | A4000 | Jetson NX |
|---------------------------------------------------------------|-------|-----------|
| [#455](https://github.com/insight-platform/Savant/issues/455) | 92.4  | 25.29     |
