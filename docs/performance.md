# Performance measurements

## Hardware

| Name      | Device                         | CPU                                              | RAM, Gi |
|-----------|--------------------------------|--------------------------------------------------|---------|
| A4000     | RTX A4000                      | AMD Ryzen 7 3700X 8-Core Processor               | 62      |
| Xavier NX | Jetson Xavier NX Developer Kit | ARMv8 Processor rev 1 (v8l), MODE 20W, Clocks On | 8       |
| Orin Nano | Orin Nano Developer Kit        | ARMv8 Processor rev 1 (v8l), MODE 15W, Clocks On | 8       |

## Pipeline FPS

### age_gender_recognition

| Savant ver.                                                                     | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|--------|-----------|-----------|
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 174.10 | 36.34     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 171.82 | 35.73     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 178.01 | 33.75     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 161.77 | 37.95     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 351.19 |           | 71.03     |

### conditional_video_processing

| Savant ver.                                                                     | A4000  | Xavier NX |
|---------------------------------------------------------------------------------|--------|-----------|
| [#244](https://github.com/insight-platform/Savant/issues/244)                   | 327    | 64        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                | 325.64 | 64.90     |
| [#334](https://github.com/insight-platform/Savant/issues/334)                   | 315.22 | 63.87     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (queue length 10) | 288.91 | 61.60     |
| [#341](https://github.com/insight-platform/Savant/issues/341)                   | 311.62 | 61.46     |
| [#347](https://github.com/insight-platform/Savant/issues/347)                   | 263.44 | 59.86     |
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 263.89 | 63.52     |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 257.63 | 62.77     |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 260.36 | 63.60     |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 248.70 | 58.37     |

### face_reid

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#142](https://github.com/insight-platform/Savant/issues/142) | 124    | 26        |           |
| [#341](https://github.com/insight-platform/Savant/issues/341) | 121.11 | 25.4      |           |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 118.79 | 25.99     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407) | 127.37 | 27.61     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456) | 127.23 | 28.71     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 130.15 | 28.72     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 220.48 |           | 50.39     |

### intersection_traffic_meter (yolov8m)

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#356](https://github.com/insight-platform/Savant/issues/356) | 93.03  | 21.22     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 94.56  | 21.56     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 260.17 |           | 49.02     |

### license_plate_recognition

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#455](https://github.com/insight-platform/Savant/issues/455) | 92.4   | 25.29     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 92.73  | 25.32     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 270.33 |           | 43.6      |

### nvidia_car_classification

| Savant ver.                                                                     | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|--------|-----------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                | 155    | 42        |           |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                | 161.61 | 45.43     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334)                   | 157.49 | 44.16     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334) (queue length 10) | 156.72 | 43.24     |           |
| [#341](https://github.com/insight-platform/Savant/issues/341)                   | 156.73 | 42.44     |           |
| [#347](https://github.com/insight-platform/Savant/issues/347)                   | 151.44 | 42.97     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 149.66 | 41.11     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 150.07 | 38.76     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 151.47 | 42.00     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 149.89 | 41.79     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 413.63 |           | 131.88    |

### opencv_cuda_bg_remover_mog2

| Savant ver.                                                                     | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|--------|-----------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                | 675    | 65        |           |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                | 689.91 | 87.73     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334)                   | 669.15 | 93.14     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334) (queue length 10) | 671.76 | 92.41     |           |
| [#341](https://github.com/insight-platform/Savant/issues/341)                   | 671.34 | 92.24     |           |
| [#347](https://github.com/insight-platform/Savant/issues/347)                   | 608.54 | 91.69     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 607.48 | 90.92     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 606.74 | 95.05     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 610.06 | 94.00     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 618.57 | 95.59     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 725.24 |           | 133.11    |

### opencv_cuda_bg_remover_mog2 (multi-stream)

| Savant ver.                                                                     | A4000  | Xavier NX |
|---------------------------------------------------------------------------------|--------|-----------|
| [#372](https://github.com/insight-platform/Savant/issues/372)                   | 510.82 | 91.62     |
| [#372](https://github.com/insight-platform/Savant/issues/372) (queue length 10) | 510.97 | 93.20     |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 595.70 | 89.13     |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 598.71 | 87.69     |

### peoplenet_detector

| Savant ver.                                                                     | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|--------|-----------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                | 125    | 30        |           |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                | 118.68 | 28.97     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334)                   | 116.35 | 29.31     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334) (queue length 10) | 114.51 | 29.46     |           |
| [#341](https://github.com/insight-platform/Savant/issues/341)                   | 117.22 | 29.27     |           |
| [#347](https://github.com/insight-platform/Savant/issues/347)                   | 116.43 | 28.05     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 116.61 | 28.54     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 116.44 | 26.03     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 124.87 | 29.13     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 110.64 | 26.73     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 381.55 |           | 128.03    |

### traffic_meter (yolov8m)

| Savant ver.                                                                     | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|--------|-----------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                | 132    | 23        |           |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                | 140.43 | 23.62     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334)                   | 137.62 | 26.59     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334) (queue length 10) | 123.98 | 25.61     |           |
| [#341](https://github.com/insight-platform/Savant/issues/341)                   | 135.19 | 24.67     |           |
| [#347](https://github.com/insight-platform/Savant/issues/347)                   | 136.49 | 24.40     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 136.16 | 24.66     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 136.80 | 23.29     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 134.26 | 23.98     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 123.88 | 19.33     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 254.98 |           | 102.61    |

### yolov8_seg

Note: `yolov8_seg` always has a queue length of 10.

| Savant ver.                                                   | A4000 | Xavier NX | Orin Nano |
|---------------------------------------------------------------|-------|-----------|-----------|
| [#131](https://github.com/insight-platform/Savant/issues/131) | 45.78 | 14.84     |           |
| [#334](https://github.com/insight-platform/Savant/issues/334) | 44.33 | 14.82     |           |
| [#341](https://github.com/insight-platform/Savant/issues/341) | 45.21 | 14.02     |           |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 44.34 | 13.07     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407) | 67.73 | 21.57     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456) | 68.48 | 21.71     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 71.28 | 21.95     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 87.09 |           | 35.99     |
