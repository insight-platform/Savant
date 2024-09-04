# Performance measurements

## Hardware

| Name      | Device                         | CPU                                              | RAM, Gi |
|-----------|--------------------------------|--------------------------------------------------|---------|
| A4000     | RTX A4000                      | AMD Ryzen 7 3700X 8-Core Processor               | 62      |
| Xavier NX | Jetson Xavier NX Developer Kit | ARMv8 Processor rev 1 (v8l), MODE 20W, Clocks On | 8       |
| Orin Nano | Orin Nano Developer Kit        | ARMv8 Processor rev 1 (v8l), MODE 15W, Clocks On | 8       |

## Pipeline FPS

### age_gender_recognition

| Savant ver.                                                                     | A4000          | Xavier NX | Orin Nano |
|---------------------------------------------------------------------------------|----------------|-----------|-----------|
| [#407](https://github.com/insight-platform/Savant/issues/407)                   | 174.10         | 36.34     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456)                   | 171.82         | 35.73     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 178.01         | 33.75     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 161.77         | 37.95     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 371.42         | 53.67?    | 70.58     |
| [#550](https://github.com/insight-platform/Savant/issues/550)                   | 371.29         |           | 71.79     |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 376.48         |           | 71.95     |
| [#641](https://github.com/insight-platform/Savant/issues/641)                   | 373.69         |           | 80.4      |
| [#783](https://github.com/insight-platform/Savant/issues/783)                   | 349.08         |           | 79.80     |
| [#785](https://github.com/insight-platform/Savant/issues/785)                   | 333.17         |           | 79.88     |
| [#679](https://github.com/insight-platform/Savant/issues/679)                   | 332.82         |           | 79.34     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | NoneType error |           |      |

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
| [#852](https://github.com/insight-platform/Savant/issues/852) |  |           |      |

### face_reid

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#142](https://github.com/insight-platform/Savant/issues/142) | 124    | 26        |           |
| [#341](https://github.com/insight-platform/Savant/issues/341) | 121.11 | 25.4      |           |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 118.79 | 25.99     |           |
| [#407](https://github.com/insight-platform/Savant/issues/407) | 127.37 | 27.61     |           |
| [#456](https://github.com/insight-platform/Savant/issues/456) | 127.23 | 28.71     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 130.15 | 28.72     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 224.22 | 36.45?    | 48.92     |
| [#550](https://github.com/insight-platform/Savant/issues/550) | 229.22 |           | 49.77     |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 229.51 |           | 50.04     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 231.26 |           | 54.8      |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 220.24 |           | 54.48     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 217.75 |           | 54.88     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 214.66 |           | 54.41     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | NoneType error |           |      |

### fisheye_line_crossing

| Savant ver.                                                   | A4000 | Xavier NX | Orin Nano |
|---------------------------------------------------------------|-------|-----------|-----------|
| [#193](https://github.com/insight-platform/Savant/issues/193) | 86.6  | 23.9      | 33.7      |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 86.6  |           | 33.71     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 86.64 |           | 35.46     |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 86.64 |           | 35.48     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 86.63 |           | 34.93     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 86.63 |           | 34.91     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 86.19 |           |      |

### keypoint_detection

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#641](https://github.com/insight-platform/Savant/issues/641) | 284.19 | 58.85     | 79.34     |
| [#692](https://github.com/insight-platform/Savant/issues/692) | 758.62 |           | 202.09    |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 690.44 |           | 203.47    |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 684.67 |           | 205.34    |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 743.31 |           | 206.89    |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 789.31 |           |      |

### intersection_traffic_meter (yolov8m)

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#356](https://github.com/insight-platform/Savant/issues/356) | 93.03  | 21.22     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 94.56  | 21.56     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 264.02 | 32.15?    | 41.14     |
| [#550](https://github.com/insight-platform/Savant/issues/550) | 268.50 |           | 41.11     |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 271.13 |           | 41.11     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 273.91 |           | 43.19     |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 265.54 |           | 43.10     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 262.47 |           | 42.92     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 264.45 |           | 43.02     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 269.38 |           |      |

### license_plate_recognition

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#455](https://github.com/insight-platform/Savant/issues/455) | 92.4   | 25.29     |           |
| [#443](https://github.com/insight-platform/Savant/issues/443) | 92.73  | 25.32     |           |
| [#352](https://github.com/insight-platform/Savant/issues/352) | 270.99 | 35.90?    | 42.24     |
| [#550](https://github.com/insight-platform/Savant/issues/550) | 272.64 |           | 41.86     |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 272.78 |           | 42.47     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 276.78 |           |           |
| [#705](https://github.com/insight-platform/Savant/issues/705) | 309.99 |           | 65.57     |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 311.85 |           | 65.53     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 310.04 |           | 62.57     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 307.23 |           | 62.52     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 307.14 |           |      |

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
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 475.19 | 64.33?    | 133.48    |
| [#550](https://github.com/insight-platform/Savant/issues/550)                   | 519.29 |           | 144.69    |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 530.02 |           | 142.89    |
| [#641](https://github.com/insight-platform/Savant/issues/641)                   | 502.43 |           | 141.95    |
| [#783](https://github.com/insight-platform/Savant/issues/783)                   | 460.95 |           | 138.32    |
| [#785](https://github.com/insight-platform/Savant/issues/785)                   | 461.14 |           | 138.96    |
| [#679](https://github.com/insight-platform/Savant/issues/679)                   | 455.11 |           | 137.86    |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 450.66 |           |      |

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
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 740.03 | 102.53?   | 130.39    |
| [#550](https://github.com/insight-platform/Savant/issues/550)                   | 749.55 |           | 128.76    |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 748.87 |           | 126.41    |
| [#641](https://github.com/insight-platform/Savant/issues/641)                   | 750.11 |           | 144.6     |
| [#783](https://github.com/insight-platform/Savant/issues/783)                   | 708.36 |           | 140       |

### opencv_cuda_bg_remover_mog2 (with auxiliary stream)


| Savant ver.                                                   | A4000   |
|---------------------------------------------------------------|---------|
| [#785](https://github.com/insight-platform/Savant/issues/785) | 1053.21 |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 1032.96 |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 1025.74 |           |      |

### opencv_cuda_bg_remover_mog2 (multi-stream)

| Savant ver.                                                                     | A4000  | Xavier NX |
|---------------------------------------------------------------------------------|--------|-----------|
| [#372](https://github.com/insight-platform/Savant/issues/372)                   | 510.82 | 91.62     |
| [#372](https://github.com/insight-platform/Savant/issues/372) (queue length 10) | 510.97 | 93.20     |
| [#443](https://github.com/insight-platform/Savant/issues/443)                   | 595.70 | 89.13     |
| [#443](https://github.com/insight-platform/Savant/issues/443) (queue length 10) | 598.71 | 87.69     |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 752.63 |           |
| [#612](https://github.com/insight-platform/Savant/issues/612) (queue length 10) | 748.87 |           |

### panoptic_driving_perception

| Savant ver.                                                   | A4000 | Xavier NX | Orin Nano |
|---------------------------------------------------------------|-------|-----------|-----------|
| [#612](https://github.com/insight-platform/Savant/issues/612) | 62.78 |           | 10.07     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 60.64 |           |           |
| [#705](https://github.com/insight-platform/Savant/issues/705) | 59.66 |           | 10.51     |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 58.67 |           |           |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 49.33 |           |           |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 50.19 |           |  |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 49.78 |           |      |

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
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 414.61 | 77.47?    | 117.53    |
| [#550](https://github.com/insight-platform/Savant/issues/550)                   | 416.32 |           | 117.34    |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 430.29 |           | 117.03    |
| [#641](https://github.com/insight-platform/Savant/issues/641)                   | 430.91 |           | 111.75    |
| [#783](https://github.com/insight-platform/Savant/issues/783)                   | 389.63 |           | 112.17    |
| [#785](https://github.com/insight-platform/Savant/issues/785)                   | 388.26 |           | 112.07    |
| [#679](https://github.com/insight-platform/Savant/issues/679)                   | 402.21 |           | 111.92    |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 397.19 |           |      |

### RTDETR R50

| Savant ver.                                                   | A4000  | Xavier NX | Orin Nano |
|---------------------------------------------------------------|--------|-----------|-----------|
| [#558](https://github.com/insight-platform/Savant/issues/558) | 137.41 |           |           |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 134.47 |           |           |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 119.21 |           |           |
| [#718](https://github.com/insight-platform/Savant/issues/718) | 119.21 |           | 25        |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 114.14 |           | 25.41     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 116.72 |           | 25.45     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 117.00 |           | 25.46     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 116.93 |           |      |

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
| [#352](https://github.com/insight-platform/Savant/issues/352)                   | 256.09 | 54.74?    | 41.02     |
| [#550](https://github.com/insight-platform/Savant/issues/550)                   | 255.94 |           | 41.01     |
| [#612](https://github.com/insight-platform/Savant/issues/612)                   | 261.18 |           | 41.03     |
| [#641](https://github.com/insight-platform/Savant/issues/641)                   | 260.02 |           | 43.17     |
| [#783](https://github.com/insight-platform/Savant/issues/783)                   | 255.72 |           | 43.15     |
| [#785](https://github.com/insight-platform/Savant/issues/785)                   | 259.70 |           | 43.14     |
| [#679](https://github.com/insight-platform/Savant/issues/679)                   | 256.11 |           | 43.21     |
| [#852](https://github.com/insight-platform/Savant/issues/852) | 267.13 |           |      |

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
| [#352](https://github.com/insight-platform/Savant/issues/352) | 85.69 | 24.73?    | 35.96     |
| [#550](https://github.com/insight-platform/Savant/issues/550) | 91.04 |           | 36.01     |
| [#612](https://github.com/insight-platform/Savant/issues/612) | 91.65 |           | 36.11     |
| [#641](https://github.com/insight-platform/Savant/issues/641) | 92.78 |           | 37.33     |
| [#783](https://github.com/insight-platform/Savant/issues/783) | 90.71 |           | 37.38     |
| [#785](https://github.com/insight-platform/Savant/issues/785) | 76.90 |           | 35.86     |
| [#679](https://github.com/insight-platform/Savant/issues/679) | 77.33 |           | 35.91     |
