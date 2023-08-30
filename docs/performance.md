# Performance measurements

## Hardware

| Name      | CPU                                | GPU              | RAM, Gi |
| --------- | ---------------------------------- | ---------------- | ------- |
| A4000     | AMD Ryzen 7 3700X 8-Core Processor | NVIDIA RTX A4000 | 62      |
| Jetson NX | MODE 20W, 6 CORE, Clocks On        |                  | 8       |

## Pipeline FPS

### age_gender_recognition

| Savant ver.                                                                      | A4000  | Jetson NX |
|----------------------------------------------------------------------------------|--------|-----------|
| [#166](https://github.com/insight-platform/Savant/issues/166)                    | 261    | 37        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 255.73 | 41.08     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 257.52 | 40.78     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 255.25 | 40.01     |
| [#290](https://github.com/insight-platform/Savant/issues/290)                    | 253.25 | 40.05     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 253.44 | 35.54     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 260.56 | 39.19     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 251.76 | 38.98     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 250.58 | 38.75     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 244.93 | 39.19     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 241.40 | 38.59     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 233.78 | 37.70     |

### conditional_video_processing

| Savant ver.                                                                      | A4000  | Jetson NX |
| -------------------------------------------------------------------------------- | ------ |-----------|
| [#244](https://github.com/insight-platform/Savant/issues/244)                    | 327    | 64        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 325.64 | 64.90     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 314.84 | 64.13     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 318.96 | 63.20     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 323.65 | 63.02     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 324.69 | 63.51     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 284.06 | 60.10     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 281.90 | 60.11     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 315.22 | 63.87     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 288.91 | 61.60     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 263.44 | 59.86     |

### face_reid

| Savant ver.                                                   | A4000  | Jetson NX |
|---------------------------------------------------------------|--------|-----------|
| [#142](https://github.com/insight-platform/Savant/issues/142) | 124    | 26        |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 118.79 | 25.99     |

### nvidia_car_classification

| Savant ver.                                                                      | A4000  | Jetson NX |
| -------------------------------------------------------------------------------- | ------ |-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 155    | 42        |
| [#207](https://github.com/insight-platform/Savant/issues/207)                    | 158    | 42        |
| [#208](https://github.com/insight-platform/Savant/issues/208)                    | 156    | 43        |
| [#84](https://github.com/insight-platform/Savant/issues/84)                      | 155    | 42        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 161.61 | 45.43     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 160.92 | 45.11     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 163.31 | 44.85     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 161.81 | 38.15     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 163.02 | 44.09     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 152.65 | 43.40     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 153.99 | 43.17     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 157.49 | 44.16     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 156.72 | 43.24     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 151.44 | 42.97     |

### opencv_cuda_bg_remover_mog2

| Savant ver.                                                                      | A4000  | Jetson NX |
| -------------------------------------------------------------------------------- | ------ | --------- |
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 675    | 65        |
| [#207](https://github.com/insight-platform/Savant/issues/207)                    | 670    | 60        |
| [#208](https://github.com/insight-platform/Savant/issues/208)                    | 662    | 60        |
| [#84](https://github.com/insight-platform/Savant/issues/84)                      | 663    | 62        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 689.91 | 87.73     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 686.05 | 87.46     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 686.13 | 87.84     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 694.47 | 83.66     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 689.34 | 86.16     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 530.94 | 86.68     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 575.76 | 87.05     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 669.15 | 93.14     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 671.76 | 92.41     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 608.54 | 91.69     |

### peoplenet_detector

| Savant ver.                                                                      | A4000  | Jetson NX |
| -------------------------------------------------------------------------------- | ------ | --------- |
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 125    | 30        |
| [#207](https://github.com/insight-platform/Savant/issues/207)                    | 113    | 30        |
| [#208](https://github.com/insight-platform/Savant/issues/208)                    | 113    | 30        |
| [#84](https://github.com/insight-platform/Savant/issues/84)                      | 111    | 29        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 118.68 | 28.97     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 118.56 | 29.53     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 118.23 | 30.09     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 119.12 | 28.93     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 119.79 | 30.57     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 102.02 | 26.38     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 109.95 | 26.58     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 116.35 | 29.31     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 114.51 | 29.46     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 116.43 | 28.05     |

### traffic_meter (yolov8m)

| Savant ver.                                                                      | A4000  | Jetson NX |
| -------------------------------------------------------------------------------- | ------ | --------- |
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3)                 | 132    | 23        |
| [#207](https://github.com/insight-platform/Savant/issues/207)                    | 138    | 24        |
| [#208](https://github.com/insight-platform/Savant/issues/208)                    | 136    | 25        |
| [#84](https://github.com/insight-platform/Savant/issues/84)                      | 136    | 24        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4)                 | 140.43 | 23.62     |
| [#61](https://github.com/insight-platform/Savant/issues/61)                      | 137.85 | 23.80     |
| [#285](https://github.com/insight-platform/Savant/issues/285)                    | 139.25 | 24.49     |
| [#229](https://github.com/insight-platform/Savant/issues/229)                    | 139.67 | 23.40     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (no queues)        | 139.75 | 26.00     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 1)  | 115.21 | 24.97     |
| [#322](https://github.com/insight-platform/Savant/issues/322) (buffer length 10) | 114.61 | 25.19     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (no queues)        | 137.62 | 26.59     |
| [#334](https://github.com/insight-platform/Savant/issues/334) (buffer length 10) | 123.98 | 25.61     |
| [#347](https://github.com/insight-platform/Savant/issues/347) (no queues)        | 136.49 | 24.40     |

### yolov8_seg

Note: `yolov8_seg` always has a buffer length of 10. `BUFFER_QUEUES` env doesn't affect it.

| Savant ver.                                                   | A4000 | Jetson NX |
|---------------------------------------------------------------|-------|-----------|
| [#131](https://github.com/insight-platform/Savant/issues/131) | 45.78 | 14.84     |
| [#334](https://github.com/insight-platform/Savant/issues/334) | 44.33 | 14.82     |
| [#347](https://github.com/insight-platform/Savant/issues/347) | 44.34 | 13.07     |
