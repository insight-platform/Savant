# Performance measurements

## Hardware

| Name      | CPU                                | GPU              | RAM, Gi |
|-----------|------------------------------------|------------------|---------|
| A4000     | AMD Ryzen 7 3700X 8-Core Processor | NVIDIA RTX A4000 | 62      |
| Jetson NX | MODE 20W, 6 CORE, Clocks On        |                  | 8       |

## Pipeline FPS

### age_gender_recognition

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [#166](https://github.com/insight-platform/Savant/issues/166)    | 261    | 37        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 255.73 | 41.08     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 257.52 | 40.78     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 255.25 | 40.01     |

### conditional_video_processing

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [#244](https://github.com/insight-platform/Savant/issues/244)    | 327    | 64        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 325.64 | 64.90     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 314.84 | 64.13     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 318.96 | 63.20     |

### nvidia_car_classification

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3) | 155    | 42        |
| [#207](https://github.com/insight-platform/Savant/issues/207)    | 158    | 42        |
| [#208](https://github.com/insight-platform/Savant/issues/208)    | 156    | 43        |
| [#84](https://github.com/insight-platform/Savant/issues/84)      | 155    | 42        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 161.61 | 45.43     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 160.92 | 45.11     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 163.31 | 44.85     |

### opencv_cuda_bg_remover_mog2

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3) | 675    | 65        |
| [#207](https://github.com/insight-platform/Savant/issues/207)    | 670    | 60        |
| [#208](https://github.com/insight-platform/Savant/issues/208)    | 662    | 60        |
| [#84](https://github.com/insight-platform/Savant/issues/84)      | 663    | 62        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 689.91 | 87.73     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 686.05 | 87.46     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 686.13 | 87.84     |

### peoplenet_detector

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3) | 125    | 30        |
| [#207](https://github.com/insight-platform/Savant/issues/207)    | 113    | 30        |
| [#208](https://github.com/insight-platform/Savant/issues/208)    | 113    | 30        |
| [#84](https://github.com/insight-platform/Savant/issues/84)      | 111    | 29        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 118.68 | 28.97     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 118.56 | 29.53     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 118.23 | 30.09     |

### traffic_meter (yolov8m)

| Savant ver.                                                      | A4000  | Jetson NX |
|------------------------------------------------------------------|--------|-----------|
| [v0.2.3](https://github.com/insight-platform/Savant/tree/v0.2.3) | 132    | 23        |
| [#207](https://github.com/insight-platform/Savant/issues/207)    | 138    | 24        |
| [#208](https://github.com/insight-platform/Savant/issues/208)    | 136    | 25        |
| [#84](https://github.com/insight-platform/Savant/issues/84)      | 136    | 24        |
| [v0.2.4](https://github.com/insight-platform/Savant/tree/v0.2.4) | 140.43 | 23.62     |
| [#61](https://github.com/insight-platform/Savant/issues/61)      | 137.85 | 23.80     |
| [#285](https://github.com/insight-platform/Savant/issues/285)    | 139.25 | 24.49     |
