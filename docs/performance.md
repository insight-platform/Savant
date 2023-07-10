# Performance measurements

## Hardware

| Name      | CPU                                | GPU              | RAM  |
| --------- | ---------------------------------- | ---------------- | ---- |
| Setup_1   | AMD Ryzen 7 3700X 8-Core Processor | NVIDIA RTX A4000 | 62Gi |
| Jetson NX |                                    |                  |      |

## Pipeline FPS

| Hardware  | Sample                      | Savant ver.                                                   | FPS |
|-----------|-----------------------------|---------------------------------------------------------------|-----|
| Setup_1   | nvidia_car_classification   | v0.2.3                                                        | 155 |
| Setup_1   | nvidia_car_classification   | [#207](https://github.com/insight-platform/Savant/issues/207) | 158 |
| Setup_1   | nvidia_car_classification   | [#208](https://github.com/insight-platform/Savant/issues/208) | 156 |
| Setup_1   | nvidia_car_classification   | [#84](https://github.com/insight-platform/Savant/issues/84)   | 155 |
| Setup_1   | peoplenet_detector          | v0.2.3                                                        | 125 |
| Setup_1   | peoplenet_detector          | [#207](https://github.com/insight-platform/Savant/issues/207) | 113 |
| Setup_1   | peoplenet_detector          | [#208](https://github.com/insight-platform/Savant/issues/208) | 113 |
| Setup_1   | peoplenet_detector          | [#84](https://github.com/insight-platform/Savant/issues/84)   | 111 |
| Setup_1   | traffic_meter-yolov8m       | v0.2.3                                                        | 132 |
| Setup_1   | traffic_meter-yolov8m       | [#207](https://github.com/insight-platform/Savant/issues/207) | 138 |
| Setup_1   | traffic_meter-yolov8m       | [#208](https://github.com/insight-platform/Savant/issues/208) | 136 |
| Setup_1   | traffic_meter-yolov8m       | [#84](https://github.com/insight-platform/Savant/issues/84)   | 136 |
| Setup_1   | traffic_meter-peoplenet     | v0.2.3                                                        | 118 |
| Setup_1   | traffic_meter-peoplenet     | [#207](https://github.com/insight-platform/Savant/issues/207) | 127 |
| Setup_1   | traffic_meter-peoplenet     | [#208](https://github.com/insight-platform/Savant/issues/208) | 125 |
| Setup_1   | traffic_meter-peoplenet     | [#84](https://github.com/insight-platform/Savant/issues/84)   | 125 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | v0.2.3                                                        | 675 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | [#207](https://github.com/insight-platform/Savant/issues/207) | 670 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | [#208](https://github.com/insight-platform/Savant/issues/208) | 662 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | [#84](https://github.com/insight-platform/Savant/issues/84)   | 663 |
| Setup_1   | age_gender_recognition      | [#166](https://github.com/insight-platform/Savant/issues/166) | 261 |
| Jetson NX | nvidia_car_classification   | v0.2.3                                                        | 42  |
| Jetson NX | nvidia_car_classification   | [#207](https://github.com/insight-platform/Savant/issues/207) | 42  |
| Jetson NX | nvidia_car_classification   | [#208](https://github.com/insight-platform/Savant/issues/208) | 43  |
| Jetson NX | nvidia_car_classification   | [#84](https://github.com/insight-platform/Savant/issues/84)   | 42  |
| Jetson NX | peoplenet_detector          | v0.2.3                                                        | 30  |
| Jetson NX | peoplenet_detector          | [#207](https://github.com/insight-platform/Savant/issues/207) | 30  |
| Jetson NX | peoplenet_detector          | [#208](https://github.com/insight-platform/Savant/issues/208) | 30  |
| Jetson NX | peoplenet_detector          | [#84](https://github.com/insight-platform/Savant/issues/84)   | 29  |
| Jetson NX | traffic_meter-yolov8m       | v0.2.3                                                        | 23  |
| Jetson NX | traffic_meter-yolov8m       | [#207](https://github.com/insight-platform/Savant/issues/207) | 24  |
| Jetson NX | traffic_meter-yolov8m       | [#208](https://github.com/insight-platform/Savant/issues/208) | 25  |
| Jetson NX | traffic_meter-yolov8m       | [#84](https://github.com/insight-platform/Savant/issues/84)   | 24  |
| Jetson NX | traffic_meter-peoplenet     | v0.2.3                                                        | 28  |
| Jetson NX | traffic_meter-peoplenet     | [#207](https://github.com/insight-platform/Savant/issues/207) | 28  |
| Jetson NX | traffic_meter-peoplenet     | [#208](https://github.com/insight-platform/Savant/issues/208) | 27  |
| Jetson NX | traffic_meter-peoplenet     | [#84](https://github.com/insight-platform/Savant/issues/84)   | 27  |
| Jetson NX | opencv_cuda_bg_remover_mog2 | v0.2.3                                                        | 65  |
| Jetson NX | opencv_cuda_bg_remover_mog2 | [#207](https://github.com/insight-platform/Savant/issues/207) | 60  |
| Jetson NX | opencv_cuda_bg_remover_mog2 | [#208](https://github.com/insight-platform/Savant/issues/208) | 60  |
| Jetson NX | opencv_cuda_bg_remover_mog2 | [#84](https://github.com/insight-platform/Savant/issues/84)   | 62  |
| Jetson NX | age_gender_recognition      | [#166](https://github.com/insight-platform/Savant/issues/166) | 37  |