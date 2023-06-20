# Performance measurements

## Hardware

| Name      | CPU                                | GPU              | RAM  |
| --------- | ---------------------------------- | ---------------- | ---- |
| Setup_1   | AMD Ryzen 7 3700X 8-Core Processor | NVIDIA RTX A4000 | 62Gi |
| Jetson NX |                                    |                  |      |

## Pipeline FPS

| Hardware  | Sample                      | Savant ver. | FPS |
| --------- | --------------------------- | ----------- | --- |
| Setup_1   | nvidia_car_classification   | v0.2.3      | 155 |
| Setup_1   | nvidia_car_classification   | latest      | 158 |
| Setup_1   | peoplenet_detector          | v0.2.3      | 125 |
| Setup_1   | peoplenet_detector          | latest      | 113 |
| Setup_1   | traffic_meter-yolov8m       | v0.2.3      | 132 |
| Setup_1   | traffic_meter-yolov8m       | latest      | 138 |
| Setup_1   | traffic_meter-peoplenet     | v0.2.3      | 118 |
| Setup_1   | traffic_meter-peoplenet     | latest      | 127 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | v0.2.3      | 675 |
| Setup_1   | opencv_cuda_bg_remover_mog2 | latest      | 670 |
| Jetson NX | nvidia_car_classification   | v0.2.3      | 42  |
| Jetson NX | nvidia_car_classification   | latest      |     |
| Jetson NX | peoplenet_detector          | v0.2.3      | 30  |
| Jetson NX | peoplenet_detector          | latest      |     |
| Jetson NX | traffic_meter-yolov8m       | v0.2.3      | 23  |
| Jetson NX | traffic_meter-yolov8m       | latest      |     |
| Jetson NX | traffic_meter-peoplenet     | v0.2.3      | 28  |
| Jetson NX | traffic_meter-peoplenet     | latest      |     |
| Jetson NX | opencv_cuda_bg_remover_mog2 | v0.2.3      | 65  |
| Jetson NX | opencv_cuda_bg_remover_mog2 | latest      |     |
