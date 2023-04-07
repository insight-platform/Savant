# NVIDIA Car Classification app

## Reproduce [deepstream-test2 app](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2)

The easiest way is to use the model configs provided in the source application repository.

In order to prepare DS model config files from original repository it is necessary to cut relative path component from file paths, leaving only file name.
E.g. in `dstest2_pgie_config.txt`:

* model-file `../../../../samples/models/Primary_Detector/resnet10.caffemodel` => `resnet10.caffemodel`
* proto-file `../../../../samples/models/Primary_Detector/resnet10.prototxt` => `resnet10.prototxt`
* model-engine-file `../../../../samples/models/Primary_Detector/resnet10.caffemodel_b1_gpu0_int8.engine` => `resnet10.caffemodel_b1_gpu0_int8.engine`
* labelfile-path `../../../../samples/models/Primary_Detector/labels.txt` => `labels.txt`
* int8-calib-file `../../../../samples/models/Primary_Detector/cal_trt.bin` => `cal_trt.bin`

Resulting config files are provided in the samples directory.

### Module config file with DeepStream configs

An example of minimal module configuration is provided in the [module.yml](module.yml).

Run this sample config with

```bash
python scripts/run_module.py samples/nvidia_car_classification/module.yml
```

#### Detector

```yaml
    - element: nvinfer@detector
      name: Primary_Detector
      model:
        format: caffe
        config_file: ${oc.env:APP_PATH}/samples/nvidia_car_classification/dstest2_pgie_config.txt
```

#### Tracker

Configuration properties copied from [dstest2_tracker_config.txt](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-test2/dstest2_tracker_config.txt)

```yaml
    - element: nvtracker
      properties:
        tracker-width: 640
        tracker-height: 384
        ll-lib-file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
        ll-config-file: ${oc.env:APP_PATH}/samples/nvidia_car_classification/config_tracker_NvDCF_perf.yml
        enable_batch_process: 1
```

#### Classifiers

For `Secondary_CarColor` classifier use `dstest2_sgie1_config.txt`

```yaml
    - element: nvinfer@classifier
      name: Secondary_CarColor
      model:
        format: caffe
        config_file: ${oc.env:APP_PATH}/samples/nvidia_car_classification/dstest2_sgie1_config.txt
        input:
          object: Primary_Detector.Car
        output:
          attributes:
            - name: color
```

Likewise,

* For `Secondary_CarMake` use `dstest2_sgie2_config.txt`
* For `Secondary_VehicleTypes` use `dstest2_sgie3_config.txt`

*Note: If you are using files from deepstream_python_apps repository, then check that
there are no duplicate property values, otherwise the launch will fail - python configparser will produce an error.

## Minimal configuration using prepared engines

In case TRT engines for supported Deepstream version have already been prepared,
it is possible to use them in the module configuration directly.
The only difference from the previous configuration is to replace the
`config_file` with the `engine_file` parameter for each model and specify the required pre/post processing parameters.

For detector you can specify `label_file` instead of `output.objects`.

### Module config file with prepared engines

An example of TRT engines usage is provided in the [module-engines-config.yml](module-engines-config.yml).

Run this sample config with

```bash
python scripts/run_module.py samples/nvidia_car_classification/module-engines-config.yml
```

## Full configuration using only Savant config parameters

Minimal changes are needed to create a module configuration without relying on nvinfer config files and/or generated TRT engines.

### Module config file

An example of config file that will allows to convert caffe models into TRT engines using only Savant config parameters
is provided in the [module-full-savant-config.yml](module-full-savant-config.yml).

Run this sample config with

```bash
python scripts/run_module.py samples/nvidia_car_classification/module-full-savant-config.yml
```

## Module configuration using etlt models

It is also simple enough to swap the models in the pipeline for other ones. For example,
replace the sample models from the DeepStream SDK with TAO models available from [Nvidia NGC](https://catalog.ngc.nvidia.com/).

### Module config file with etlt models

Config file that uses etlt models for this sample is provided in the [module-etlt-config.yml](module-etlt-config.yml).

Run this sample config with

```bash
python scripts/run_module.py samples/nvidia_car_classification/module-etlt-config.yml
```
