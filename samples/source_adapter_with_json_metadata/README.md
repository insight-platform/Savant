# Using json medadata in source adapter

A simple pipeline demonstrates how you can add metadata to input frames using source 
adapter. In the demo, ground truth boxes are added to images 
from the COCO dataset, and a simple function evaluates IOU.

In the demo it is assumed that there is only one person in the picture and 
the IOU of the true box and from the Yolo detection model is calculated. 
The IOU value is added as a tag to the frame metadata.

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Build Engines

The demo uses models that are compiled into TensorRT engines the first time the demo is run. This takes time. Optionally, you can prepare the engines before running the demo by using the command:

```bash
# you are expected to be in Savant/ directory

./scripts/run_module.py --build-engines samples/traffic_meter/module.yml
```

## Download Data

```bash
# you are expected to be in Savant/ directory

mkdir -p data 
wget -P data https://eu-central-1.linodeobjects.com/savant-data/demo/source_adapter_with_json_metadata.zip
unzip data/source_adapter_with_json_metadata.zip -d data
rm data/source_adapter_with_json_metadata.zip
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/source_adapter_with_json_metadata/docker-compose.x86.yml up

# if Jetson. The nvv4l2decoder has a bug in the nvv4l2decoder on 
# the Jetson platform so the example currently does not work correctly on that platform.
# https://github.com/insight-platform/Savant/issues/314

# docker compose -f samples/source_adapter_with_json_metadata/docker-compose.l4t.yml up module image-json-sink

# Ctrl+C to stop running the compose bundle
```

**Note**: The source adapter runs on the images directory, so when it sends all the images it will terminate.
The module and sink adapter run the whole time, so they should be stopped manually.

Results will be saved in the `data/results` folder.

You can use the convert_coco_to_savant.py script as a starting point to prepare 
your input metadata. This script reads data from the COCO dataset annotations, 
selects only objects with "person" label, and converts it into an input data format 
for the framework. A detailed description of the input JSON file format with metadata 
for the adapter is described in the documentation ([link](https://docs.savant-ai.io/advanced_topics/9_input_json_metadata.html)). 
Install the pycocotools library before running it.

```bash
pip install pycocotools
```
