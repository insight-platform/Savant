# Bypass Model

The pipeline shows the usage of the bypass model, which is a simple model that does not perform any computation on the input data. 
It demonstrates how data is pre-processed before being processed by the model and can be used for troubleshooting.

Identity model is prepared by converting `nn.Identity` PyTorch model to ONNX format (using standard `torch.onnx.export`).

The demo is prepared for use on dGPU.

Tested on platforms:
- Nvidia Turing
- Nvidia Jetson Orin Nano

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

./scripts/run_module.py --build-engines samples/bypass_model/demo.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/bypass_model/docker-compose.x86.yml up -d

# if Jetson
docker compose -f samples/bypass_model/docker-compose.l4t.yml up -d

# check docker logs of the module container to see pre-processed data
docker logs -f bypass_model-module-1

# Ctrl+C to stop running the compose bundle
```
