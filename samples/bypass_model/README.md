# Bypass Model

The pipeline shows the usage of the bypass model, which is a simple model that does not perform any computation on the input data. 
It demonstrates how data is pre-processed before being processed by the model and can be used for troubleshooting.
The demo uses a `maintain_aspect_ratio` flag to show how pre-processed data can be compared with the raw data.

Identity model is prepared by converting `nn.Identity` PyTorch model to ONNX format (using standard `torch.onnx.export`). For more details, please refer to the [export ONNX model](#export-onnx-model) section.

Tested on platforms:
- Nvidia Turing
- Nvidia Jetson Orin family

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

# check docker logs of the module container to see raw data, pre-processed data and the result of the comparison
docker logs -f bypass_model-module-1
```

## Clean Up

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/bypass_model/docker-compose.x86.yml down -v

# if Jetson
docker compose -f samples/bypass_model/docker-compose.l4t.yml down -v
```

## Export ONNX Model

The model input has dynamic tensor axes `batch x 3 x height x width` where `batch`, `height` and `width` are defined at runtime. So you can provide a source of different shapes to the model.
However, if you want to change the shape of the input tensor, you need to modify the `export.py` script accordingly. And then you can export the model using the following command:

The model takes input in the form of dynamic tensor axes `batch x 3 x height x width`, where `batch`, `height`, and `width` are defined at runtime. 
This allows you to provide the model with sources of different shapes. 
If you need to change the shape of the input tensor, you will have to make modifications to the [export.py](export.py) script.
After these modifications, you can export the model using the following command:

```bash
# you are expected to be in Savant/ directory

docker run --rm \
  -v "$(pwd)/samples/bypass_model:/opt/bypass_model" \
  -w /opt/bypass_model \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  ghcr.io/insight-platform/savant-deepstream-extra \
  /opt/bypass_model/export.py

```

When the command is executed, the model definition will be printed in the console output. The model definition will look like this:

```
graph main_graph (
  %input[FLOAT, batchx3xheightxwidth]
) {
  %output = Identity(%input)
  return %output
}
```

After running the command, the ONNX model will be saved in the `samples/bypass_model` directory.
To use the model in the pipeline, you need to change `demo.yml` to replace `remote` section with the following:

```yaml
local_path: /cache/models/bypass_model/identity-local
```

And uncomment the volume mount in the `module` service in [docker-compose.x86.yml](docker-compose.x86.yml) or [docker-compose.l4t.yml](docker-compose.l4t.yml) file.
