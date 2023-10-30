# AnimeGAN demo

Use [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2) to apply Hayao Miyazaki anime-like style to a video.

Preview:

![](assets/animegan-loop.webp)

Tested on platforms:

- Nvidia Turing, Ampere.

Demonstrated adapters:

- Multi-stream source adapter;
- Video/Metadata sink adapter.

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Download sample video:

```bash
# you are expected to be in Savant/ directory

mkdir -p data
curl -o data/deepstream_sample_720p.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/deepstream_sample_720p.mp4
```

Run the demo:

```bash
# you are expected to be in Savant/ directory

./samples/animegan/run.sh
```

## Change input video

Input video is expected to be located in the `Savant/data` directory. File name can be set through `INPUT_FILENAME` environment variable

```bash
INPUT_FILENAME=input.mp4 ./samples/animegan/run.sh
```

## Prepare other weights

The sample uses [generator_Hayao_weight](https://github.com/TachibanaYoshino/AnimeGANv2/tree/master/checkpoint/generator_Hayao_weight) AnimeGANv2 checkpoint that was:

1. Converted to Pytorch with the help of [convert_weights.py](https://github.com/bryandlee/animegan2-pytorch/blob/main/convert_weights.py) script.
2. Exported to ONNX using standard Pytorch methods with (dynamic, 3, 720, 1280) inference dimensions.
3. Simplified using [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
