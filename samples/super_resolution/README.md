# Super Resolution

The pipeline shows the use of super-resolution neural networks to improve the quality and resolution of the input video. 

We prepared ONNX versions (using standard `torch.onnx.export`) of two lightweight NinaSR models ninasr_b0 and ninasr_b1 with different scales (x2/3/4) from the [TorchSR repository](https://github.com/Coloquinte/torchSR). The demo uses input video in 360p format and model ninasr_b0 with scale 3 to output video in 1080p format. The model and scale can be changed in the module configuration [file](module.yml).

The demo is prepared for use on dGPU. Separately, the module can be run on the Jetson platform (Xavier, Orin). 

Preview:

![](assets/shuffle_dance_360p_1080p.webp)


Tested on platforms:
- Nvidia Turing, Ampere.

## Run the demo

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/super_resolution

../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
