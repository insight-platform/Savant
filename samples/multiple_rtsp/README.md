# Multiple RTSP Streams Demo

A simple pipeline demonstrates how multiplexed processing works in Savant. In the demo, two RTSP streams are ingested in the module and processed with the PeopleNet model. 

The resulting streams can be accessed via LL-HLS on `http://locahost:888/stream/city-traffic` and `http://locahost:888/stream/town-centre`.

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

./scripts/run_module.py --build-engines samples/peoplenet_detector/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/multiple_rtsp/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/multiple_rtsp/docker-compose.l4t.yml up

# visit 'http://127.0.0.1:888/stream/city-traffic' and 'http://127.0.0.1:888/stream/town-centre' to see how it works

# Ctrl+C to stop running the compose bundle
```
