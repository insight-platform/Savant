# Kafka-Redis Adapters Demo

A pipeline demonstrates how Kafka-Redis adapters works in Savant. In the demo video from Video Loop Source adapter passed to Kafka-Redis Sink adapter. Kafka-Redis Sink adapter saves frame content to KeyDB (alternative of Redis) and metadata to Kafka. Kafka-Redis Source adapter reads metadata from Kafka and frame content from KeyDB and passes it to the module. Then the module passes the processed frames to Always-On-RTSP Sink adapter.

![kafka-redis-adapter-demo.png](assets/kafka-redis-adapter-demo.png)

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
docker compose -f samples/kafka_redis_adapter/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/kafka_redis_adapter/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

[docker-compose-no-keydb.x86.yml](docker-compose-no-keydb.x86.yml) and [docker-compose-no-keydb.l4t.yml](docker-compose-no-keydb.l4t.yml) contain the sample without using KeyDB. The frame content is stored internally in the video frame.
