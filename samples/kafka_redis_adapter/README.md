# Kafka-Redis Adapters Demo

A pipeline demonstrates how Kafka-Redis adapters works in Savant. In the demo video from Video Loop Source adapter passed to Kafka-Redis Sink adapter. Kafka-Redis Sink adapter saves frame content to KeyDB (alternative of Redis) and metadata to Kafka. Kafka-Redis Source adapter reads metadata from Kafka and frame content from KeyDB and passes it to the module. Then the module passes the processed frames to Always-On-RTSP Sink adapter.

![kafka-redis-adapter-demo.png](assets/kafka-redis-adapter-demo.png)

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/kafka_redis_adapter
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

[docker-compose-no-keydb.x86.yml](docker-compose-no-keydb.x86.yml) and [docker-compose-no-keydb.l4t.yml](docker-compose-no-keydb.l4t.yml) contain the sample without using KeyDB. The frame content is stored internally in the video frame.
