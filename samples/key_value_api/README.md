# Key-Value Storage API Demonstration

A pipeline demonstrating how to work with the embedded Key-Value store.

Tested on platforms:

- Nvidia Turing, Ampere


## How To Run The Demo

The demo implemented only for X86, for Jetson the use is the same, consult other samples for specifics.

```bash
# if x86
docker compose -f samples/key_value_api/docker-compose.x86.yml up
```

## How To Access The Key-Value Store With REST API

The key-value store is accessible via REST API. Use the script to read from the store:

```bash
# retrieve the value of the key 'frame_counter' through HTTP API request
docker compose -f samples/key_value_api/docker-compose.x86.yml exec -it first python /scripts/get_frame_counter.py
```

The documentation for the Key-Value API is available at the Savant documentation [website](https://docs.savant-ai.io/develop/advanced_topics/15_embedded_kvs.html).

## How to Access the Key-Value Subscription With REST API

The key-value store is accessible via REST API. To test how to subscribe to the key-value store, 
use the [websocat](https://github.com/vi/websocat) tool or similar:

Only metadata in JSON format:

```bash
websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/meta
```

Metadata in JSON and Attributes in binary format:

```bash
# be aware that the terminal may not display the binary data correctly
websocat -U --ping-interval 1 --ping-timeout 2 ws://localhost:8080/kvs/events/full
```

The binary data contains Attributes in the protobuf-serialized format. Take a look at the [script](scripts/get_frame_counter.py) showing how to deserialize them.