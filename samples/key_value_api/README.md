# Key-Value Storage API Demonstration

A pipeline demonstrating how to work with the embedded Key-Value store.

Tested on platforms:

- Nvidia Turing, Ampere
- Nvidia Jetson Orin family


## How To Run The Demo

```bash
# if x86
docker compose -f samples/key_value_api/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/key_value_api/docker-compose.l4t.yml up
```

## How To Access The Key-Value Store With REST API

The key-value store is accessible via REST API. Use the script to read and write data to the store:

```bash
# set key-value pair
./scripts/set_float_attribute_with_ttl.py namespace attribute 0.1 ttl
./scripts/get_float_with_ttl.py namespace attribute
./scripts/get_frame_counter.py
./scripts/search_attributes.py
```

