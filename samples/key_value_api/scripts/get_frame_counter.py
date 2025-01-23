#!/usr/bin/env python3

import requests
import savant_rs.webserver.kvs as kvs

response = requests.get(f'http://localhost:8080/kvs/get/counter/frame_counter')
assert response.status_code == 200
attributes = kvs.deserialize_attributes(response.content)
if len(attributes) == 1:
    attr = attributes[0]
    print(attr)
    value = attr.values[0].as_integer()
    print(f'Got value: {value}')
