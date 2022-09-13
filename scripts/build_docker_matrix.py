#!/usr/bin/env python3
"""Build `matrix` for Savant docker images."""
import json

matrix = {
    'version': ['0.1.0-6.1-base', '0.1.0-6.1-samples'],
    'arch': ['linux/amd64', 'linux/arm64'],
}
print(json.dumps(matrix))
