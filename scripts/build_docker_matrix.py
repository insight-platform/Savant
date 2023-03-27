#!/usr/bin/env python3
"""Build `matrix` for Savant docker images."""
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from savant.utils.version import version  # noqa: F401

ARCH_OPTIONS = ['linux/amd64', 'linux/arm64']  # dGPU/L4T

IMAGE_OPTIONS = [
    'deepstream',
    'adapters-deepstream',
    'adapters-gstreamer',
    'adapters-py',
]

DEEPSTREAM_OPTIONS = [version.DEEPSTREAM]
DEEPSTREAM_L4T_OPTIONS = [version.DEEPSTREAM]

TAG_SUFFIX_OPTIONS = ['base', 'samples']

matrix_include = []
for arch in ARCH_OPTIONS:
    for image in IMAGE_OPTIONS:
        items = []
        with_deepstream = 'deepstream' in image
        is_adapter = 'adapters' in image

        deepstream_options = DEEPSTREAM_OPTIONS
        if arch == 'linux/arm64':
            deepstream_options = DEEPSTREAM_L4T_OPTIONS

        tag_suffix_options = ['']
        if 'adapters' not in image:
            tag_suffix_options = TAG_SUFFIX_OPTIONS

        tag_prefix_options = [version.SAVANT]
        if with_deepstream:
            tag_prefix_options = []
            for deepstream_version in deepstream_options:
                tag_prefix_options.append(version.SAVANT + '-' + deepstream_version)

        for tag_prefix in tag_prefix_options:
            for tag_suffix in tag_suffix_options:
                tag = tag_prefix
                if tag_suffix:
                    tag += '-' + tag_suffix

                docker_image = 'savant-' + image
                docker_file = 'deepstream' if with_deepstream else image

                if arch == 'linux/arm64':
                    docker_image += '-l4t'
                    if with_deepstream:
                        docker_file += '-l4t'

                target = ''
                if with_deepstream:
                    if is_adapter:
                        target = 'adapters'
                    elif 'samples' in tag:
                        target = 'samples'
                    else:
                        target = 'base'

                deepstream_version = version.DEEPSTREAM

                items.append(
                    {
                        'arch': arch,
                        'docker_image': docker_image + ':' + tag,
                        'docker_file': 'docker/Dockerfile.' + docker_file,
                        'target': target,
                        'deepstream_version': deepstream_version,
                    }
                )

        matrix_include += items

# print(json.dumps({'include': matrix_include}, indent=2))
# print('TOTAL:', len(matrix_include))
print(json.dumps({'include': matrix_include}))
