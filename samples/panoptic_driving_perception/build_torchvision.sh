#!/bin/bash

cd /opt/torchvision
python3 setup.py bdist_wheel
cp /opt/torchvision/dist/torchvision*.whl /torchvision