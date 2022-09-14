# PeopleNet detector

This is a sample application with one regular detector.

It takes a streaming video as input and performs people detection using
[PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_peoplenet) model from NGC.

Module is configured to download model files from NVIDIA remote.
It is possible to pick a different model version by commenting/uncommenting lines in the config [module.yml](module.yml).

Run this sample with

```bash
python scripts/run_module.py samples/peoplenet_detector/module.yml
```
