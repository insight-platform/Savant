# Savant module template

Start your own project with this template.

## Develop with IDE

Copy the template, rename. Open template with your favorite IDE.

### PyCharm Professional
1. [Configure a docker interpreter](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html) using savant-deepstream container `ghcr.io/insight-platform/savant-deepstream:latest`.
2. Setup Run Configuration for `run.py`:
 - add `--gpus=all` in Docker container settings
 - add `PYTHONPATH=/opt/savant` to Environment variables ([PyCharm rewrite PYTHONPATH](https://youtrack.jetbrains.com/issue/PY-32618/The-original-PYTHONPATH-is-replaced-by-PyCharm-when-running-configurations-using-Docker-interpreter))
3. Run `run.py` to check.

### PyCharm Community
TBD

There is no possibility to use docker as an interpreter, so we need the Savant.

### VSCode
TBD
