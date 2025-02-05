#!/bin/bash
# This script used in the Dockerfile to cythonize the source files.
# Should be run in the root of the project directory (eg /opt/savant).

set -x

python3 -m pip install --no-cache-dir cython

if [ -d "savant" ]; then
  find savant -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
  find savant -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
fi

if [ -d "adapters" ]; then
  exclude_dirs="adapters/gst/gst_plugins adapters/gst/sinks adapters/python/bridge adapters/python/sinks adapters/python/sources"
  exclude_dirs=${exclude_dirs// / -o -path }
  find adapters \( -path $exclude_dirs \) -prune -o -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
  find adapters \( -path $exclude_dirs \) -prune -o -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
fi

if [ -d "watchdog" ]; then
  find watchdog -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
  find watchdog -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
fi

find . -name "*.c" -type f -delete
find . -name "build" -type d -print0 | xargs -0 rm -rf
find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf
