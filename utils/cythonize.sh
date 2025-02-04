#!/bin/bash
# This script used in the Dockerfile to cythonize the source files.
# Should be run in the root of the project directory (eg /opt/savant).

set -x

python3 -m pip install --no-cache-dir cython

if [ -d "savant" ]; then
    find savant -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
    find savant -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
    find savant -name "*.c" -delete
fi

if [ -d "adapters" ]; then
    find adapters -path adapters/gst/gst_plugins/python -prune -o -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
    find adapters -path adapters/gst/gst_plugins/python -prune -o -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
    find adapters -name "*.c" -delete
fi

if [ -d "watchdog" ]; then
    find watchdog -name "*.py" -not -name "__main__.py" -print0 | xargs -0 cythonize -i -3
    find watchdog -name "*.py" -not -name "__main__.py" -print0 | xargs -0 rm
    find watchdog -name "*.c" -delete
fi
