#!/bin/bash
# This script used in the Dockerfile to cythonize the specified source files.

set -x

if [ "$#" -lt 2 ]; then
  echo "Usage: ./copy_code.sh source_dir target_dir [cythonize]"
  exit 1
fi

source_dir=$1
target_dir=$2

cp -r $source_dir $target_dir

# if there is third argument, cythonize the source files
if [ "$#" -eq 3 ]; then
  python3 -m pip install --no-cache-dir cython

  find $target_dir -type f -name "*.py" ! -path "*/gst_plugins/python/*" | while read -r file; do
    if ! grep -E -q "if[[:space:]]+__name__[[:space:]]*==[[:space:]]*['\"]__main__['\"]" "$file"; then
      cythonize -i -3 "$file"
      rm "$file"
      rm "${file%.*}.c"
    fi
  done

  find $target_dir/.. -name "build" -type d -print0 | xargs -0 rm -rf
  find $target_dir -name "__pycache__" -type d -print0 | xargs -0 rm -rf
fi
