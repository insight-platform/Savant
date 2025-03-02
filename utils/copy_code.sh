#!/bin/bash
# This script used in the Dockerfile to copy and optionally process the specified source files.

set -x

if [ "$#" -lt 2 ]; then
  echo "Usage: ./copy_code.sh source_dir target_dir"
  exit 1
fi

source_dir=$1
target_dir=$2

cp -r $source_dir $target_dir

# do some processing on the source files
