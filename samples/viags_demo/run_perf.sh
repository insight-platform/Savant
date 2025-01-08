#!/bin/bash
# you are expected to be in Savant/ directory
# Usage: run_perf.sh [MULTISTREAM] [YQ_ARGS]...
# The MULTISTREAM argument is an integer, 0 means using uridecodebin source,
# >0 means using the the multistream source adapter with specified number of streams.
# The YQ_ARGS are module configuration updates in yq notation,
# e.g. ".parameters.batch_size=4".

MODULE_CONFIG=samples/viags_demo/module.yml
DATA_LOCATION=cache/videos/cam_1_sample_savant_1_part_1.mp4

source samples/assets/run_perf_helper.sh

if [ "$(uname -m)" = "aarch64" ]; then
  YQ_ARGS+=('.parameters.model_output_converter="converter"')
else
  YQ_ARGS+=('.parameters.model_output_converter="gpu_converter"')
fi

run_perf $MODULE_CONFIG $DATA_LOCATION