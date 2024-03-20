#!/bin/bash
# you are expected to be in Savant/ directory
# Usage: run_perf.sh [MULTISTREAM] [YQ_ARGS]...
# The MULTISTREAM argument is an integer, 0 means using uridecodebin source,
# >0 means using the the multistream source adapter with specified number of streams.
# The YQ_ARGS are module configuration updates in yq notation,
# e.g. ".parameters.batch_size=4".

MODULE_CONFIG=samples/keypoint_detection/module.yml
DATA_LOCATION=data/shuffle_dance.mp4

source samples/assets/run_perf_helper.sh
run_perf $MODULE_CONFIG $DATA_LOCATION
