#!/bin/bash

# bash script first argument is the Savant version to be released
# validate the version number
if [[ ! $1 =~ ^[0-9]+\.[0-9]+\.?[0-9]*$ ]]; then
    echo "Invalid version number: $1"
    echo "Version number must be in the format: x.y.z"
    exit 1
fi

# bash script second argument is the DeepStream version used in the release
# validate the DeepStream version number
if [[ ! $2 =~ ^[0-9]+\.[0-9]+\.?[0-9]*$ ]]; then
    echo "Invalid DeepStream version number: $2"
    echo "Version number must be in the format: x.y.z"
    exit 1
fi
SAVANT_VER=$1
DS_VER=$2

# create a git branch named releases/x.y.z from the current branch
git checkout -b releases/$SAVANT_VER

VERSION_FILE=savant/VERSION
# replace the version numbers in the VERSION file
sed -i "s/SAVANT=.*/SAVANT=$SAVANT_VER/;s/DEEPSTREAM=.*/DEEPSTREAM=$DS_VER/" $VERSION_FILE
git add $VERSION_FILE

DEFAULT_TAG=latest
PATTERN_DS="savant-(adapters-)?deepstream(-l4t)?"
PATTERN_NO_DS="savant-(adapters-)?(gstreamer|py)(-l4t)?"
SED_DS="s/($PATTERN_DS):$DEFAULT_TAG/\1:$SAVANT_VER-$DS_VER/g"
SED_NO_DS="s/($PATTERN_NO_DS):$DEFAULT_TAG/\1:$SAVANT_VER/g"
SED_CMD="$SED_DS;$SED_NO_DS"

# find files with the name pattern "[Dd]ocker*" in the samples directory
# and save the list of files to a variable
readarray -d '' array < <(find samples -type f -name "[Dd]ocker*" -print0)

# iterate over the list of files in array
for file in "${array[@]}"; do
    # replace the version numbers in the Dockerfiles and docker-compose files
    sed -i -r $SED_CMD $file
    git add $file
done

# commit the changes
git commit -m "Prepare release $SAVANT_VER"
