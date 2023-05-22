#!/bin/bash
VERSION_FILE=savant/VERSION
# parse the version numbers from the VERSION file
SAVANT_VER=$(cat $VERSION_FILE | awk -F= '$1=="SAVANT"{print $2}' | sed 's/"//g')
DS_VER=$(cat $VERSION_FILE | awk -F= '$1=="DEEPSTREAM"{print $2}' | sed 's/"//g')

# create a git branch named releases/x.y.z from the current branch
git checkout -b releases/$SAVANT_VER

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
