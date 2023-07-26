# Preparing release 

## Build and publish OpenCV packages

This is required if the release includes an update to OpenCV or Deepstream versions, or to the savant OpenCV module (code in `libs/opencv/savant`).

Run

```
make build-opencv
```

Move the resulting files into `savant-data/opencv-packages/x86` and `savant-data/opencv-packages/aarch64` buckets on the S3 storage.

## Build and publish Savant release

While on the `develop` git branch, run

```
./utils/prepare_release.sh
```

The script creates a local git branch named `releases/X.Y.Z` where X.Y.Z is the Savant version defined in the `savant/VERSION` file. The branch will contain a commit that writes X.Y.Z Savant version into the samples Dockerfiles and samples docker-compose files.

The branch can be pushed to remote

```
git push -u origin releases/X.Y.Z
```

Next, create a release (`vX.Y.Z`) from `releases/X.Y.Z`. The release initiates git workflows that build the Savant package and docker images for the version.

