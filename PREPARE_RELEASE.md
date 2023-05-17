# Preparing release 

## Build and publish OpenCV packages

This is required if the release includes an update to OpenCV or Deepstream versions, or to the savant OpenCV module (code in `libs/opencv/savant`).

Run

```
make build-opencv
```

Do this for both `x86` and `aarch64` platforms.

Move the resulting files into `savant-data/opencv-packages/x86` and `savant-data/opencv-packages/aarch64` buckets on the S3 storage.

## Build and publish Savant release

While on the `develop` git branch, run

```
./utils/prepare_release.sh
```

The script creates a local git branch named `releases/vX.Y.Z` where X.Y.Z is the Savant version defined in the `savant/VERSION` file. The branch will contain a commit that writes X.Y.Z Savant version into the samples Dockerfiles and samples docker-compose files.

The branch can be pushed to remote

```
git push -u origin releases/vX.Y.Z
```

Next, the developer is expected to create a PR from `releases/vX.Y.Z` into `main`, and tag the commit after merge. This starts the git workflows that will build the Savant package and docker images for the version.
