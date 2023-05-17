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
./utils/prepare_release.sh savant_ver ds_ver
```

where

- `savant_ver` is the Savant version in the form X.Y.Z, .Z component is optional
- `ds_ver` is the Deepstream version in the form M.N.K, .K component is optional

The script creates a local git branch named `releases/vX.Y.Z` with the X.Y.Z Savant version fixed in the samples Dockerfiles, samples docker-compose files and the `savant/VERSION` file.

The branch can be pushed to remote

```
git push -u origin releases/vX.Y.Z
```

To start the git workflows that will build the Savant package and docker images for this version

1. Tag the commit in the branch as `vX.Y.Z`

```
git tag -a vX.Y.X -m "Release version X.Y.Z"
```

2. And push the tag to remote

```
git push origin vX.Y.Z
```
