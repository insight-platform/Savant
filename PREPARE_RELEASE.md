# Preparing release 

## Build and publish extra packages

This is required if the release includes an update to OpenCV or Deepstream versions, or to the savant OpenCV module (code in `libs/opencv/savant`).

To build OpenCV, run the following command:
```
make build-opencv
```
Note: The command will produce OpenCV packages for x86 and l4t platforms.

To build other packages (PyCUDA, Torch2TRT), run the following command:
```
make build-extra-packages
```
Note: The command will only build packages for the platform on which it was executed. 

Move the resulting files into `savant-data/packages` bucket on the S3 storage.
```
aws s3 --endpoint-url=https://eu-central-1.linodeobjects.com sync ./packages/ s3://savant-data/packages/
```

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

The last step is to change the version number to the next release version number in the `savant/VERSION` file.