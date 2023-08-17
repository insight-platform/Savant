#!/usr/bin/env bash

SAVANT_RS_VERSION=0.1.51
curl -L \
    "https://github.com/insight-platform/savant-rs/releases/download/${SAVANT_RS_VERSION}/artifacts.tar.gz" \
    -o savant-rs-capi.tar.gz
tar -xvzf savant-rs-capi.tar.gz
mkdir -p savant_capi/lib
mkdir -p savant_capi/include
cp "savant_capi/artifacts/libsavant_capi_$(arch).so" savant_capi/lib/libsavant_capi.so
cp "savant_capi/artifacts/savant_capi.h" savant_capi/include
rm -rf savant-rs-capi.tar.gz savant_capi/artifacts
