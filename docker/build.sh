#!/usr/bin/env bash

IMG_NAME="huggingface:0.0.1"

docker build --tag "$IMG_NAME" docker \
    -f docker/Dockerfile
