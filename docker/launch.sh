#!/usr/bin/env bash

IMG_NAME="huggingface:0.0.1"
CONTAINER_NAME="huggingface_$USER"

docker_run_user () {
  tempdir=$(mktemp -d)
  getent passwd > ${tempdir}/passwd
  getent group > ${tempdir}/group
  docker run -v${HOME}:${HOME} --runtime=nvidia -w$(pwd) --rm -u$(id -u):$(id -g\
) $(for i in $(id -G); do echo -n ' --group-add='$i; done) -v ${tempdir}/passwd:/etc/passwd:ro -v ${tempdir}/group:/etc/group:ro "$@"
}


docker_run_user --name $CONTAINER_NAME -it --rm \
    --net=host \
    --ipc=host \
    --gpus all \
    --entrypoint /bin/bash \
    -v "$PWD:/workspace" \
    "$IMG_NAME"
