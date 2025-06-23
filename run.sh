#!/usr/bin/env bash
set -e

IMAGE=sudochris/falcon_docker_template:v1         # after you `docker build -t $IMAGE .`

# allow root container to access X
xhost +local:root

docker run -it --rm \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v $(pwd)/models:/workspace/models \
  $IMAGE "$@"
