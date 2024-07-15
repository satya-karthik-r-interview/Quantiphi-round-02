# -----------------------------------
# base image
FROM nvcr.io/nvidia/pytorch:24.01-py3
LABEL version="0.0.1"
ENV DEBIAN_FRONTEND=noninteractive
# -----------------------------------
# update image os
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
# -----------------------------------
# upgrade pip
RUN pip install --upgrade pip
# -----------------------------------
# Copy requriements file
COPY ./requirements.txt ./requirements.txt
# install isnisghtface from source
RUN pip install --default-timeout=300 -r requirements.txt
# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# -----------------------------------
# unistall opencv packages and install them again
# this is to fix the circular import error.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python && \
    pip install opencv-python==4.5.5.64

