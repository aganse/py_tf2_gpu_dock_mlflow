# Resulting container is 7.3GB but note AMD64 only:
FROM tensorflow/tensorflow:latest-gpu
#
# Resulting container is 12.2GB but AMD64+ARM64:
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# RUN apt-get update && apt-get install -y python3 build-essential python3-dev python3-pip python3-venv
# RUN ln -s /usr/bin/python3 /usr/bin/python
# (the base, runtime, cudnn8-runtime images don't have all needed libs,
# but egads this thing is huge, i guess from the dual architectures)

# Set the working directory in the container
WORKDIR /app

# Copy project requirements.txt into the container
COPY ./requirements.txt .

# Install the Python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy latest versions of python files into the image
COPY ./*.py ./
