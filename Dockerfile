FROM python:3.8-slim-buster

# Install python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

# Set the working directory in the container
WORKDIR /app

# Copy project requirements.txt into the container
COPY ./requirements.txt .

# Install the Python dependencies
RUN python -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy latest versions of python files into the image
COPY ./*.py ./
