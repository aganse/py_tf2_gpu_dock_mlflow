# Makefile for MLproject usage.
# First do make build, then make run

PROJECT = malaria

env:
	# Create python environment .venv within py_tf2_gpu_dock_mlflow directory,
	# activating it, and installing mlflow into it
	python3 -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	pip install mlflow==1.30.0

build:
	# Building container for deep learning run in an MLflow Project
	docker build -t $(PROJECT) .

load_tfdata:
	# Downloading prefab Tensorflow dataset once
	# **** Note it's ok to see GPU/cudnn errors here because here we're just
	# using Tensorflow to download this dataset and nothing more.  All actual
	# computations with Tensorflow will be in the Docker container later, which
	# is configured internally for GPU usage. ****
	python3 load_tfdata.py

mlflowquickcheck:
	# Just making sure can access mlflow
	mlflow experiments list

run:
	# Run the deep learning run in an MLflow Project
	./project_driver.bash

