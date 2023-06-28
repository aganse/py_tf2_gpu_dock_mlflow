# Makefile for MLproject based deep learning training run.
# Targets are in order of usage.

PROJECT = malaria


env:
	# Create python environment .venv within py_tf2_gpu_dock_mlflow directory,
	# activating it, and installing mlflow into it
	./make_env.bash

load_tfdata:
	# Downloading prefab Tensorflow dataset once
	# **** Note it's ok to see GPU/cudnn errors in this one because here we're just
	# using Tensorflow to download this dataset and nothing more.  All actual
	# computations with Tensorflow will be inside the Docker container later, which
	# is configured internally for GPU usage. ****
	python3 load_tfdata.py

mlflowquickcheck:
	# Just making sure can access mlflow
	mlflow experiments list

build:
	# Build the MLflow Project container for deep learning training run
	docker build -t $(PROJECT) .

run:
	# Run the deep learning training run in the MLflow Project
	./project_driver.bash


