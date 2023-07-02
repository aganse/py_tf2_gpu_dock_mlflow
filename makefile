# Makefile for MLproject based deep learning training run.
# Targets are in order of usage.

PROJECT = malaria

# Default MLFLOW_TRACKING_URI if not set in environment variable uses the
# default port for docker_mlflow_db on the linux-based docker host:
MLFLOW_TRACKING_URI ?= http://172.17.0.1:5000
.PHONY: all


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
	# NOTE this requires/assumes existence of /storage/tf_data directory.
	# TODO add verification that dir exists first!
	python3 load_tfdata.py

mlflowquickcheck:
	# Just making sure can access mlflow
	mlflow experiments search

build:
	# Build the MLflow Project container for deep learning training run
	docker build -t $(PROJECT) .

run:
	# Run the deep learning training run in the MLflow Project
	#
	# Unless first time building, this build should use existing image and just re-add *.py files.
	# This is only here because --build-image in mlflow run (in project_driver.bash), which is
	# supposed to do this, hangs with pegged cpu as of mlflow 2.4.1.  Unecesary here once fixed.
	docker build -t $(PROJECT) .
	#
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) ./project_driver.bash
