# Makefile for MLproject usage.
# First do make build, then make run

PROJECT = patch_camelyon_vgg16

env:
	# Create python environment .venv within py_tf2_gpu_dock_mlflow directory,
	# activating it, and installing mlflow into it
	python3 -m venv .venv
	source .venv/bin/activate
	pip install mlflow

build:
	# Building container for deep learning run in an MLflow Project
	docker build -t $(PROJECT) .

load_tfdata:
	# Downloading prefab Tensorflow dataset once
	python3 load_tfdata.py

run:
	# Run the deep learning run in an MLflow Project
	./project_driver.bash

