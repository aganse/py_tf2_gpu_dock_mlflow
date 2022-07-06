# Makefile for MLproject usage.
# First do make build, then make run

PROJECT = patch_camelyon_vgg16

build:
	# Building container for deep learning run in an MLflow Project
	docker build -t $(PROJECT) .

load_tfdata:
	# Downloading prefab Tensorflow dataset once
	python load_tfdata.py

run:
	# Run the deep learning run in an MLflow Project
	./project_driver.bash

