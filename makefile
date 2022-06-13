# Makefile for mlproject usage.
# first make build, then make run

PROJECT = patch_camelyon_vgg16

build:
	docker build -t $(PROJECT) .

load_tfdata:
	docker run -it -v /storage/tf_data:/app/data $(PROJECT) python load_tfdata.py

run:
	./project_driver.bash

