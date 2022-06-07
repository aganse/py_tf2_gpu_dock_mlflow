# Makefile for py_tf2_gpu_dock_mlflow
# first make build, then make run

build:
	docker build -t py_tf2_gpu_mlflow .

run:
	./test_proj_driver.bash

