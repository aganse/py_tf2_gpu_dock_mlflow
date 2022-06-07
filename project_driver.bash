#!/bin/bash

#mlflow run https://github.com/aganse/py_tf2_gpu_dock_mlflow \
mlflow run . \
    -A gpus=all                                         \
    -b local                                            \
    --experiment-name='Test/Debug'                      \
    -P batch_size=64                                    \
    -P epochs=15                                        \
    -P convolutions=3                                   \
    -P training_samples=260000                          \
    -P validation_samples=30000                         \
    -P randomize_images=True 

