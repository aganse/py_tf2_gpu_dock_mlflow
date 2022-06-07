#!/bin/bash

#mlflow run https://github.com/aganse/py_tf2_gpu_dock_mlflow \
mlflow run . \
    -A gpus=all                                         \
    -b local                                            \
    --experiment-name='Test/Debug'                      \
    -P batch_size=32                                    \
    -P epochs=10                                        \
    -P convolutions=3                                   \
    -P training_samples=15000                           \
    -P validation_samples=2000                          \
    -P randomize_images=True 

