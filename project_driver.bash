#!/bin/bash

# (Fixme: should add a check here to ensure MLFLOW_TRACKING_URI is set)

# After development pinned/slowed, we can also run this directly off
# its repo without cloning it locally; only this driver script is
# needed in that case:
#mlflow run https://github.com/aganse/py_tf2_gpu_dock_mlflow \
mlflow run . \
    -A gpus=all                                         \
    -b local                                            \
    --experiment-name='Test/Debug'                      \
    -P run_name='patch_camelyon'                        \
    -P randomize_images=True                            \
    -P convolutions=0                                   \
    -P epochs=15                                        \
    -P batch_size=128                                   \
    -P training_samples=260000                          \
    -P validation_samples=30000                         
