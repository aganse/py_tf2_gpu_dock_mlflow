#!/bin/bash

# (Fixme: should add a check here to ensure MLFLOW_TRACKING_URI is set)

# After development is frozen/slowed, we can also run trainings directly off
# the repo without cloning it locally; ie for just trying different parameters.
# Only this driver script is needed in that case, as the rest gets pulled from
# repo.  According to mlflow documentation a git commit or branch name at that
# repo uri can be specified to use like `-v abcde123` or `-v feature/mybranch`.
# I've not tried this yet though.  (Without -v and get the default branch.)
# mlflow run https://github.com/aganse/py_tf2_gpu_dock_mlflow -v abcde123 ...
# For more details see https://mlflow.org/docs/1.30.0/projects.html#running-projects

mlflow run .                                            \
    -A gpus=all                                         \
    -b local                                            \
    --experiment-name='Test/Debug'                      \
    -P run_name='malaria'                               \
    -P randomize_images=True                            \
    -P convolutions=0                                   \
    -P epochs=15                                        \
    -P batch_size=10                                    \
    -P training_samples=100                              \
    -P validation_samples=100

    # -P batch_size=128                                 \
    # -P training_samples=13779                         \
    # -P validation_samples=13779                              
