### A more generalized description of setup to run the py_tf2_gpu_dock_mlflow model training

Here are more general instructions to prepare/setup what's needed to run the
py_tf2_gpu_dock_mlflow training process, whether it's so you can use your own
separate MLflow instance, or your own already-running server or EC2 instance,
or whatever.  Also, these more general instructions can give additional
context to what steps are taken by the canned setup in Option #1.

1. have GPU and Docker already working on your system (e.g. make sure
   [these checks](check_gpu_docker.md) work).
2. have your MLFLOW_TRACKING_URI environment var pointing to a running MLflow 
   server, which must be running version 2+.  For example you might use a bash
   line like this to set that:
   `export MLFLOW_TRACKING_URI=http://localhost:5000`.  You might like to put
   that in your shell resource file (.bashrc for example).
3. either have your MLflow server's artifact storage directory accessible within 
   `/storage/mlruns` (noting that `/storage` is volume-mapped into the container - 
   see `MLproject` file), or your MLflow instance configured to hold everything in 
   S3.
4. have your training data files accessible somewhere within `/storage` (which is 
   volume-mapped into the container - see `MLproject` file).  For example, for 
   this repo's default malaria problem the dataset is stored in `/storage/tfdata`.
5. git clone this repo, cd into it, create python env via `make env`
6. enter the python environment that was just made:  `source .venv/bin/activate`.
7. `make load_tfdata` (if using the default tf dataset shown in this readme) to
   download the data to /storage/tfdata.
8. then `make build` to create the training Docker container.
9. then `make run` (only this step actually requires the python env, just for 
   mlflow cli).  The first thing mlflow does on starting the run is to add the 
   latest state of the train script and other files on top of the image built from
   the Dockerfile.  For this reason the run may initially look like it's frozen 
   while one cpu is pegged at 100%; but it's building that new image which
   takes several minutes.
   The resulting new image takes the name of the one created by the `make build`
   command, appended with a hash-based label of the present git commit hash, 
   looking something like `<original_image_name>:4e23a5b`.
10. The training of the default malaria example should crank up. If you log into
   your server in a second terminal window, you can run nvidia-smi to confirm the
   GPU is being used.  Also while the training is in progress you can go into the
   run-in-progress in MLflow, click on one of the metrics, and the plot of that
   metric over iterations will update.
