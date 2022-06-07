# py_tf2_gpu_dock_mlflow
An example Python/Tensorflow2 model using GPU and MLflow in a docker container.

In this example, we use the 
[patch_camelyon dataset](https://www.tensorflow.org/datasets/catalog/patch_camelyon)
built-in to Tensorflow to train/test a CNN-based image classification model to
detect metastatic tissue in histopathologic scans of lymph node sections.
The focus here is not on that specific image classification problem (and pardon
me but I don't know much about that problem itself, nor claim to have come up
with a model here that works well on it); rather it's to provide a convenient
template to rapidly throw together new models that use 
[Python](https://www.python.org)/[Tensorflow](https://www.tensorflow.org)
models running on GPUs in a [Docker](https://www.docker.com) container and log
the results to [MLflow](https://mlflow.org) using its "Project" functionality.
The code and setup here are heavily based on
[George Novack's 2020 article in Towards Data Science, "Create Reusable
ML Modules with MLflow Projects & Docker"](
https://towardsdatascience.com/create-reusable-ml-modules-with-mlflow-projects-docker-33cd722c93c4)
(Thank you!)  I've just pulled things together into one repo, added a little
bit of functionality, and set things up for some more portability.  And alas
the Celebreties ([`celeb_a`](https://www.tensorflow.org/datasets/catalog/celeb_a))
dataset used in his original example appears to not be available in Tensorflow
datasets anymore so I've chosen this other medical one.

![lymph node section example images](./pcam.png)
<sub><sup>
[Veeling, Linmans, Winkens, Cohen, Welling - 2018](https://doi.org/10.1007/978-3-030-00934-2_24)
</sup></sub>


### How to install/run

#### First, ensure your system's all ready:
Per [Google's Tensorflow Docker documentation](https://www.tensorflow.org/install/docker),
check that the NVidia GPU device is present:
```
> lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2080 SUPER] (rev a1)
01:00.1 Audio device: NVIDIA Corporation TU104 HD Audio Controller (rev a1)
01:00.2 USB controller: NVIDIA Corporation TU104 USB 3.1 Host Controller (rev a1)
01:00.3 Serial bus controller [0c80]: NVIDIA Corporation TU104 USB Type-C UCSI Controller (rev a1)
```
Then verify your nvidia-docker installation, e.g.:
```
> docker run --gpus all --rm nvidia/cuda nvidia-smi
Sun Jun  5 16:31:20 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
| 18%   26C    P8     4W / 250W |    134MiB /  7982MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
And set your MLFLOW_TRACKING_URI to whatever address you use for it, e.g.:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```
You might want to put that in your shell resource file (.bashrc for example).


#### Then two main steps to run things:
(Well ok first git clone this repo and cd into it.  Then two steps...)

1. `make build` :  Load the dataset and build the docker image.  Note this
                   dataset is 7.5GB and can take a while to download, but at
                   least that's a one-time event.  Do note the whole dataset
                   is then stored in the project container, which probably
                   wouldn't be what you would do in practice assuming you'd
                   be operating on more data, but at least it gets us started
                   in this example (it's what the original example code did).
2. `make run`   :  Run the training, which will progressively log state into
                   mlflow.  Again the present state of this repo is not meant
                   as any competitive modeling on this topic - it's terrible
                   in fact - but it functions, and its provides a template.
                   Hyperparameters can be adjusted in the bash script.

![MLflow logged run example image](./mlflow_run.png)

The `make run` macro runs the `project_driver.bash` shell script, but a Python
script `project_driver.py` with mostly-corresponding functionality is included
too.  However, importantly note: as of this writing, it appears that GPU usage
can only be set for models in Docker containers in MLFlow Projects if using the
shell script call to mlflow (ie the shell command `mlflow` now just recently
takes a `gpus=all` argument, but the Python mlflow.projects.run() method still
does not do so yet!).


### Next steps

1. Log the resulting model into the MLFlow registry.
2. Serve the resulting model from MLFlow registry via MLFlow Serving.
3. Generalize the training and other code further away from Novack's original
   scripts (but thank you again!) to make it easier to rapidly adapt this to
   new quickie problems that arise.


### References/links

Useful in ironing out GPU usage in the Docker container:  <https://www.tensorflow.org/install/docker>

Possibly of use later as I pull together some example variations:

* <https://cosminsanda.com/posts/experiment-tracking-with-mlflow-inside-amazon-sagemaker>
* <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>
* <https://www.tensorflow.org/tutorials/load_data/images>
* <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>
* <https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files>
* <https://github.com/tensorflow/io/issues/174>  (looks like TF-IO has built-in HDF5 reader?)

