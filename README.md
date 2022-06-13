# py_tf2_gpu_dock_mlflow
_**Get new Python/Tensorflow/GPU models running quickly, and logging performance
and resulting model in MLflow, keeping everything in Docker via MLflow Projects.**_

An example Python/Tensorflow2 model using GPU and MLflow in a docker container.
Python3, Tensorflow2, and the NVidia/GPU handling are entirely in the container;
the host system only needs the Nvidia driver and Docker installed.  An MLflow
instance is looked for at the address in the MLFLOW_TRACKING_URI environment
variable per usual MLflow usage - see my 
[docker_mlflow_db](https://github.com/aganse/docker_mlflow_db) repo for an easy
Docker-based way to get that running quickly too.

In this example, we use the 
[patch_camelyon](https://www.tensorflow.org/datasets/catalog/patch_camelyon)
breast-cancer detection dataset from the Tensorflow datasets to train/test a
VGG166-based image classification model to detect metastatic tissue in
histopathologic scans of lymph node sections.  
I don't claim to know much about that specific problem itself, but I have based
the model configuration and parameters in this demonstration upon several
[papers on the topic](#References).  The idea is to provide a convenient
template to rapidly throw together new models that use 
[Python](https://www.python.org)/[Tensorflow2](https://www.tensorflow.org)
running on GPUs in a [Docker](https://www.docker.com) container and log
the results to [MLflow](https://mlflow.org) using its "Project" functionality.
The models are logged to the MLflow registry and can be easily served via REST
API or downloaded from the registry into python code for further use.

The code and setup were initially based on [George Novack's 2020 article in
Towards Data Science, "Create Reusable ML Modules with MLflow Projects & Docker"](
https://towardsdatascience.com/create-reusable-ml-modules-with-mlflow-projects-docker-33cd722c93c4)
(thank you).  I've pulled things together into an immediately usable repo meant
to use as a template, added some functionality (artifact logging, model registry,
gpu capability), and generalized it a bit.

<img src="./pcam.png" alt="lymph node section example images" width="60%"/><BR>
<sup>
[Veeling, Linmans, Winkens, Cohen, Welling - 2018](https://doi.org/10.1007/978-3-030-00934-2_24)
</sup>


## How to install/run

### First, ensure your system's all ready:
Per [Google's Tensorflow Docker documentation](https://www.tensorflow.org/install/docker),
check that your NVidia GPU device is present:
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
If any trouble with those, that 
[Google Tensorflow Docker documentation](https://www.tensorflow.org/install/docker)
is really helpful.

Lastly set your MLFLOW_TRACKING_URI to whatever address you use for it, e.g.:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```
You might want to put that in your shell resource file (.bashrc for example).


### Then follow these steps to run things:

1. git clone this repo and cd into it...
2. `make build` :  Build the docker image; super quick.
3. `make load_tfdata` :  Download and setup the patch_camelyon dataset from
                   Tensorflow datasets.  Note this dataset is 7.5GB and this
                   step can take a little while, but it's a one-time event.
2. `make run`   :  Run the training, which will progressively log state into
                   mlflow.  This too can take a while.  For context, on a
                   NVIDIA GeForce RTX 2080 SUPER it took about two hours.

Once the run is running, you should find metrics progress logging in your
MLFlow instance, something like this (yes this example is a bit overfit):<BR>
<img src="./mlflow_run.png" alt="MLflow logged run example image" width="60%"/>

The `make run` macro runs the `project_driver.bash` shell script, but a Python
script `project_driver.py` with mostly-corresponding functionality is included
here too.  However, importantly note:  as of this writing, it appears that GPU
usage can only be done for models in Docker containers in MLFlow Projects if
using the _shell script_ call to mlflow.  I.e. the shell command `mlflow` now
just recently takes a `gpus=all` argument, whereas the Python
`mlflow.projects.run()` method still does not do so yet.  Note that's strictly
about an issue with MLflow, not with Python or Tensorflow or Docker.


## References

About this patch_camelyon breast-cancer cell detection problem:
* <https://github.com/basveeling/pcam/blob/master/README.md#why-pcam>
* <https://www.diva-portal.org/smash/get/diva2:1597512/FULLTEXT01.pdf>
* <https://arxiv.org/pdf/1909.11870.pdf>
* <https://ieeexplore.ieee.org/document/9626116>

About relevant Tensorflow details in particular:
* <https://www.tensorflow.org/install/docker>
* <https://www.tensorflow.org/guide/keras/preprocessing_layers>
* <https://www.tensorflow.org/api_docs/python/tf/keras/metrics>
* <https://www.tensorflow.org/tutorials/structured_data/imbalanced_data>
* <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>
* <https://www.tensorflow.org/tutorials/load_data/images>
* <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>

About additional computational tools used here (Docker, MLflow, etc):
* <https://towardsdatascience.com/using-mlflow-to-track-and-version-machine-learning-models-efd5daa08df0>
* <https://santiagof.medium.com/effortless-models-deployment-with-mlflow-2b1b443ff157>
* <https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16%20is%20a%20convolution%20neural,vision%20model%20architecture%20till%20date.>
* <https://cosminsanda.com/posts/experiment-tracking-with-mlflow-inside-amazon-sagemaker>
* <https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files>
* <https://github.com/tensorflow/io/issues/174>  (looks like TF-IO has built-in HDF5 reader?)

