# py_tf2_gpu_dock_mlflow
An example python/Tensorflow2 model using GPU and MLflow in a docker container

Heavily based on:  [George Novack's 2020 article in Towards Data Science "Create Reusable
ML Modules with MLflow Projects & Docker"](
https://towardsdatascience.com/create-reusable-ml-modules-with-mlflow-projects-docker-33cd722c93c4)
Thank you!


### How to install/run

First, per [Google's Tensorflow Docker documentation](https://www.tensorflow.org/install/docker),
check that the NVidia GPU device is present:
```
> lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2080 SUPER] (rev a1)
01:00.1 Audio device: NVIDIA Corporation TU104 HD Audio Controller (rev a1)
01:00.2 USB controller: NVIDIA Corporation TU104 USB 3.1 Host Controller (rev a1)
01:00.3 Serial bus controller [0c80]: NVIDIA Corporation TU104 USB Type-C UCSI Controller (rev a1)
```
And then verify your nvidia-docker installation:
```
docker run --gpus all --rm nvidia/cuda nvidia-smi
```
Set your MLFLOW_TRACKING_URI:
`MLFLOW_TRACKING_URI=http://localhost:5000`

And then two steps regarding this repo:

1. First load the dataset and build the docker image:  `make build`
2. Then run the training, which will progressively log state into mlflow:  `make run`



### References/links


Useful in ironing out GPU usage in the Docker container:  <https://www.tensorflow.org/install/docker>

Possibly of use later as I pull together example variations:

* <https://cosminsanda.com/posts/experiment-tracking-with-mlflow-inside-amazon-sagemaker>
* <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>
* <https://www.tensorflow.org/tutorials/load_data/images>
* <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>
* <https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files>
* <https://github.com/tensorflow/io/issues/174>  (looks like TF-IO has built-in HDF5 reader?)

