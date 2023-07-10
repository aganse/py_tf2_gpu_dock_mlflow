#!/bin/bash

# Optional install script to prep AWS DLAMI on g4dn.XXXX image to run trainings
# to get going quickly on AWS.


# Verify started EC2 instance with correct architecture and AMI image
if [[ $(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | grep architecture | sed 's/^.*: "\([a-z0-9_]*\)",/\1/') != "x86_64" ]]; then echo Error: you are not running instance with x86_64 processor, which this script and repo were written for.; exit; fi
if [[ $(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | grep imageId | sed 's/^.*: \"//' | sed 's/\",$//') != "ami-01126ee2e34c5f04e" ]]; then echo Error: you are not running the "Deep Learning AMI GPU TensorFlow 2.12.0 (Ubuntu 20.04) 20230529" AMI, which this script was written for.; exit; fi


# Install docker compose (following https://docs.docker.com/compose/install/linux)
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
# verify good to go by outputting version:
docker compose version


# Install a few remaining python bits.
# Theme here is that we must do some python stuff in host system.  (#2) could
# be done in the docker project container (but why bother given #2), but (#1)
# is inherently about kicking off the MLproject from the host system.
#
# 1.) for creating the python env:
sudo apt install -y python3-venv
# 2.) installing tfdataset wanted to compile some piece it couldn’t find binary
# package for, requiring this:
sudo apt install -y gcc python3-dev


# Grant access to Github before we can install the below repos
ssh-keygen -q -t rsa -N '' <<< $'\ny' >/dev/null 2>&1
echo
echo Paste this public key into Github account to allow access to clone repos...
cat ~/.ssh/id_rsa.pub
# paste pub key into GitHub settings / SSH keys
read -rsp $'Press enter to continue installation after public key was pasted into Github account to allow repo cloning...\n'


# Install/start mlflow
mkdir -p ~/src/python
cd ~/src/python 
git clone git@github.com:aganse/docker_mlflow_db.git
cd docker_mlflow_db
git checkout feature/update_versions
echo -n mydbadminpassword  > ~/.pgadminpw
echo db:5432:mlflow:postgres:mydbadminpassword > ~/.pgpass
chmod 600 ~/.pg*
make start
echo MLflow started.


# Install/start py_tf2_gpu_dock_mlflow  [put into script file]
sudo mkdir -p /storage/tf_data    # malaria tfdataset will take up 1.1GB
sudo mkdir -p /storage/mlruns    # for when not using s3 for mlflow artifacts
sudo chown -R ubuntu:ubuntu /storage
cd ~/src/python
git clone git@github.com:aganse/py_tf2_gpu_dock_mlflow.git 
cd ~/src/python/py_tf2_gpu_dock_mlflow
git checkout feature/malaria
make env
source .venv/bin/activate
make load_tfdata   # only takes a few min for relatively small malaria dataset
echo Malaria dataset installed locally

