#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlflow

