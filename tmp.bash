/bin/bash

python train.py \
      --batch-size 2 \
      --epochs 15 \
      --convolutions 0 \
      --training-samples 10 \
      --validation-samples 10 \
      --randomize-images True \
      --run-name 'malaria'

