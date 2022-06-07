import tensorflow_datasets as tfds

tfds.load('patch_camelyon', split=['train', 'test'], data_dir="/app/data/")
