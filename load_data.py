import tensorflow_datasets as tfds

# tfds.load('celeb_a', split=['train', 'test'], data_dir="/app/data/")
tfds.load('beans', split=['train', 'test'], data_dir="/app/data/")
