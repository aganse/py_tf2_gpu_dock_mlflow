import tensorflow_datasets as tfds

# Only relevant when using prefab Tensorflow datasets - these are the lines to
# download that dataset once for local use, easily called via makefile.
# Note these are not tiny; after download they are expanded to much larger than
# their download size (but still within 100GB).

tfds.load('malaria', split=['train'], data_dir="/storage/tf_data/")
# note malaria dataset has all its images under 'train'; must split separately

# tfds.load('patch_camelyon', split=['train', 'test'], data_dir="/storage/tf_data/")
# tfds.load('celeb_a', split=['train', 'test'], data_dir="/storage/tf_data/")
# tfds.load('beans', split=['train', 'test'], data_dir="/storage/tf_data/")
