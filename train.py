import argparse

import mlflow
import mlflow.tensorflow
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from mlflow_callback import MlFlowCallback

# keeping settings for some other datasets
# IMAGE_SHAPE = (218, 178, 3)
# ds = tfds.load('celeb_a', split=['train', 'test'], data_dir="/app/data/")
# IMAGE_SHAPE = (500, 500, 3)
# ds = tfds.load('beans', split=['train', 'test'], data_dir="/app/data/")
IMAGE_SHAPE = (96, 96, 3)
ds = tfds.load('patch_camelyon', split=['train', 'test'], data_dir="/app/data/")


def data_generator(
    batch_size,
    samples,
    train=True
):

    def generator():
        index = 0 if train else 1
        for sample in ds[index].take(samples).batch(batch_size).repeat():
            # keeping settings for some other datasets
            # yield sample['image']/255, tf.map_fn(lambda label: 1 if label else 0, sample['attributes']['Smiling'], dtype=tf.int32)  # celeb_a
            # yield sample['image'] / 255, tf.map_fn(lambda label: 1 if label == 2 else 0, sample['label'], dtype=tf.int32)  # beans
            yield sample['image'] / 255, sample['label']

    return generator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size')
    parser.add_argument('--epochs')
    parser.add_argument('--convolutions')
    parser.add_argument('--training-samples')
    parser.add_argument('--validation-samples')
    parser.add_argument('--randomize-images')
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    convolutions = int(args.convolutions)
    training_samples = int(args.training_samples)
    validation_samples = int(args.validation_samples)
    randomize_images = bool(args.randomize_images)

    train_dataset = tf.data.Dataset.from_generator(
        generator=data_generator(batch_size, training_samples, train=True),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=(tf.TensorShape([None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]), tf.TensorShape([None])))

    validation_dataset = tf.data.Dataset.from_generator(
        generator=data_generator(batch_size, validation_samples, train=False),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=(tf.TensorShape([None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]), tf.TensorShape([None])))

    with mlflow.start_run():

        mlflow.tensorflow.autolog()

        model = Sequential()
        model.add(Input(shape=IMAGE_SHAPE))

        if randomize_images:
            model.add(RandomFlip())

        for x in range(convolutions):
            model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
        model.fit(train_dataset,
                  validation_data=validation_dataset,
                  epochs=epochs,
                  steps_per_epoch=training_samples / batch_size,
                  validation_steps=validation_samples / batch_size,
                  callbacks=[MlFlowCallback()])
