import argparse
import mlflow
import mlflow.tensorflow
from mlflow_callback import MlFlowCallback
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.saved_model import signature_constants, tag_constants


# Keeping settings for some other datasets:
# Note /app/data is a volume mapping from inside the container to host filesys
# IMAGE_SHAPE = (218, 178, 3)
# ds = tfds.load('celeb_a', split=['train', 'test'], data_dir="/app/data/")
# IMAGE_SHAPE = (500, 500, 3)
# ds = tfds.load('beans', split=['train', 'test'], data_dir="/app/data/")
IMAGE_SHAPE = (96, 96, 3)
ds = tfds.load('patch_camelyon', split=['train', 'test'], data_dir="/app/data/")


def data_generator(batch_size, samples, train=True):
    """Dataset definition/supply is entirely in this function.
    Choose between built-in tensorflow-provided dataset or a directory with photos
    """
    def generator():
        index = 0 if train else 1
        for sample in ds[index].take(samples).batch(batch_size).repeat():
            # Keeping settings for some other datasets:
            # yield sample['image']/255, tf.map_fn(lambda label: 1 if label else 0, sample['attributes']['Smiling'], dtype=tf.int32)  # celeb_a
            # yield sample['image'] / 255, tf.map_fn(lambda label: 1 if label == 2 else 0, sample['label'], dtype=tf.int32)  # beans
            yield sample['image'] / 255, sample['label']  # patch_camelyon
    return generator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size')
    parser.add_argument('--epochs')
    parser.add_argument('--convolutions')
    parser.add_argument('--training-samples')
    parser.add_argument('--validation-samples')
    parser.add_argument('--randomize-images')
    parser.add_argument('--run-name')
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    convolutions = int(args.convolutions)
    training_samples = int(args.training_samples)
    validation_samples = int(args.validation_samples)
    randomize_images = bool(args.randomize_images)
    run_name = args.run_name

    train_dataset = tf.data.Dataset.from_generator(
        generator=data_generator(batch_size, training_samples, train=True),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=(tf.TensorShape([None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]), tf.TensorShape([None])))

    validation_dataset = tf.data.Dataset.from_generator(
        generator=data_generator(batch_size, validation_samples, train=False),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=(tf.TensorShape([None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]), tf.TensorShape([None])))

    with mlflow.start_run(run_name=run_name):

        mlflow.tensorflow.autolog()
        # Alas log_models and registered_model_name args were not working in
        # autolog above, so "manually" specifying via log_model at end instead.
        # Similarly, run_name arg in mlflow.start_run() is not propagating to
        # mlflow.runName used by mlflow website, so manually specifying here.
        mlflow.set_tags({"mlflow.runName": run_name})

        # Define the network
        model = Sequential()
        model.add(Input(shape=IMAGE_SHAPE))
        if randomize_images:
            model.add(RandomFlip())
        if convolutions == 0:  # 0 is flag to use vgg16
            VGG16_MODEL = VGG16(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
            # VGG16_MODEL.trainable=False
            model.add(VGG16_MODEL)
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
        else:
            for i in range(convolutions):
                model.add(Conv2D(64 * (2**i), (3, 3), padding='same', activation='relu'))
                model.add(MaxPool2D(strides=(2, 2)))
                model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(2048, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

        # Compile and fit the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        model.fit(train_dataset,
                  validation_data=validation_dataset,
                  epochs=epochs,
                  steps_per_epoch=training_samples / batch_size,
                  validation_steps=validation_samples / batch_size,
                  callbacks=[MlFlowCallback()])

        # Save and register the model in the registry.
        # (weird, fyi while current mlflow version requires saved_model_dir to
        # be set in tf.keras.models.save_model and mlflow.tensorflow.log_model,
        # whatever I set there gets overwritten by "tfmodel".  Since must have
        # something set, just simply using "tfmodel" then, no problem it's just
        # a subdir created in the mlflow run artifacts directory.)
        tag = [tag_constants.SERVING]
        key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        tf.keras.models.save_model(model, "tfmodel")
        mlflow.tensorflow.log_model(tf_saved_model_dir="tfmodel",
                                    tf_meta_graph_tags=tag,
                                    tf_signature_def_key=key,
                                    artifact_path="model",
                                    registered_model_name=run_name + "_model")

        # Log the model architecture summary as a run artifact.
        stringlist = []
        model.summary(line_length=78, print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        mlflow.log_text(model_summary, "model/model_arch.txt")
