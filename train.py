import argparse
import json
import mlflow
import mlflow.tensorflow
from mlflow_callback import MlFlowCallback
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.saved_model import signature_constants, tag_constants


# This IMAGE_SHAPE is used globally, but this initialized value is only used
# in two of the three use-cases in the define_data_generator() wrapper:
# define_data_generator_dataframe() and define_data_generator_imagedir(), which
# allow image resizing to this size.  But define_data_generator_tfdataset()
# overwrites this initialized value for IMAGE_SHAPE to that of the tf dataset.
# We should add that resizing to the define_data_generator_tfdataset() as well.
IMAGE_SHAPE = (128, 128, 3)


def define_data_generator(batch_size, samples, train=True):
    """ General wrapper for data generator defintion: reference the subfunction
    variation based on desired data source (tf dataset, directory of images,
    database query of image filepaths and labels).
    For now, just uncomment the desired wrapped define_data_generator_***()
    function and comment out the others.
    """
    return define_data_generator_tfdataset(batch_size, samples, train)
    # return define_data_generator_imagedir(batch_size, samples, train)
    # return define_data_generator_dataframe(batch_size, samples, train)


def define_data_generator_tfdataset(batch_size, samples, train=True):
    """ Create data generator based on a prefab Tensorflow dataset.  There
    aren't a ton of such datasets but they have their uses. """

    global IMAGE_SHAPE

    # Settings for some other datasets (match with yield lines below):
    # Note /storage/tf_data is volume mapped from host file system
    # ds = tfds.load("celeb_a", split=["train", "test"], data_dir="/storage/tf_data/")
    # IMAGE_SHAPE = (218, 178, 3)  # celeb_a
    # ds = tfds.load("beans", split=["train", "test"], data_dir="/storage/tf_data/")
    # IMAGE_SHAPE = (500, 500, 3)  # beans
    # ds = tfds.load("patch_camelyon", split=["train", "test"], data_dir="/storage/tf_data/")
    # IMAGE_SHAPE = (96, 96, 3)  # patch_camelyon

    # Unlike the above datasets, 'malaria' only has 'train' section so split that.
    # Using slicing form (like [:50%]) rather than even_splits() for later flexibility.
    ds = tfds.load("malaria", split=["train[:50%]", "train[50%:]"], data_dir="/storage/tf_data/")
    # fyi the split usage below works on split[0] and split[1] not names.
    IMAGE_SHAPE = (100, 100, 3)  # malaria

    def gen_callable(batch_size, samples, train=True):
        """ A callable function that returns a generator needed to form the
        dataset. """
        tindex = 0 if train else 1

        def generator():
            for sample in ds[tindex].take(samples).batch(batch_size).repeat():
                # Settings for some other datasets (match with ds lines above):
                # yield sample["image"]/255, tf.map_fn(lambda label: 1 if label else 0, sample["attributes"]["Smiling"], dtype=tf.int32)  # celeb_a
                # yield sample["image"] / 255, tf.map_fn(lambda label: 1 if label == 2 else 0, sample["label"], dtype=tf.int32)  # beans
                yield sample["image"] / 255, sample["label"]  # malaria
        return generator

    # the keras model.fit() function doesn't like the form of this generator
    # directly, but accepts a dataset created from it, so creating that here
    dataset = tf.data.Dataset.from_generator(
        generator=gen_callable(batch_size, samples, train),
        output_types=(tf.uint8, tf.uint8),
        output_shapes=(
            tf.TensorShape([None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]),
            tf.TensorShape([None])
        )
    )
    return dataset


def define_data_generator_imagedir(batch_size, samples, train=True):
    """ Create data generator based on contents of an image directory."""

    print("WARNING, THIS FUNCTION HAS NOT YET BEEN TESTED; MAY NEED DEBUGGING.")

    # Note train_datagen includes image augmentations and val_datagen does not,
    # otherwise we could just use one ImageDataGenerator with validation_split
    # for both.
    if train:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.8,
        )
        generator = train_datagen.flow_from_directory(
            "data/train",
            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
            batch_size=batch_size,
            class_mode="raw"
        )
    else:
        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
        )
        generator = val_datagen.flow_from_directory(
            "data/validation",
            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
            batch_size=batch_size,
            class_mode="raw"
        )
    return generator


def define_data_generator_dataframe(batch_size, samples, train=True):
    """ Create generator based image path/label contents of pandas dataframe.
    (eg from a database, but here for example just manually spec tiny df.)
    """

    # This datadict and df are just to demonstrate without a database connected;
    # in real usage populate df with contents pulled from a database query.
    print("NOTE IF YOU HAVEN'T UPDATED THE CONTENTS OF files_dataframe.json")
    print("FROM ITS EXAMPLE ENTRIES TO REAL IMAGES THAT EXIST AT THOSE PATHS")
    print("THEN YOU WILL GET INVALID-IMAGE-FILENAME AND FOLLOW-ON ERRORS BELOW.")
    with open("files_dataframe.json", "r") as f:
        datadict = json.load(f)
    df = pd.DataFrame(datadict)

    # Note train_datagen includes image augmentations and val_datagen does not,
    # otherwise we could just use one ImageDataGenerator with validation_split
    # for the two generators.
    if train:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.8,
        )
        generator = train_datagen.flow_from_dataframe(
            dataframe=df,
            x_col="imagepath",  # no directory arg so path is absolute
            y_col="label",
            batch_size=batch_size,
            shuffle=False,
            class_mode="raw",
            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        )
    else:
        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
        )
        generator = val_datagen.flow_from_dataframe(
            dataframe=df,
            x_col="imagepath",  # no directory arg so path is absolute
            y_col="label",
            batch_size=batch_size,
            shuffle=False,
            class_mode="raw",
            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        )
    return generator


def define_network(randomize_images, convolutions):
    """ Encapsulating the neural network definition. """
    model = Sequential()
    model.add(Input(shape=IMAGE_SHAPE))
    if randomize_images:
        model.add(RandomFlip())
    if convolutions == 0:  # 0 is flag to use vgg16
        VGG16_MODEL = VGG16(
            input_shape=IMAGE_SHAPE,
            include_top=False,
            weights="imagenet"
        )
        # VGG16_MODEL.trainable=False   # pin all layers
        # VGG16_MODEL.layers[n].trainable=False  # pin layer n
        model.add(VGG16_MODEL)
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
    else:
        for i in range(convolutions):
            model.add(Conv2D(64 * (2**i), (3, 3), padding="same", activation="relu"))
            model.add(MaxPool2D(strides=(2, 2)))
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(2048, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

    # for l in model.layers:
    #     print(l.name, l.trainable)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=["accuracy", "AUC", "Precision", "Recall"]
    )
    return model


def train_model(
    batch_size,
    epochs,
    convolutions,
    train_samples,
    val_samples,
    randomize_images,
    run_name,
):
    """ Run the model training and log performance and model to mlflow. """

    # Generic call to define data; any changes for different data sources are
    # found up in define_data_generator() (or the functions that it wraps).
    train_gen = define_data_generator(batch_size, train_samples, train=True)
    valid_gen = define_data_generator(batch_size, val_samples, train=False)

    with mlflow.start_run(run_name=run_name):

        mlflow.tensorflow.autolog()
        # Alas log_models and registered_model_name args were not working in
        # autolog above, so "manually" specifying via log_model at end instead.
        # Similarly, run_name arg in mlflow.start_run() is not propagating to
        # mlflow.runName used by mlflow website, so manually specifying here.
        mlflow.set_tags({"mlflow.run_name": run_name})
        # mlflow.set_tags({"mlflow.runName": run_name})  # delete after confirm change works

        # Define the network
        model = define_network(randomize_images, convolutions)

        # Log the model architecture summary as a run artifact.
        stringlist = []
        model.summary(line_length=78, print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        mlflow.log_text(model_summary, "model/model_arch.txt")

        print("starting model.fit...", flush=True)
        model.fit(train_gen,
                  validation_data=valid_gen,
                  epochs=epochs,
                  steps_per_epoch=train_samples / batch_size,
                  validation_steps=val_samples / batch_size,
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


if __name__ == "__main__":
    """ Call train_model() from the commandline, as the default entrypoint
    per standard usage of the MLflow projects call.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size")
    parser.add_argument("--epochs")
    parser.add_argument("--convolutions")
    parser.add_argument("--training-samples")
    parser.add_argument("--validation-samples")
    parser.add_argument("--randomize-images")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    print("starting train_model...", flush=True)
    print("using args", args, flush=True)
    exit()
    # train_model(
    #     batch_size=int(args.batch_size),
    #     epochs=int(args.epochs),
    #     convolutions=int(args.convolutions),
    #     train_samples=int(args.training_samples),
    #     val_samples=int(args.validation_samples),
    #     randomize_images=bool(args.randomize_images),
    #     run_name=args.run_name,
    # )
