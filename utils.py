"""
This module supports the define_data_generator_dataframe_custom() function in
train.py.  Sorry still really messy; work in progress...  -AG

Define bespoke TF data generator for images that require custom data-loading
processes to access (as opposed to just being in jpg/png/etc files).

With credit to Arjun Muraleedharan's Apr 2021 post in Medium:
https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
and Deepak Raj's Sept 2021 post in StackOverflow:
https://stackoverflow.com/questions/63827339/how-to-build-a-custom-data-generator-for-keras-tf-keras-where-x-images-are-being
(sortof combined those together here)
"""

import math
import numpy as np
from PIL import Image
import tensorflow as tf


class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 df,
                 X_col,
                 y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True,
                 augmentation=False,
                 ):

        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.n = len(self.df)
        self.n_name = df[y_col['label']].nunique()
        # self.n_type = df[y_col['type']].nunique()

    def on_epoch_end(self):
        # Another source mentioned that shuffle on epoch end was not working;
        # in that case, note commented out version in __len__() as alternative.
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __data_augmentation(self, img):
        img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    # def __get_input(self, path, target_size):
    #     # Bbox cropping unused in current application but keeping bbox var to
    #     # avoid changing broader code structure.  It's ok that bbox is null here.

    #     # def __get_input(self, path, bbox, target_size):

    #     # xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    #     # image = tf.keras.preprocessing.image.load_img(path)
    #     # print("DEBUG GET_INPUT:", path)
    #     image = dl.load_preview_image(path)

    #     # image = self.__data_augmentation(image)
    #     image_arr = tf.keras.preprocessing.image.img_to_array(image)
    #     # image_arr = image_arr[ymin:ymin + h, xmin:xmin + w]
    #     # print("DEBUG OUTPUT:", target_size, flush=True)
    #     image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

    #     return image_arr / 255.

    # def __get_output(self, label, num_classes):
    #     # print("DEBUG OUT:", num_classes, label, flush=True)
    #     return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    # def __get_data(self, batches):
    #     # Generates data containing batch_size samples

    #     path_batch = batches[self.X_col['path']]
    #     # bbox_batch = batches[self.X_col['bbox']]
    #     # bbox is  unused in this application but but leaving placeholder there
    #     # to avoid changing broader code structure

    #     name_batch = batches[self.y_col['label']]
    #     # type_batch = batches[self.y_col['type']]

    #     # X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])
    #     X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

    #     y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
    #     # y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

    #     # return X_batch, tuple([y0_batch, y1_batch])
    #     return X_batch, y0_batch

    def __get_image(self, path):
        """ open image with file_id path and apply data augmentation """
        img = np.asarray(Image.open(file_id))
        # (or some special data-file-loading function for custom data files;
        # PIL.Image.open above is just placeholder - built-in funcs exist for that)
        img = np.array(img)  # (W0, H0, 3) Numpy.array
        img = tf.image.resize(img, self.input_size)  # (W1, H1, 3) TF tensor
        if self.augmentation:
            img = self.__data_augmentation(img)
        # img = preprocess_input(img)
        return img

    def __get_label(self, label_id):
        """ uncomment the below line to convert label into categorical format """
        # label_id = tf.keras.utils.to_categorical(label_id, num_classes)
        return label_id

    def __getitem__(self, index):
        # goal - we want batches to be able to contain (much) more samples than
        # the length of df as long as augmentation is on; we could just turn
        # that on for all and just keep cycling df via modulo index?  but I
        # think doesn't do that yet here...

        # if (index + 1) * self.batch_size > self.df.shape[0]:
        #     print("Error: (why not caught at beginning?): #samples>databasequeryresults but augmentation off.")

        # batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        # X, y = self.__get_data(batches)
        # return tf.convert_to_tensor(X), tf.convert_to_tensor(y)
        batch_index_range = (index * self.batch_size, (index + 1) * self.batch_size)
        batch_x = self.df.loc[batch_index_range[0]:batch_index_range[1], "previewname"]
        batch_y = self.df.loc[batch_index_range[0]:batch_index_range[1], "label"]
        x = [self.__get_image(file) for file in batch_x]
        y = [self.__get_label(label) for label in batch_y]

        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    def __len__(self):
        # Another source mentioned that shuffle on epoch end was not working
        # and tried doing it in __len__() as alternative.  Uncomment if needed.
        # if self.shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)

        return math.ceil(self.n / self.batch_size)
