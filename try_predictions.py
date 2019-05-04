import numpy as np
import matplotlib.pyplot as plt
import keras
import itertools
import os
import cv2
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import pickle
import PIL
import io
import base64
from prettytable import PrettyTable



def main():

    with open ('model/config/config.cfg', 'rb') as fp:
        classi = pickle.load(fp)
    print("Classi caricate correttamente")

    def get_model():

        global model

        global graph

        graph = tf.get_default_graph()

        vgg19_model = keras.applications.vgg19.VGG19()

        vgg19_model.layers.pop()

        model = Sequential()

        for layer in vgg19_model.layers:
            model.add(layer)

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(len(classi), activation='softmax'))

        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights('model/h5_weights.h5')


        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


        print("Model loaded successfully")


    def preprocess_image(image, target_size):
        if image.mode != "RGB":
            image = image.convert("RGB")
            image = image.resize(target_size)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            return image
    print("---")


    try:
        print("Loading model...")
        get_model()
    except:
        print("Model failed to load, try running main script again.")
        return
    while True:
        img_path = input("Type path to the image (with extension): ")
        try:
            with open(img_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read())
        except:
            print("Error in the image filepath/name, try again...")

        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        try:
            processed_image = preprocess_image(image, target_size=(224, 224))
        except:
            print("Error while processing the given input image, are you sure it's an image?")
            return

        prediction = model.predict(processed_image).tolist()

        print(classi)

        print(prediction)
        t = PrettyTable()
        names = np.asarray(classi)
        t.field_names = names
        for value in prediction:
            t.add_row(value)
        print(t)

if __name__ == '__main__':
    main()