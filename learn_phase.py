import numpy as np
import matplotlib.pyplot as plt
import keras
import itertools
import os

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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import pickle
import PIL


def main():
    print("...")
    print("learning phase inizialized, this may take a while")
    print("...")


    train_path = 'model/train'
    valid_path = 'model/valid'
    test_path = 'model/test'

    wkdir = ''

    pb_filename = 'pesi_pb.pb'
    pbtxt_filename = 'pesi_pb.pbtxt'

    # %%

    with open('model/config/config.cfg', 'rb') as fp:
        classi = pickle.load(fp)
    print("Classi caricate correttamente")

    # %%
    for i in classi:
        questa = 'model/config/' + i + '.cfg'
        with open(questa, 'rb') as fp:
            batchSizes = pickle.load(fp)


    trainSize = 10
    testSize = 10
    validSize = 10

    print("train size totale: " + str(trainSize))
    print("test size totale: " + str(testSize))
    print("valid size totale: " + str(validSize))

    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=classi,
                                                             batch_size=trainSize)
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=classi,
                                                            batch_size=testSize)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=classi,
                                                             batch_size=validSize)

    stepsEpoche = len(train_batches) / 10
    stepsValid = len(valid_batches) / 10
    numEpoche = 10

    vgg19_model = keras.applications.vgg19.VGG19()

    # ELIMINA L'ULTIMO LAYER CON I 1000 OUTPUT
    vgg19_model.layers.pop()

    model = Sequential()

    for layer in vgg19_model.layers:
        layer.trainable = False
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(len(classi), activation='softmax', name='predictions'))

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    hist = model.fit_generator(train_batches, steps_per_epoch=stepsEpoche, validation_data=valid_batches,
                               validation_steps=stepsValid, epochs=numEpoche, verbose=2)
    json_config = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)
    json_file.close()

    # Save weights to disk

    model.save_weights('model/h5_weights.h5')
    model.save('model/h5_model.h5')

    print("Pesi del modello salvati sul disco come 'h5_weights.h5' ")
    print("Modello salvati sul disco come 'h5_model.h5' ")
    print("JSON del modello salvati sul disco come 'model_config.json' ")

    return "ok, learning phase successfully completed! Bye!"

if __name__ == '__main__':
    main()
