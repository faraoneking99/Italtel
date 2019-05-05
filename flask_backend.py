import base64
import numpy as np
import io
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
from keras.engine.saving import model_from_json
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import pickle

#from flask_cors import CORS

app = Flask(__name__)
#CORS(app)

#run server
if __name__ == '__main__':
    app.run(debug=True)

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

    #model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights('model/h5_weights.h5')

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


    print("Modello caricato corettamente")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image


print("Carico modello...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    with graph.as_default():
        prediction = model.predict(processed_image).tolist()
    #prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    return jsonify(response)
