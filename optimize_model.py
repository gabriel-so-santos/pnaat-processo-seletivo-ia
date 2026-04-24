import os

import tensorflow as tf
from tensorflow.keras.models import load_model

keras_model = load_model('mnist_model.h5')

model = tf.lite.TFLiteConverter.from_keras_model(keras_model)
model.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = model.convert()

with open("mnist_model.tflite", "wb") as f:
    f.write(tflite_model)