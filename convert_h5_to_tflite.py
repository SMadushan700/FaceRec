import tensorflow as tf
import keras
from keras.models import load_model

emotion_model = load_model('emotion_detection_model_file.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

open("emotion_detection_model_opt.tflite", "wb").write(tflite_model)
