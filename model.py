from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class ERModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_file, model_weights_file):

        with open(model_file, "r") as file:
            loaded_model_json = file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        pred_percentage = self.preds.max() * 100
        pred_percentage = round(pred_percentage, 2)
        print("pred_percentage", pred_percentage)
        print("self.preds[0][self.evl.argmax()] * 100", self.preds[0][self.preds.argmax()] * 100)
        return ERModel.EMOTIONS_LIST[np.argmax(self.preds)], pred_percentage

