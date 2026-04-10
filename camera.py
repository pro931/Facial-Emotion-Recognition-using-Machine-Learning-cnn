import random
from io import BytesIO
import cv2
import keras
from PIL import Image
from database import Emotion, db
from model import ERModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = ERModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def save_emotion_to_db(app, pred, rate, image):
    with app.app_context():
        emotion = Emotion(emotion=pred, rate=rate, image=image)
        db.session.add(emotion)
        db.session.commit()

class VideoCamera(object):
    def __init__(self, app):
        self.app = app
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred, pred_percentage = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(fr, '%s - %.2f%%' % (pred, pred_percentage), (x - 10, y - 10), font, 1.5, (255,0,0), 3)

            # Save emotion to database
            save_emotion_to_db(self.app, pred, pred_percentage / 100, fr)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

def get_emotion_face(app, img, image_name2):
    gray_fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    pred = "Neutral"

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred, pred_percentage = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, '%s - %.2f%%' % (pred, pred_percentage), (x - 10, y - 10), font, 0.4, (255,0,0),1)

    _, jpeg = cv2.imencode('.jpg', img)
    cv2.imwrite(image_name2, img)

    # Save emotion to database
    save_emotion_to_db(app, pred, pred_percentage / 100, img)

    return jpeg