import json
import os
from flask import Flask, render_template, Response, request, jsonify, send_file, url_for
from camera import VideoCamera, get_emotion_face
import cv2
from werkzeug.utils import secure_filename
from flask_cors import CORS
from database import create_data, get_emotions, db
from datetime import datetime
from flask import jsonify, json as jsonn

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotions_db.db'
app.config['UPLOAD_FOLDER'] = 'images/'
CORS(app)

def test_connection():
    with app.app_context():
        db.init_app(app)
        db.create_all()
        create_data()

test_connection()

@app.route('/dashboard')
def dashboard():
    emotions = get_emotions()
    print("emotions", len(emotions))
    data = json.dumps([{'date': str(emotion.date), 'emotion': emotion.emotion, 'rate': emotion.rate, 'count': emotion.count} for emotion in emotions])
    return {'emotions': data}, 200

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    image_file = request.files['image']
    image_filename = secure_filename(image_file.filename)
    image_name2 = 'images/result' + image_filename
    image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    print("image_file", image_file)
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    img = get_emotion_face(app, image, image_name2)
    return send_file(image_name2, mimetype='image/jpg')

@app.route('/')
def index():
    print("!" * 99)
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(app)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)