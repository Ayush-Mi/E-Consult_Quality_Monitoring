from flask import Flask, render_template, Response
import cv2
from face_detection import FaceDetection
from emotion_recognition import EmotionRecognition
import cv2
import tensorflow as tf
import os
import time

app = Flask(__name__)
# Creating Face Detection Object
fd = FaceDetection()
# Creating Face Recognition Object
er = EmotionRecognition()
# Capturing video from camera
video = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    count = 0
    emotion = ' ' 
    while True:
        count += 1
        ret, frame = video.read()
        try:
            # Predicting on every 10th frame only
            if count%10 == 0:
                face = fd.detect_face(frame)
                emotion = er.pred_emotion(face[0])
                frame = cv2.putText(frame, emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (255, 0, 0), 2, cv2.LINE_AA)
                x, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            else:
                frame = cv2.putText(frame, emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (255, 0, 0), 2, cv2.LINE_AA)
                x, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            continue

if __name__ == "__main__":
    app.run(debug=True)