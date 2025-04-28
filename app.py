import os
from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import face_recognition
import pickle
import time
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import config

# Initialize Firebase
cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {
    'databaseURL': config.FIREBASE_DATABASE_URL
})

# Initialize Flask app
app = Flask(__name__)

# Load encoding file
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownIds


# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')


# Helper function for processing the image
def process_image(image_data):
    # Decode and process image
    image_data = base64.b64decode(image_data.split(',')[1])
    np_img = np.array(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(np_img, -1)

    # Face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    if faceCurrentFrame:
        for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                id = studentIds[matchIndex]
                student_ref = db.reference(f'Students/{id}')
                student_info = student_ref.get()

                if student_info:
                    # Convert the 'last_attendance_time' string to a datetime object
                    last_attendance_time_str = student_info.get('last_attendance_time')
                    if last_attendance_time_str:
                        last_attendance_time = datetime.strptime(last_attendance_time_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        last_attendance_time = datetime.now()  # If no previous attendance time exists

                    # Get current time
                    current_time = datetime.now()

                    # Calculate the difference in seconds
                    time_difference = (current_time - last_attendance_time).total_seconds()

                    if time_difference > 30:  # Attendance can only be marked after 30 seconds
                        total_attendance = student_info.get('total_attendance', 0) + 1
                        student_ref.update({
                            'total_attendance': total_attendance,
                            'last_attendance_time': current_time.strftime('%Y-%m-%d %H:%M:%S')  # Store as string
                        })
                        print(f"Attendance marked for {student_info['name']}")
                        return {"status": "success", "message": f"Attendance marked for {student_info['name']}"}
                    else:
                        return {"status": "fail", "message": "Recently marked, wait before next scan"}

    return {"status": "fail", "message": "No face recognized"}


# Route to handle face recognition request
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.get_json()
    image_data = data['image']
    result = process_image(image_data)
    return jsonify(result)


# Run Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

