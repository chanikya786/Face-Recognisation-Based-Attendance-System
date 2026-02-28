# -----------------------------
# FACE RECOGNITION ATTENDANCE
# Simplified & Teaching Friendly
# -----------------------------

import cv2
import os
import joblib
import pandas as pd
import numpy as np
from datetime import date, datetime
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# BASIC SETUP
# -----------------------------

app = Flask(__name__)

# Project folders
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FACES_DIR = os.path.join(BASE_DIR, "static", "faces")
MODEL_PATH = os.path.join(BASE_DIR, "static", "face_recognition_model.pkl")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Date formatting for attendance file
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Attendance folder and file
ATT_DIR = os.path.join(BASE_DIR, "Attendance")
os.makedirs(ATT_DIR, exist_ok=True)
ATT_FILE = os.path.join(ATT_DIR, f"Attendance-{datetoday}.csv")

# Create CSV with header if not exists
if not os.path.isfile(ATT_FILE):
    with open(ATT_FILE, "w") as f:
        f.write("Name,Roll,Time\n")

# Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def total_registered():
    """Count how many users are registered."""
    if not os.path.isdir(FACES_DIR):
        return 0
    return len(os.listdir(FACES_DIR))


def capture_images(name, roll):
    """
    Capture 25 face images from webcam for training.
    Stores them in: static/faces/name_roll
    """
    folder = os.path.join(FACES_DIR, f"{name}_{roll}")
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0
    target = 25

    while count < target:
        ret, frame = cam.read()
        if not ret:
            break

        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_crop)
            count += 1

        cv2.putText(frame, f"Images: {count}/{target}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
        cv2.imshow("Registering User", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def train_model():
    """
    Train KNN classifier on saved face images.
    Converts each image to 50x50 and flattens.
    """
    faces = []
    labels = []

    for user in os.listdir(FACES_DIR):
        folder = os.path.join(FACES_DIR, user)
        for imgname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, imgname))
            if img is None:
                continue
            resized = cv2.resize(img, (50, 50))
            faces.append(resized.flatten())
            labels.append(user)

    if len(faces) == 0:
        return False

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, MODEL_PATH)
    return True


def recognize_face():
    """
    Detect face once using webcam, predict identity using trained model.
    Returns either 'name_roll' or None.
    """
    model = joblib.load(MODEL_PATH)
    cam = cv2.VideoCapture(0)
    identity = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identity = model.predict(cropped.reshape(1, -1))[0]
            break

        cv2.imshow("Taking Attendance", frame)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return identity


def mark_attendance(person):
    """
    Adds attendance if not already marked today.
    Returns 'success' or 'duplicate'.
    """
    name, roll = person.split("_")
    df = pd.read_csv(ATT_FILE, dtype=str)

    if roll in df["Roll"].values:
        return "duplicate"

    with open(ATT_FILE, "a") as f:
        f.write(f"{name},{roll},{datetime.now().strftime('%H:%M:%S')}\n")
    return "success"


def read_attendance():
    """Read today's attendance for display."""
    df = pd.read_csv(ATT_FILE, dtype=str)
    return df["Name"].tolist(), df["Roll"].tolist(), df["Time"].tolist(), len(df)


# -----------------------------
# FLASK ROUTES
# -----------------------------

@app.route("/")
def home():
    names, rolls, times, count = read_attendance()
    return render_template("home.html",
                           datetoday2=datetoday2,
                           totalreg=total_registered(),
                           names=names, rolls=rolls, times=times, l=count)


@app.route("/add", methods=["POST"])
def add():
    name = request.form["newusername"].strip()
    roll = request.form["newuserid"].strip()

    if len(roll) == 1:
        roll = "0" + roll

    capture_images(name, roll)
    trained = train_model()

    msg = f"User {name} ({roll}) added successfully" if trained else "Training failed"
    return render_template("home.html", mess=msg, mtype="success",
                           datetoday2=datetoday2, totalreg=total_registered(),
                           *read_attendance())


@app.route("/start")
def start():
    person = recognize_face()

    if not person:
        msg = "No face detected"
        mtype = "warning"
    else:
        result = mark_attendance(person)
        name, roll = person.split("_")
        if result == "success":
            msg = f"Present marked for {name} ({roll})"
            mtype = "success"
        else:
            msg = f"{name} ({roll}) attendance already recorded today"
            mtype = "warning"

    names, rolls, times, count = read_attendance()
    return render_template("home.html", mess=msg, mtype=mtype,
                           datetoday2=datetoday2, totalreg=total_registered(),
                           names=names, rolls=rolls, times=times, l=count)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
