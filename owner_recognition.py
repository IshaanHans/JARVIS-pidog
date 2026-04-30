import cv2
import os
import numpy as np
import time
from picamera2 import Picamera2

KNOWN_FACES_DIR = "/home/jarvis/wakeword/known_faces"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_faces():
    """Train OpenCV LBPH recogniser on known faces."""
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    images, labels, names = [], [], []
    label_map = {}
    current_label = 0

    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person
        for file in os.listdir(person_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img  = cv2.imread(os.path.join(person_dir, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                images.append(gray[y:y+h, x:x+w])
                labels.append(current_label)

        print(f"Trained: {person} ({len(images)} images)")
        current_label += 1

    recogniser.train(images, np.array(labels))
    print("Training complete!")
    return recogniser, label_map

def recognise_owner(recogniser, label_map, timeout=5):
    """Open camera and try to recognise owner. Returns name or None."""
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    cam.start()

    result   = None
    deadline = time.time() + timeout

    try:
        while time.time() < deadline:
            frame = cam.capture_array()
            gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = recogniser.predict(face_roi)

                # Lower confidence = better match (LBPH)
                print(f"Confidence: {confidence:.1f}")
                if confidence < 80:
                    result = label_map.get(label, "unknown")
                    print(f"Recognised: {result}")
                    return result
                else:
                    print("Unknown face")
    finally:
        cam.stop()

    return result