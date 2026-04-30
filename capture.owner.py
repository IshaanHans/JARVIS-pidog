import cv2
import os
import shutil

SAVE_DIR = "/home/jarvis/wakeword/known_faces/luke"
UPLOAD_DIR = "/home/jarvis/uploads"  # change to your upload folder
os.makedirs(SAVE_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

valid = 0
for file in os.listdir(UPLOAD_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(UPLOAD_DIR, file)
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        dst = os.path.join(SAVE_DIR, f"luke_{valid}.jpg")
        shutil.copy(path, dst)
        print(f"✓ Saved: {file}")
        valid += 1
    else:
        print(f"✗ Skipped: {file} (no face found)")

print(f"\nDone! {valid} photo(s) saved")