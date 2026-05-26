import os
import sys
import cv2
import numpy as np
import face_recognition
import subprocess
import threading
import tempfile
import time
import pickle
import shutil

try:
    from picamera2 import Picamera2
except ImportError:
    print("[FACE] picamera2 not found. Install with: sudo apt install python3-picamera2")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KNOWN_FACES_DIR  = "known_faces"
ENCODINGS_FILE   = "face_data.pkl"
TOLERANCE        = 0.55
FRAME_SCALE      = 0.5
PROCESS_EVERY_N  = 3
OWNER_NAME       = "luke"

LAST_SEEN_FILE   = "/tmp/jarvis_last_seen.txt"
VISION_SNAPSHOT  = "/tmp/jarvis_vision.jpg"

def write_last_seen(name):
    try:
        with open(LAST_SEEN_FILE, "w") as f:
            f.write(name)
    except Exception as e:
        print(f"[FACE] Could not write last seen: {e}")

def write_vision_snapshot(frame_rgb):
    try:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(VISION_SNAPSHOT, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    except Exception as e:
        print(f"[FACE] Could not write vision snapshot: {e}")


# ─────────────────────────────────────────────
# ENCODE KNOWN FACES → face_data.pkl
# ─────────────────────────────────────────────

def encode_known_faces():
    known_encodings = {}
    print(f"[FACE] Encoding faces from '{KNOWN_FACES_DIR}'...")

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[FACE] ERROR: '{KNOWN_FACES_DIR}' folder not found.")
        return

    for entry in os.listdir(KNOWN_FACES_DIR):
        entry_path = os.path.join(KNOWN_FACES_DIR, entry)

        if os.path.isfile(entry_path) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
            name   = os.path.splitext(entry)[0].lower()
            images = [entry_path]
        elif os.path.isdir(entry_path):
            name   = entry.lower()
            images = [
                os.path.join(entry_path, f)
                for f in os.listdir(entry_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        else:
            continue

        encodings = []
        for img_path in images:
            img     = face_recognition.load_image_file(img_path)
            results = face_recognition.face_encodings(img)
            if results:
                encodings.append(results[0])
                print(f"  [+] Encoded: {img_path}")
            else:
                print(f"  [!] No face in: {img_path} — skipping")

        if encodings:
            known_encodings[name] = encodings
            print(f"  [{name}] {len(encodings)} encoding(s) saved.")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_encodings, f)

    total = sum(len(v) for v in known_encodings.values())
    print(f"[FACE] Done. {total} encoding(s) for {len(known_encodings)} person(s) → {ENCODINGS_FILE}")


# ─────────────────────────────────────────────
# LOAD ENCODINGS
# ─────────────────────────────────────────────

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_names, known_encs = [], []
        for name, encs in data.items():
            for enc in encs:
                known_names.append(name)
                known_encs.append(enc)
        print(f"[FACE] Loaded {len(known_encs)} encoding(s) for: {list(data.keys())}")
        return known_names, known_encs

    print(f"[FACE] No {ENCODINGS_FILE} — loading directly from '{KNOWN_FACES_DIR}'...")
    known_names, known_encs = [], []

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[FACE] ERROR: '{KNOWN_FACES_DIR}' not found.")
        return [], []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name       = os.path.splitext(filename)[0].lower()
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image      = face_recognition.load_image_file(image_path)
        encodings  = face_recognition.face_encodings(image)
        if encodings:
            known_encs.append(encodings[0])
            known_names.append(name)
            print(f"  Loaded: {name}")
        else:
            print(f"  [!] No face found in {filename}, skipping")

    print(f"[FACE] Loaded {len(known_names)} face(s) from folder.")
    return known_names, known_encs


# ─────────────────────────────────────────────
# IDENTIFY FACES IN A FRAME
# ─────────────────────────────────────────────

def identify_faces(frame, known_names, known_encs):
    small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    locations = face_recognition.face_locations(small, model="hog")
    encodings = face_recognition.face_encodings(small, locations)

    results = []
    for enc, loc in zip(encodings, locations):
        name       = "Unknown"
        confidence = 0.0

        if known_encs:
            distances = face_recognition.face_distance(known_encs, enc)
            best_idx  = int(np.argmin(distances))
            best_dist = distances[best_idx]
            if best_dist <= TOLERANCE:
                name       = known_names[best_idx]
                confidence = round(1.0 - float(best_dist), 2)

        scale = int(1 / FRAME_SCALE)
        top, right, bottom, left = [v * scale for v in loc]
        results.append({
            "name":       name,
            "confidence": confidence,
            "location":   (top, right, bottom, left)
        })

    return results


# ─────────────────────────────────────────────
# FACE RECOGNITION THREAD
# ─────────────────────────────────────────────

class FaceRecognitionThread(threading.Thread):

    def __init__(self, known_names, known_encs):
        super().__init__(daemon=True)
        self.known_names        = known_names
        self.known_encs         = known_encs
        self.current_detections = []
        self.owner_present      = False
        self._stop_event        = threading.Event()
        self._lock              = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def get_detections(self):
        with self._lock:
            return list(self.current_detections)

    def run(self):
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        ))
        picam2.start()
        print("[FACE] Camera started. Running silently — feeding names to JARVIS.")

        frame_count = 0

        while not self._stop_event.is_set():
            try:
                frame = picam2.capture_array()
            except Exception as e:
                print(f"[FACE] Frame capture error: {e}")
                time.sleep(0.5)
                continue

            frame_count += 1

            # Always write vision snapshot for chatbot
            write_vision_snapshot(frame)

            if frame_count % PROCESS_EVERY_N != 0:
                time.sleep(0.01)
                continue

            detections = identify_faces(frame, self.known_names, self.known_encs)

            with self._lock:
                self.current_detections = detections

            known = [d for d in detections if d["name"] != "Unknown"]
            if known:
                best = max(known, key=lambda d: d["confidence"])
                write_last_seen(best["name"])
                print(f"[FACE] Detected: {best['name']} ({best['confidence']:.0%})")

            time.sleep(0.01)

        picam2.stop()
        print("[FACE] Camera stopped.")


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if "--encode" in sys.argv:
        encode_known_faces()
        sys.exit(0)

    known_names, known_encs = load_encodings()
    if not known_encs:
        print("[FACE] No faces loaded.")
        print("  → Add photos to known_faces/ and run: python3 face_detect.py --encode")
        sys.exit(1)

    face_thread = FaceRecognitionThread(known_names, known_encs)
    face_thread.start()
    print("[FACE] Running silently. Feeding last seen name to JARVIS.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[FACE] Shutting down...")
        face_thread.stop()
        face_thread.join(timeout=3)
        print("[FACE] Done.")
