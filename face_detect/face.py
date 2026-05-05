# face.py

import os
import sys
import cv2
import numpy as np
import face_recognition
import subprocess
import threading
import time
import pickle

try:
    from picamera2 import Picamera2
except ImportError:
    print("[FACE] picamera2 not found. Install with: sudo apt install python3-picamera2")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "face_data.pkl"
TOLERANCE       = 0.55
FRAME_SCALE     = 0.5
PROCESS_EVERY_N = 3
SPEAK_COOLDOWN  = 5       # seconds between repeating the same name
OWNER_NAME      = "luke"  # triggers owner-specific behaviour

# ─────────────────────────────────────────────
# SPEAK
# ─────────────────────────────────────────────

def speak(text):
    print(f"[FACE] Speaking: {text}")
    subprocess.run(
        ['espeak-ng', '-a', '200', '-g', '5', '-p', '50', '-s', '130', text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ─────────────────────────────────────────────
# ENCODE — build face_data.pkl from known_faces/
# ─────────────────────────────────────────────

def encode_known_faces():
    """
    Supports both layouts:
      known_faces/luke.jpg          (flat)
      known_faces/luke/img1.jpg     (folder — recommended for multiple photos)
    """
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
                print(f"  [!] No face found in: {img_path} — skipping")

        if encodings:
            known_encodings[name] = encodings
            print(f"  [{name}] {len(encodings)} encoding(s) saved.")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_encodings, f)

    total = sum(len(v) for v in known_encodings.values())
    print(f"\n[FACE] Done. {total} encoding(s) for {len(known_encodings)} person(s) → {ENCODINGS_FILE}")

# ─────────────────────────────────────────────
# LOAD ENCODINGS
# ─────────────────────────────────────────────

def load_encodings():
    """
    Returns flat parallel lists for fast comparison.
    Falls back to scanning known_faces/ directly if no .pkl exists.
    """
    # Try cached pkl first
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

    # Fallback: load directly from known_faces/ (your original approach)
    print(f"[FACE] No {ENCODINGS_FILE} found — loading directly from '{KNOWN_FACES_DIR}'...")
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

    print(f"[FACE] Loaded {len(known_names)} face(s) directly from folder.")
    return known_names, known_encs

# ─────────────────────────────────────────────
# IDENTIFY FACES IN A FRAME
# ─────────────────────────────────────────────

def identify_faces(frame, known_names, known_encs):
    """
    Takes an RGB frame (from Picamera2), returns list of dicts:
      [{"name": "luke", "confidence": 0.87, "location": (top, right, bottom, left)}, ...]
    """
    small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

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
# OWNER CALLBACKS — plug in your PiDog actions
# ─────────────────────────────────────────────

def on_owner_detected(confidence):
    print(f"[FACE] 👋 Owner detected! Confidence: {confidence:.0%}")
    speak(f"Hello {OWNER_NAME}, good to see you")
    # pidog.do_action("wag_tail", speed=80)
    # set_mood(MoodState.HAPPY)

def on_owner_lost():
    print("[FACE] Owner left the frame.")
    # pidog.do_action("sit")

def on_unknown_detected():
    print("[FACE] ⚠ Unknown face detected.")
    speak("I see an unknown person")
    # set_mood(MoodState.CAUTIOUS)

def on_known_detected(name):
    print(f"[FACE] Known person: {name}")
    speak(f"Hello {name}")

# ─────────────────────────────────────────────
# CAMERA THREAD (Picamera2)
# ─────────────────────────────────────────────

class FaceRecognitionThread(threading.Thread):
    """
    Daemon thread running Picamera2 capture + face recognition loop.

    Exposes:
      .current_detections  — latest list of face dicts (thread-safe via get_detections())
      .owner_present       — True if owner is currently visible
      .stop()              — graceful shutdown
    """

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

    def run(self):
        # Init Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        ))
        picam2.start()
        print("[FACE] Picamera2 started.")

        frame_count    = 0
        was_owner_here = False
        last_spoken    = {}   # name -> last spoken timestamp

        while not self._stop_event.is_set():
            try:
                frame = picam2.capture_array()
            except Exception as e:
                print(f"[FACE] Frame capture error: {e}")
                time.sleep(0.5)
                continue

            frame_count += 1

            # Only run recognition every Nth frame to reduce CPU load
            if frame_count % PROCESS_EVERY_N != 0:
                time.sleep(0.01)
                continue

            detections = identify_faces(frame, self.known_names, self.known_encs)

            with self._lock:
                self.current_detections = detections

            names_seen = [d["name"] for d in detections]
            owner_now  = OWNER_NAME in names_seen
            now        = time.time()

            # ── Owner state change callbacks ──
            if owner_now and not was_owner_here:
                best = max(
                    (d for d in detections if d["name"] == OWNER_NAME),
                    key=lambda d: d["confidence"]
                )
                self.owner_present = True
                on_owner_detected(best["confidence"])
                last_spoken[OWNER_NAME] = now

            elif not owner_now and was_owner_here:
                self.owner_present = False
                on_owner_lost()

            # ── Per-face speak with cooldown ──
            for d in detections:
                name      = d["name"]
                last_time = last_spoken.get(name, 0)

                if now - last_time > SPEAK_COOLDOWN:
                    if name == "Unknown":
                        on_unknown_detected()
                    elif name != OWNER_NAME:
                        on_known_detected(name)
                    # Owner greeting already handled in state change above
                    last_spoken[name] = now

                print(f"[FACE] {name} ({d['confidence']:.0%}) @ {d['location']}")

            was_owner_here = owner_now
            time.sleep(0.01)

        picam2.stop()
        print("[FACE] Camera thread stopped.")

    def get_detections(self):
        """Thread-safe snapshot of latest detections."""
        with self._lock:
            return list(self.current_detections)

# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Encode faces into face_data.pkl
    if "--encode" in sys.argv:
        encode_known_faces()
        sys.exit(0)

    # Load face data
    known_names, known_encs = load_encodings()
    if not known_encs:
        print("[FACE] No faces loaded.")
        print("  → Add photos to known_faces/ and run: python3 face.py --encode")
        sys.exit(1)

    # Start recognition thread
    face_thread = FaceRecognitionThread(known_names, known_encs)
    face_thread.start()

    print("[FACE] Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            detections = face_thread.get_detections()
            if detections:
                for d in detections:
                    label = "✓ OWNER" if d["name"] == OWNER_NAME else d["name"]
                    print(f"  → {label} | conf: {d['confidence']:.0%} | loc: {d['location']}")
            else:
                print("  → No faces in frame")

    except KeyboardInterrupt:
        print("\n[FACE] Shutting down...")
        face_thread.stop()
        face_thread.join(timeout=3)
        print("[FACE] Done.")