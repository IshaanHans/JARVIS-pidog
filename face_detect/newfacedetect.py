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

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "face_data.pkl"
TOLERANCE       = 0.55
FRAME_SCALE     = 0.5
PROCESS_EVERY_N = 3
SPEAK_COOLDOWN  = 5
OWNER_NAME      = "luke"
ALSA_DEVICE     = "plughw:2,0"   # Google Voice HAT — card 2 (cards 0/1 are HDMI, silent)

def speak(text):
    print(f"[FACE] Speaking: {text}")

    if not shutil.which("espeak-ng"):
        print("[FACE] WARNING: espeak-ng not found. Install: sudo apt install espeak-ng")
        return

    fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    try:
        # Generate WAV with espeak-ng
        gen = subprocess.run(
            ['espeak-ng', '-w', wav_path,
             '-a', '200', '-g', '5', '-p', '50', '-s', '130', text],
            capture_output=True, timeout=15
        )
        if gen.returncode != 0:
            print(f"[FACE] espeak-ng failed: {gen.stderr.decode(errors='ignore').strip()}")
            return

        # Play through robot_hat Music (enables the amp correctly)
        from robot_hat import Music
        music = Music()
        music.sound_play(wav_path)

    except Exception as e:
        print(f"[FACE] speak() error: {e}")
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
# ─────────────────────────────────────────────
# SPEAK
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────

def audio_self_test():
    print("[FACE] Running audio self-test...")
    print(f"[FACE]   Target device : {ALSA_DEVICE}")
    print(f"[FACE]   espeak-ng     : {'OK' if shutil.which('espeak-ng') else 'NOT FOUND'}")
    print(f"[FACE]   aplay         : {'OK' if shutil.which('aplay') else 'NOT FOUND'}")

    fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        gen = subprocess.run(
            ['espeak-ng', '-w', wav_path, '-a', '200', '-s', '130',
             "Audio test. If you hear this, sound is working."],
            capture_output=True, timeout=15
        )
        if gen.returncode != 0:
            print(f"[FACE]   espeak-ng write FAILED: {gen.stderr.decode(errors='ignore').strip()}")
            return False

        play = subprocess.run(
            ['aplay', '-q', '-D', ALSA_DEVICE, wav_path],
            capture_output=True, timeout=15
        )
        if play.returncode == 0:
            print("[FACE]   Playback: OK — you should have heard the test phrase.")
            return True
        else:
            print(f"[FACE]   Playback FAILED ({play.returncode}): "
                  f"{play.stderr.decode(errors='ignore').strip()}")
            print(f"[FACE]   Manual test: espeak-ng -w /tmp/t.wav hello && aplay -D {ALSA_DEVICE} /tmp/t.wav")
            return False
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# ENCODE KNOWN FACES → face_data.pkl
# ─────────────────────────────────────────────

def encode_known_faces():
    """
    Supports two layouts:
      known_faces/luke.jpg           (flat — one photo per person)
      known_faces/luke/img1.jpg      (folder — multiple photos, more accurate)
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
# OWNER / FACE CALLBACKS
# ─────────────────────────────────────────────

def on_owner_detected(confidence):
    print(f"[FACE] Owner detected! Confidence: {confidence:.0%}")
    speak(f"Hello {OWNER_NAME}, good to see you")

def on_owner_lost():
    print("[FACE] Owner left the frame.")

def on_unknown_detected():
    print("[FACE] Unknown face detected.")
    speak("I see an unknown person")

def on_known_detected(name):
    print(f"[FACE] Known person: {name}")
    speak(f"Hello {name}")


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
        print("[FACE] Camera started.")

        frame_count    = 0
        was_owner_here = False
        last_spoken    = {}

        while not self._stop_event.is_set():
            try:
                frame = picam2.capture_array()
            except Exception as e:
                print(f"[FACE] Frame capture error: {e}")
                time.sleep(0.5)
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N != 0:
                time.sleep(0.01)
                continue

            detections = identify_faces(frame, self.known_names, self.known_encs)

            with self._lock:
                self.current_detections = detections

            names_seen = [d["name"] for d in detections]
            owner_now  = OWNER_NAME in names_seen
            now        = time.time()

            # Owner enter/leave events
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

            # Cooldown greetings for all faces
            for d in detections:
                name      = d["name"]
                last_time = last_spoken.get(name, 0)
                if now - last_time > SPEAK_COOLDOWN:
                    if name == "Unknown":
                        on_unknown_detected()
                    elif name != OWNER_NAME:
                        on_known_detected(name)
                    last_spoken[name] = now
                print(f"[FACE] {name} ({d['confidence']:.0%}) @ {d['location']}")

            was_owner_here = owner_now
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

    if "--no-audio-test" not in sys.argv:
        audio_self_test()

    known_names, known_encs = load_encodings()
    if not known_encs:
        print("[FACE] No faces loaded.")
        print("  → Add photos to known_faces/ and run: python3 face.py --encode")
        sys.exit(1)

    face_thread = FaceRecognitionThread(known_names, known_encs)
    face_thread.start()
    print("[FACE] Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            detections = face_thread.get_detections()
            if detections:
                for d in detections:
                    label = "OWNER" if d["name"] == OWNER_NAME else d["name"]
                    print(f"  → {label} | conf: {d['confidence']:.0%} | loc: {d['location']}")
            else:
                print("  → No faces in frame")

    except KeyboardInterrupt:
        print("\n[FACE] Shutting down...")
        face_thread.stop()
        face_thread.join(timeout=3)
        print("[FACE] Done.")