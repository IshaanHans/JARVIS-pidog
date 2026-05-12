import cv2
import numpy as np
import face_recognition
import subprocess
import os
import re
import time
from picamera2 import Picamera2
import shutil


def _parse_aplay_cards():
    """Parse `aplay -l` into list of dicts: card, dev, line, is_hdmi, score."""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        out = []
        for line in result.stdout.splitlines():
            if not line.startswith('card '):
                continue
            m = re.match(
                r'card (\d+):\s*([^,]+),\s*device (\d+):\s*(.+)',
                line.strip(),
            )
            if not m:
                continue
            card_s, _name1, dev_s, _name2 = m.groups()
            card_num = int(card_s)
            dev_num = int(dev_s)
            low = line.lower()
            is_hdmi = 'hdmi' in low or 'vc4hdmi' in low
            # Prefer Pi jack / built-in, then generic analog, then USB. Avoid picking
            # "last non-HDMI" — that is often the wrong interface (e.g. USB gadget).
            if is_hdmi:
                score = -1000
            elif 'headphone' in low or 'bcm2835' in low or 'headphones' in low:
                score = 500
            elif 'wm8960' in low or 'i2s' in low or 'hifiberry' in low:
                score = 400
            elif 'googlevoice' in low or 'voicehat' in low or 'voice-hat' in low:
                score = 450
            elif 'robot-hat' in low or 'robot_hat' in low or 'sndrpigooglevoi' in low:
                score = 455
            elif 'usb' in low:
                score = 100
            else:
                score = 200
            out.append({
                'card': card_num,
                'dev': dev_num,
                'line': line.strip(),
                'is_hdmi': is_hdmi,
                'score': score,
            })
        return out
    except Exception as e:
        print(f"Audio device listing failed: {e}")
        return []


def _get_audio_device():
    """
    Pick best ALSA playback device for speech on Raspberry Pi.
    Uses scoring + lowest card id on tie (not "last in list").
    """
    cards = _parse_aplay_cards()
    if not cards:
        return None
    non_hdmi = [c for c in cards if not c['is_hdmi']]
    pool = non_hdmi if non_hdmi else cards
    best = max(pool, key=lambda c: (c['score'], -c['card']))
    return f"plughw:{best['card']},{best['dev']}"


def _plughw_card_index(plughw):
    """Extract card number from 'plughw:N,M' or None."""
    if not plughw or not plughw.startswith('plughw:'):
        return None
    try:
        part = plughw.split(':', 1)[1]
        return int(part.split(',')[0])
    except (ValueError, IndexError):
        return None


def _unmute_alsa_output(card_index=None):
    """
    Unmute / raise volume on the relevant mixer.
    Without `-c N`, amixer only touches the default card — wrong if we play on plughw:2,0.

    If a specific card is requested, we must NOT fall back to the default card: that was
    unmoving the Voice HAT / HAT amp while Pulse still "succeeded" on HDMI.
    """
    def _try_sset(base_cmd, control):
        return subprocess.run(
            base_cmd + ['sset', control, '100%', 'unmute'],
            capture_output=True,
            timeout=5,
        )

    if card_index is not None:
        base = ['amixer', '-c', str(card_index)]
        # SunFounder PiDog / robot-hat: playback is *not* "Master" — use these names first.
        for mixer in [
            'robot-hat speaker',
            'sunfounder speaker',
            'Master', 'PCM', 'Speaker', 'Headphone', 'Digital',
            'Playback', 'Line Out', 'Mono', 'Boost',
        ]:
            r = _try_sset(base, mixer)
            if r.returncode == 0:
                return mixer, card_index

        sc = subprocess.run(
            ['amixer', '-c', str(card_index), 'scontrols'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dyn_names = re.findall(r"Simple mixer control '([^']+)'", sc.stdout)

        def _playback_first(name):
            low = name.lower()
            if 'speaker' in low or 'master' in low:
                return (0, name)
            if 'pcm' in low and 'mic' not in low:
                return (0, name)
            if 'mic' in low or 'capture' in low:
                return (2, name)
            return (1, name)

        dyn_names = sorted(dyn_names, key=_playback_first)
        for name in dyn_names:
            r = _try_sset(base, name)
            if r.returncode == 0:
                return name, card_index
            subprocess.run(
                base + ['sset', name, '100%'],
                capture_output=True,
                timeout=5,
            )
            r = subprocess.run(
                base + ['sset', name, 'unmute'],
                capture_output=True,
                timeout=5,
            )
            if r.returncode == 0:
                return name, card_index

        print(
            f"TTS warning: could not set volume on ALSA card {card_index}. "
            f"Check: amixer -c {card_index} scontrols; speaker may still be muted."
        )
        return None, card_index

    for mixer in ['Master', 'PCM', 'Speaker', 'Headphone', 'Digital']:
        r = _try_sset(['amixer'], mixer)
        if r.returncode == 0:
            return mixer, None
    return None, None


def _espeak_aplay_pipe(text, aplay_cmd):
    """Play espeak WAV on stdout through aplay. aplay_cmd must include '-t', 'wav'."""
    espeak_cmd = ['espeak-ng', '--stdout', '-a', '200', '-g', '5', '-p', '50', '-s', '130', text]
    espeak_proc = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    aplay_proc = subprocess.Popen(aplay_cmd, stdin=espeak_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    espeak_proc.stdout.close()
    _, espeak_err = espeak_proc.communicate(timeout=30)
    _, aplay_err = aplay_proc.communicate(timeout=30)
    return espeak_proc.returncode, aplay_proc.returncode, espeak_err, aplay_err


def _espeak_to_wav_file_then_aplay(text, aplay_cmd_prefix):
    """Fallback: write WAV with espeak-ng -w then play file (most reliable on Pi)."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        r = subprocess.run(
            ['espeak-ng', '-w', path, '-a', '200', '-g', '5', '-p', '50', '-s', '130', text],
            capture_output=True,
            timeout=30,
        )
        if r.returncode != 0:
            return r.returncode, -1, r.stderr, b''
        play = subprocess.run(
            aplay_cmd_prefix + [path],
            capture_output=True,
            timeout=30,
        )
        return 0, play.returncode, b'', play.stderr
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _espeak_write_wav(text, path):
    return subprocess.run(
        ['espeak-ng', '-w', path, '-a', '200', '-g', '5', '-p', '50', '-s', '130', text],
        capture_output=True,
        timeout=30,
    )


def _print_playback_devices():
    """Log ALSA devices once so wrong routing (e.g. USB vs jack) is obvious."""
    cards = _parse_aplay_cards()
    if not cards:
        print("  (no ALSA playback cards parsed from aplay -l)")
        return
    print("  ALSA playback devices:")
    for c in sorted(cards, key=lambda x: (x['card'], x['dev'])):
        tag = "HDMI" if c['is_hdmi'] else "analog/BT/USB"
        print(f"    plughw:{c['card']},{c['dev']}  [{tag}]  {c['line']}")
    chosen = _get_audio_device()
    print(f"  Auto-selected: {chosen or 'default'}  (override: PIDOG_ALSA_DEVICE=plughw:C,D)")


def speak(text):
    print(f"Speaking: {text}")

    if not shutil.which("espeak-ng"):
        print("TTS warning: espeak-ng not found, trying pyttsx3 fallback.")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"TTS error: fallback failed ({exc})")
        return

    env_dev = os.environ.get('PIDOG_ALSA_DEVICE', '').strip()
    audio_device = env_dev or _get_audio_device()
    card_idx = _plughw_card_index(audio_device) if audio_device else None
    mixer, mc = _unmute_alsa_output(card_idx)
    if mixer:
        print(f"Volume: amixer -c {mc if mc is not None else 'default'} {mixer} -> 100% unmute")

    aplay_with_dev = ['aplay', '-q', '-t', 'wav', '-D', audio_device] if audio_device else None
    aplay_default = ['aplay', '-q', '-t', 'wav']

    import tempfile
    fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        gen = _espeak_write_wav(text, wav_path)
        if gen.returncode != 0:
            print(f"TTS error: espeak-ng -w failed ({gen.returncode})")
            if gen.stderr:
                print(f"  {gen.stderr.decode(errors='ignore').strip()}")
            return

        use_paplay_first = (
            os.environ.get('PIDOG_USE_PAPLAY_FIRST', '').strip() in ('1', 'true', 'yes')
            or not audio_device
        )

        # Pulse default sink first only when requested or when we have no explicit ALSA
        # device. Otherwise paplay "succeeds" to HDMI/silent sink while the Voice HAT
        # (e.g. plughw:2,0) never gets the WAV.
        if use_paplay_first and shutil.which('paplay'):
            pr = subprocess.run(
                ['paplay', wav_path],
                capture_output=True,
                timeout=60,
            )
            if pr.returncode == 0:
                return
            err = pr.stderr.decode(errors='ignore').strip()
            if err and 'Connection refused' not in err:
                print(f"TTS note: paplay failed ({pr.returncode}): {err}")

        # ALSA explicit device (Voice HAT / USB DAC / bcm2835 jack)
        if aplay_with_dev:
            pr = subprocess.run(
                aplay_with_dev + [wav_path],
                capture_output=True,
                timeout=60,
            )
            if pr.returncode == 0:
                return
            err = pr.stderr.decode(errors='ignore').strip()
            print(f"TTS warning: aplay -D {audio_device} failed ({pr.returncode}): {err}")

        # ALSA default device
        pr = subprocess.run(
            aplay_default + [wav_path],
            capture_output=True,
            timeout=60,
        )
        if pr.returncode == 0:
            return
        err = pr.stderr.decode(errors='ignore').strip()
        print(f"TTS warning: aplay (default) failed ({pr.returncode}): {err}")

        # Late Pulse try: explicit ALSA path failed but a desktop sink might work
        if not use_paplay_first and shutil.which('paplay'):
            pr = subprocess.run(
                ['paplay', wav_path],
                capture_output=True,
                timeout=60,
            )
            if pr.returncode == 0:
                return
            err = pr.stderr.decode(errors='ignore').strip()
            if err and 'Connection refused' not in err:
                print(f"TTS note: paplay failed ({pr.returncode}): {err}")

        # 4) Pipe to aplay (stdin)
        try:
            pipe_aplay_cmd = aplay_with_dev if aplay_with_dev else aplay_default
            ec, ac, espeak_err, aplay_err = _espeak_aplay_pipe(text, pipe_aplay_cmd)
            if ec == 0 and ac == 0:
                return
            if espeak_err or aplay_err:
                print(
                    f"TTS warning: pipe espeak/aplay espeak={ec} aplay={ac} "
                    f"{espeak_err.decode(errors='ignore').strip()} "
                    f"{aplay_err.decode(errors='ignore').strip()}"
                )
        except Exception as e:
            print(f"TTS error (pipe method): {e}")

        # 5) espeak-ng direct (portaudio / default)
        print("Trying direct espeak-ng...")
        result = subprocess.run(
            ['espeak-ng', '-a', '200', '-g', '5', '-p', '50', '-s', '130', text],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return
        print(f"TTS warning: direct espeak-ng failed ({result.returncode})")
        if result.stderr:
            print(f"  {result.stderr.decode(errors='ignore').strip()}")
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# --- Load known faces ---
known_encodings = []
known_names = []
known_faces_dir = "known_faces"

print("Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"Loaded: {name}")
        else:
            print(f"No face found in {filename}, skipping")

print(f"Loaded {len(known_names)} known faces")

# --- Start camera ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()
print("Camera started")

print(
    "\nAudio routing: HAT/speaker uses ALSA first (not Pulse). "
    "Desktop Bluetooth: export PIDOG_USE_PAPLAY_FIRST=1"
)
_print_playback_devices()
print()

# --- Startup audio test ---
speak("PiDog face recognition started")

last_spoken = {}
speak_cooldown = 5

try:
    while True:
        frame = picam2.capture_array()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_names[best_match]

            current_time = time.time()
            last_time = last_spoken.get(name, 0)

            if current_time - last_time > speak_cooldown:
                if name == "Unknown":
                    speak("I see an unknown person")
                else:
                    speak(f"Hello {name}")
                last_spoken[name] = current_time

            print(f"Recognised: {name}")

except KeyboardInterrupt:
    print("Stopped")
finally:
    picam2.stop()