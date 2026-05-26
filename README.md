# 🐶 J.A.R.V.I.S — Just A Rather Very Intelligent Sniffer

**PiDog AI Companion | La Trobe University | Team PiDog 1 | 2026**

An AI-powered robotic dog built on the [SunFounder PiDog](https://github.com/sunfounder/pidog) platform with a Raspberry Pi 5. J.A.R.V.I.S combines voice recognition, computer vision, natural language processing, and physical actuation to create an intelligent, interactive companion.

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Hardware Setup Guide](#hardware-setup-guide)
- [Software Installation](#software-installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Running JARVIS](#running-jarvis)
- [Known Issues & Sprint Notes](#known-issues--sprint-notes)
- [Base Repository](#base-repository)

---

## Overview

J.A.R.V.I.S is not a pre-programmed robot — it listens, thinks, responds, and sees. It features four core AI systems running concurrently:

**1. AI Voice Chatbot** — Say "Hey JARVIS" to activate. JARVIS listens to your question, sends it to the Claude API (Anthropic), and speaks a response aloud while performing physical actions like wagging its tail, barking, nodding, or shaking hands — all driven by the LLM's output. RGB LED mood states reflect the emotional tone of each response.

**2. Computer Vision** — The onboard Pi Camera continuously captures frames shared with the chatbot. When you ask vision-related questions ("what can you see?", "solve this", "look at that"), JARVIS attaches the current camera snapshot to the Claude API request for multimodal responses.

**3. Hand Sign Recognition** — JARVIS recognises common hand signs through its camera using multimodal Claude API vision and responds to them proactively. A separate YOLOv8-based gesture pipeline (`sl_pipeline/`) exists for future Auslan sign language support pending MediaPipe Python 3.13 compatibility.

**4. Face Detection** — JARVIS recognises registered team members and personalises its greeting when a known face is detected. Face encodings are stored locally.

---

## Team

| Name | Student ID | Role |
| ---- | ---------- | ---- |
| Ishaan Hans | 21752626 | Project Lead & AI/NLP Engineer |
| Suhansa Benthotage | 21966347 | Integration & Demo Engineer |
| Luke Haykal | 21606899 | Hardware & Locomotion Engineer |
| Parth Jadav | 21372668 | Computer Vision Engineer |
| Dhil Balasingam | 21914856 | Documentation & QA Lead |

---

## Hardware Setup Guide

### What's in the kit

- SunFounder PiDog chassis, servo motors, and linkage arms
- Raspberry Pi 5
- Robot HAT+ 5 with speaker output
- Pi Camera Module (OV5647)
- Ultrasonic sensor module
- Dual touch sensor
- Sound direction sensor
- RGB LED strip
- Microphone

### Step-by-step assembly

> ⚠️ **Read the known issues section below before starting assembly.** Two critical steps (SSH configuration and servo zeroing) must be done before physical assembly.

**1. Flash the SD card (do this first)**

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash Raspberry Pi OS (64-bit).

Before writing, open **Advanced Options** (`Ctrl+Shift+X`) and configure:
- ✅ Enable SSH
- ✅ Set username and password
- ✅ Configure WiFi SSID and password
- ✅ Set hostname (e.g. `jarvis.local`)

**2. Boot the Pi and verify SSH access**

```bash
ssh jarvis@jarvis.local
```

**3. Zero all servos before physical assembly**

> ⚠️ **Critical — do not attach servo horns or leg linkages until this step is complete.**

On the **Raspberry Pi 5 with Robot HAT+ 5**, press and hold the physical **ZERO button** on the Robot HAT board until all servo shafts rotate to their neutral 0° position, then attach servo horns and linkage arms.

**4. Assemble the chassis**

Follow the [SunFounder PiDog assembly guide](https://docs.sunfounder.com/projects/pidog/en/latest/).

**5. Verify locomotion**

```bash
cd ~/pidog/examples
sudo python3 4_response.py
```

---

## Software Installation

```bash
# Clone the repository
git clone https://github.com/IshaanHans/JARVIS-pidog.git ~/pidog
cd ~/pidog

# Install SunFounder robot-hat (2.5.x branch — required for Pi 5)
git clone -b 2.5.x --depth=1 https://github.com/sunfounder/robot-hat.git ~/robot-hat
cd ~/robot-hat && sudo python3 install.py

# Install vilib
git clone --depth=1 https://github.com/sunfounder/vilib.git ~/vilib
cd ~/vilib && sudo python3 install.py

# Install pidog SDK
cd ~/pidog && sudo pip3 install . --break-system-packages

# Install i2s audio
sudo bash i2samp.sh

# Install AI dependencies
pip install ultralytics opencv-python numpy scikit-learn anthropic face-recognition --break-system-packages
sudo apt-get install -y portaudio19-dev espeak-ng mpg123
pip install pyaudio --break-system-packages

# Set Claude API key (get free key at console.anthropic.com)
echo 'export ANTHROPIC_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

---

## Project Structure

```
pidog/
├── examples/
│   ├── jarvis_chatbot.py       # Main AI voice chatbot
│   ├── voice_active_dog.py     # SunFounder VoiceActiveDog base class
│   ├── squid_game.py           # Squid Game Red Light Green Light mode
│   └── 4_response.py           # Sensor verification script
├── sl_pipeline/                # Gesture recognition pipeline (standalone)
│   ├── main.py                 # Pipeline entry point
│   ├── detector.py             # YOLOv8 pose keypoint extraction
│   ├── classifier.py           # RandomForest classifier with smoothing
│   └── tts.py                  # Non-blocking Piper TTS
├── face_detect/                # Face detection and recognition
│   ├── face_detect.py          # Face recognition + vision snapshot writer
│   └── known_faces/            # Add photos here for face recognition
├── sounds/                     # Audio files
│   └── squid_game/             # Squid Game doll audio files
├── model/                      # Gesture classifier
│   ├── signs.py                # Gesture vocabulary definition
│   └── train.py                # RandomForest training script
├── data_collection/            # Training data collection
│   └── collect.py              # YOLOv8 keypoint capture script
├── utils/
│   └── landmark_utils.py       # Keypoint normalisation
├── start_jarvis.py             # Full system startup script
└── README.md
```

---

## Features

### 🎤 AI Voice Chatbot

- Wake word detection — say "Hey JARVIS", "Hey Buddy" or similar phrases
- Speech-to-text via Vosk (offline)
- Natural language responses powered by **Claude API** (claude-haiku-4-5-20251001)
- Piper TTS for voice output
- Physical action execution — LLM responses trigger real movements (wag tail, bark, sit, stand, handshake, high five, nod, stretch, and more)
- **Mood-based RGB LED states** — response tone changes LED colour (green = happy, blue = sad, red = alert, yellow = excited, cyan = curious, pink = neutral)
- **Multimodal vision** — questions involving visual context ("what can you see?", "solve this", "look at that") attach a live camera snapshot to the Claude API request

### 🤟 Hand Sign Recognition (via Vision + Claude API)

- JARVIS recognises common hand signs through its camera using multimodal Claude API vision
- When a hand sign is detected in the camera frame, JARVIS responds to it proactively without being asked
- Supported signs: Peace ✌️, Thumbs Up 👍, Wave/Open Hand 👋, Pointing ☝️, Fist ✊, Heart 🤞, OK Sign 👌
- No separate pipeline needed — Claude interprets the sign from the camera snapshot directly
- Works alongside the voice chatbot in the main `start_jarvis.py` system

### 🤖 Gesture Recognition Pipeline (standalone — `sl_pipeline/`)

A separate YOLOv8-based gesture recognition pipeline exists for research and future development:
- YOLOv8n-pose body keypoint extraction (14.7 FPS on Pi 5 CPU)
- RandomForest classifier trained on HELP and NO body gestures
- Not integrated into the main system due to reliability limitations at body keypoint level

> **Platform Constraint:** MediaPipe Hands (21 finger keypoints) was the planned solution for full Auslan sign language recognition but is incompatible with Python 3.13 (Raspberry Pi OS Trixie default). Picamera2 requires Python 3.13, creating an irreconcilable conflict. Full Auslan sign language support is planned once MediaPipe releases Python 3.13 support: [github.com/google-ai-edge/mediapipe/issues/5708](https://github.com/google-ai-edge/mediapipe/issues/5708)

### 👁 Face Detection & Vision

- Recognises registered faces and greets them by name on wake
- Continuously writes camera snapshots to `/tmp/jarvis_vision.jpg` for chatbot vision
- Add photos to `face_detect/known_faces/name.jpg` to register new faces

### 🦑 Squid Game Mode

- Red Light / Green Light game using MOG2 background subtraction motion detection
- RGB strip turns green (JARVIS looks away) / red (JARVIS watches for movement)
- Authentic squid game doll audio for green/red light announcements
- JARVIS barks and eliminates players caught moving during red light
- 5 rounds with randomised durations

### 🐾 Personality & Sensors

- RGB LED mood states reflecting emotional tone of responses
- Reacts to touch — nods and wags tail when petted front-to-rear
- Reacts to proximity — backs away if something gets too close
- Personality: witty, confident, slightly sarcastic — like JARVIS from Iron Man

---

## Tech Stack

| Component | Technology |
| --------- | ---------- |
| Platform | Raspberry Pi 5 + SunFounder PiDog SDK |
| LLM | Claude API — `claude-haiku-4-5-20251001` (Anthropic) |
| Wake word / STT | Vosk (offline) |
| Hand sign recognition | Claude API multimodal vision |
| Gesture pipeline | YOLOv8n-pose + picamera2 + RandomForest |
| Text-to-speech | Piper TTS (`en_US-ryan-medium`) |
| Face recognition | face_recognition library (dlib backend) |
| Vision | Shared camera snapshot via face_detect → Claude API |
| Motion detection | OpenCV MOG2 background subtraction |
| Language | Python 3.13 |

---

## Running JARVIS

### Full system (recommended)

```bash
cd ~/pidog
python3 start_jarvis.py
```

Launches face detection and voice chatbot together. Say **"Hey JARVIS"** to activate.

### Voice chatbot only

```bash
cd ~/pidog/examples
sudo -E python3 jarvis_chatbot.py
```

### Gesture recognition pipeline (standalone)

```bash
cd ~/pidog
sudo -E python3 sl_pipeline/main.py --no-display
```

### Squid Game mode

```bash
cd ~/pidog/examples
sudo python3 squid_game.py
```

### Collect gesture training data

```bash
cd ~/pidog
python3 data_collection/collect.py --sign HELP --samples 300
python3 data_collection/collect.py --sign NO --samples 300
python3 model/train.py
```

### Add a face for recognition

```bash
# Add photo to face_detect/known_faces/firstname.jpg
# Then re-encode:
cd ~/pidog/face_detect
python3 face_detect.py --encode
```

---

## Known Issues & Sprint Notes

### Issue #1 — SSH Disabled by Default
**Sprint:** 2 | **Severity:** Blocker

SSH is disabled by default on Raspberry Pi OS Trixie. Must be enabled via Raspberry Pi Imager Advanced Options before first boot.

### Issue #2 — Servo Misalignment
**Sprint:** 2 | **Severity:** High

Attaching servo horns before zeroing causes misaligned legs. Press the ZERO button on Robot HAT+ 5 before any physical assembly.

### Issue #3 — GPIO Busy Errors
**Sprint:** 2 | **Severity:** Blocker

Background systemd services claim GPIO pins at boot. Resolution: `sudo systemctl disable pidog.service`. Never run JARVIS as a systemd service.

### Issue #4 — MediaPipe Incompatible with Python 3.13
**Sprint:** 2-5 | **Severity:** High

MediaPipe Hands requires Python 3.11. Picamera2 requires Python 3.13. These cannot run simultaneously on Pi 5. Workaround: Claude API multimodal vision for hand sign recognition in the main chatbot. Standalone YOLOv8 gesture pipeline available in `sl_pipeline/` for body-level gesture research. Tracking: [mediapipe/issues/5708](https://github.com/google-ai-edge/mediapipe/issues/5708)

### Issue #5 — Ollama LLM Too Slow
**Sprint:** 2 | **Severity:** High

Local LLM inference (llama3.2:3b) took 13+ seconds per response on Pi 5 CPU. Resolution: Claude API (1-2 second response time).

### Issue #6 — SD Card Storage Running Out
**Sprint:** 5 | **Severity:** High

Repeated `sudo pip install` auto-pulled NVIDIA/CUDA packages (~3.5GB) despite Pi having no GPU. Resolution: Removed nvidia/triton/cuda packages, duplicate .git folder (~9GB), and editor server files (~2.4GB). Recovered ~21GB.

---

## Base Repository

This project builds on the official SunFounder PiDog repository: https://github.com/sunfounder/pidog

Full hardware documentation: https://docs.sunfounder.com/projects/pidog/en/latest/