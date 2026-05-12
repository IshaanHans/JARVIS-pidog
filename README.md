# 🐶 J.A.R.V.I.S — Just A Rather Very Intelligent Sniffer

**PiDog AI Companion | La Trobe University | Team PiDog 1 | 2026**

An AI-powered robotic dog built on the [SunFounder PiDog](https://github.com/sunfounder/pidog) platform with a Raspberry Pi 5. J.A.R.V.I.S combines voice recognition, computer vision, natural language processing, and physical actuation to create an intelligent, interactive companion — with a headline feature of **real-time Auslan sign language translation**.

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
- [Demo](#demo)
- [Base Repository](#base-repository)

---

## Overview

J.A.R.V.I.S is not a pre-programmed robot — it listens, thinks, and responds. It features two core AI systems:

**1. AI Voice Chatbot** — Say "Hey JARVIS" to activate. JARVIS listens to your question, sends it to the Claude API (Anthropic), and speaks a response aloud while performing physical actions like wagging its tail, barking, nodding, or shaking hands — all driven by the LLM's output.

**2. Sign Language Translation** — The onboard Pi Camera detects static hand signs using YOLOv8 pose estimation, classifies them with a trained RandomForest model, and speaks the meaning aloud via TTS — acting as a live accessibility bridge between deaf and hearing people. The system uses static held signs (not dynamic gestures) for reliable real-time classification, covering a vocabulary of 8 common signs: HELLO, HELP, YES, NO, STOP, OKAY, SORRY, and I LOVE YOU.

---

## Team

| Name | Student ID | Role |
|---|---|---|
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

> See [Issue #1](#issue-1--ssh-disabled-by-default-on-raspberry-pi-os) — SSH is disabled by default and must be enabled before first boot.

**2. Boot the Pi and verify SSH access**

```bash
ssh jarvis@jarvis.local
# or use IP address:
ssh jarvis@<PI_IP_ADDRESS>
```

**3. Zero all servos before physical assembly**

> ⚠️ **Critical — do not attach servo horns or leg linkages until this step is complete.**
> See [Issue #2](#issue-2--servo-misalignment-due-to-skipped-zero-calibration).

On the **Raspberry Pi 5 with Robot HAT+ 5**, use the physical **ZERO button** on the Robot HAT board to zero all servos — press and hold it until all servo shafts rotate to their neutral 0° position, then attach the servo horns and linkage arms.

> On Raspberry Pi 4 (original Robot HAT), zeroing was done via software script instead. The Pi 5 Robot HAT+ 5 has a dedicated ZERO button that handles this mechanically without needing to run any code.

**4. Assemble the chassis**

Follow the [SunFounder PiDog assembly guide](https://docs.sunfounder.com/projects/pidog/en/latest/). Key tips:
- Torque servo horn screws firmly — loose horns cause leg wobble
- Route cables away from joint pivot points
- Plug the ultrasonic module into port `D0` on the Robot HAT+ 5
- Plug the dual touch sensor into ports `D2` and `D3`

**5. Verify locomotion**

```bash
cd ~/pidog/examples
sudo python3 4_response.py
```

---

## Software Installation

```bash
# Clone the repository
git clone https://github.com/IshaanHans/JARVIS-pidog.git
cd pidog

# Install SunFounder robot-hat (2.5.x branch — required for Pi 5)
cd ~/
git clone -b 2.5.x --depth=1 https://github.com/sunfounder/robot-hat.git
cd robot-hat
sudo python3 install.py

# Install vilib
cd ~/
git clone --depth=1 https://github.com/sunfounder/vilib.git
cd vilib
sudo python3 install.py

# Install pidog SDK
cd ~/pidog
sudo pip3 install . --break-system-packages

# Install i2s audio
sudo bash i2samp.sh

# Install AI dependencies
pip install faster-whisper ultralytics opencv-python numpy scikit-learn anthropic --break-system-packages
sudo apt-get install -y portaudio19-dev espeak-ng
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
│   ├── jarvis_chatbot.py       # Main AI voice chatbot (run this)
│   ├── voice_active_dog.py     # SunFounder VoiceActiveDog class
│   ├── 4_response.py           # Sensor test script
│   └── ...                     # Other SunFounder examples
├── sl_pipeline/                # Sign language translation pipeline
│   ├── main.py                 # Sign language pipeline entry point
│   ├── detector.py             # YOLOv8 pose keypoint extraction
│   ├── classifier.py           # Sign classifier with smoothing buffer
│   └── tts.py                  # Non-blocking text-to-speech
├── face_detection/             # Face detection and recognition
│   └── face_detect.py
├── sounds/                     # Audio files for robot expressions
├── model/                      # Trained sign language model (generated)
├── data_collection/            # Training data collection scripts
├── start_jarvis.sh             # Auto-start script for boot
└── README.md
```

---

## Features

### 🎤 AI Voice Chatbot
- Wake word detection — say "Hey JARVIS", "Hey Buddy" or similar phrases
- Speech-to-text via Vosk (offline, no API needed)
- Natural language responses powered by **Claude API** (Anthropic)
- Piper TTS for high-quality voice output
- Physical action execution — Claude's responses trigger real movements:
  - Wag tail, bark, sit, stand, handshake, high five, nod, shake head, stretch, and more

### 🤟 Sign Language Translation
- Real-time static sign detection via Pi Camera
- YOLOv8n-pose for body/hand keypoint extraction (14.7 FPS on Pi 5)
- RandomForest classifier trained on 8 static hand signs
- Speaks detected signs aloud via TTS
- Vocabulary: HELLO, HELP, YES, NO, STOP, OKAY, SORRY, I LOVE YOU
- Uses static held signs (not dynamic gestures) for reliable YOLOv8 classification

### 👁 Computer Vision
- Face detection using OpenCV DNN
- Person detection via YOLOv8n

### 🐾 Personality System
- RGB LED mood states (pink breath = idle, red bark = alert, purple = listening)
- Reacts to proximity — backs away and barks if something gets too close
- Reacts to touch — nods and wags tail when petted
- Personality prompt: witty, confident, slightly sarcastic — like JARVIS from Iron Man

---

## Tech Stack

| Component | Technology |
|---|---|
| Platform | Raspberry Pi 5 + SunFounder PiDog SDK |
| LLM | Claude API — `claude-haiku-4-5-20251001` (Anthropic) |
| Wake word / STT | faster-whisper (tiny model, offline) + Vosk |
| Sign detection | YOLOv8n-pose + picamera2 (static signs) |
| Sign classification | scikit-learn RandomForest |
| Text-to-speech | Piper TTS (`en_US-ryan-low`) |
| Object detection | YOLOv8n + OpenCV |
| Face detection | OpenCV DNN (res10 SSD model) |
| Language | Python 3.13 |

---

## Running JARVIS

### AI Voice Chatbot (main demo)
```bash
cd ~/pidog/examples
sudo python3 jarvis_chatbot.py
```

Say **"Hey JARVIS"** to activate, then ask anything. JARVIS will respond verbally and physically.

### Sign Language Pipeline
```bash
cd ~/pidog
python3 sl_pipeline/main.py --always-on
```

### Collect sign language training data
```bash
python3 data_collection/collect.py --sign HELLO --samples 80
```

### Train sign classifier
```bash
python3 model/train.py
```

### Auto-start on boot
```bash
# Add to desktop autostart
cp start_jarvis.sh ~/
chmod +x ~/start_jarvis.sh
```

---

## Known Issues & Sprint Notes

### Issue #1 — SSH Disabled by Default on Raspberry Pi OS

**Sprint:** Sprint 2 — Hardware Assembly
**Severity:** Blocker

**What happened:** When the SD card was flashed with a fresh Raspberry Pi OS image, SSH access was unavailable — SSH is disabled by default as a security measure. The team had no spare monitor or keyboard during the assembly session, making the Pi completely inaccessible.

**Resolution:** Re-flashed using Raspberry Pi Imager with SSH explicitly enabled via Advanced Options (`Ctrl+Shift+X`). WiFi credentials and hostname were also pre-configured at this stage.

**Prevention:** Pre-configuring the SD card via Raspberry Pi Imager Advanced Options is now the mandatory first step. Added to team assembly checklist.

---

### Issue #2 — Servo Misalignment Due to Skipped Zero Calibration

**Sprint:** Sprint 2 — Hardware Assembly
**Severity:** High

**What happened:** Servo motors were attached to leg brackets without first running the software zero calibration. The PiDog geometry assumes all joints start at 0°. Because this was skipped, several joints were assembled at incorrect offsets — the dog could not stand level on first boot.

**Resolution:** Disassembled affected joints, ran the servo zero routine via SDK, reattached horns at correct 0° positions. Dog stood level after reassembly.

**Prevention:** Servo zeroing is documented as Step 3 of the hardware setup guide above and is a mandatory checklist item before any physical assembly.

---

### Issue #3 — GPIO Pins Claimed at Boot by Background Service

**Sprint:** Sprint 2 — Hardware Assembly
**Severity:** Blocker

**What happened:** The ultrasonic sensor, dual touch sensor, and sound direction sensor all failed to initialise with `GPIO busy` errors after a team member enabled a background service (`pidog.service`) that auto-started on boot and claimed GPIO pins 4, 17, 22, and 27 via `lgpio` before any pidog script could access them.

**Resolution:** Stopped and permanently deleted the service using `sudo systemctl stop pidog.service` and `sudo systemctl disable pidog.service`. Removed the service file from `/etc/systemd/system/`.

**Prevention:** Any background service using GPIO must be integrated into the main pidog process — never run as a standalone systemd service. Wake word and voice pipeline are now launched manually or via desktop autostart only.

---

### Issue #4 — MediaPipe Incompatible with Python 3.13



**Sprint:** Sprint 2 — AI Pipeline
**Severity:** High

**What happened:** MediaPipe does not support Python 3.13. The Pi 5 running Raspberry Pi OS Trixie ships with Python 3.13 only, making MediaPipe unavailable for hand landmark detection.

**Resolution:** Switched to YOLOv8n-pose (`ultralytics`) for body/hand keypoint extraction, combined with `picamera2` for camera capture. Achieved 14.7 FPS average on Pi 5 — acceptable for sign language detection. The sign vocabulary was also changed to static held signs rather than dynamic gestures, as YOLOv8 pose classifies single-frame keypoint positions rather than motion sequences.

**Prevention:** Documented in tech stack. If MediaPipe releases Python 3.13 support, migration guide will be added.

---

### Issue #5 — Ollama LLM Too Slow for Real-Time Conversation

**Sprint:** Sprint 2 — AI Pipeline
**Severity:** High

**What happened:** Running `llama3.2:3b` locally via Ollama took 13+ seconds per response on Pi 5 CPU. Even the smallest model `qwen2.5:0.5b` took 10+ seconds — unacceptable for a conversational demo.

**Resolution:** Replaced Ollama with Claude API (`claude-haiku-4-5-20251001`). Response time dropped to 1-2 seconds. A custom `ClaudeLLM` wrapper class was written to make Claude API compatible with the SunFounder `VoiceAssistant` interface.

**Prevention:** For Pi-based deployments, always prefer cloud LLM APIs over local models unless hardware acceleration (GPU) is available.

---

## Demo

🎥 Demo video — coming Sprint 4

---

## Base Repository

This project builds on the official SunFounder PiDog repository:
https://github.com/sunfounder/pidog

Full hardware documentation:
https://docs.sunfounder.com/projects/pidog/en/latest/
