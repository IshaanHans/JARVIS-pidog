#!/usr/bin/env python3
"""
JARVIS Startup Script
Boots the full JARVIS system: face detection and voice chatbot.
"""

import subprocess
import threading
import sys
import os
import time
import signal

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR  = os.path.join(BASE_DIR, "examples")
FACE_DETECT_DIR = os.path.join(BASE_DIR, "face_detect")

JARVIS_CHATBOT = os.path.join(EXAMPLES_DIR, "jarvis_chatbot.py")
FACE_DETECT    = os.path.join(FACE_DETECT_DIR, "face_detect.py")

# ── process registry ──────────────────────────────────────────────────────────
processes = []

def start_process(name, script_path, cwd=None):
    if not os.path.exists(script_path):
        print(f"[JARVIS] WARNING: {name} not found at {script_path}, skipping.")
        return None
    print(f"[JARVIS] Starting {name}...")
    proc = subprocess.Popen(
        [sys.executable, script_path],
        cwd=cwd or os.path.dirname(script_path),
    )
    processes.append((name, proc))
    print(f"[JARVIS] {name} started (PID {proc.pid})")
    return proc

def shutdown(signum=None, frame=None):
    print("\n[JARVIS] Shutting down all systems...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"[JARVIS] Stopping {name} (PID {proc.pid})...")
            proc.terminate()
    time.sleep(2)
    for name, proc in processes:
        if proc.poll() is None:
            proc.kill()
    print("[JARVIS] All systems offline. Goodbye.")
    sys.exit(0)

def monitor_process(name, proc):
    proc.wait()
    if proc.returncode != 0:
        print(f"[JARVIS] WARNING: {name} exited with code {proc.returncode}")

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    print("=" * 50)
    print("  J.A.R.V.I.S  —  Booting all systems")
    print("=" * 50)

    # 1. Start face detection silently in background
    face_proc = start_process("FaceDetect", FACE_DETECT, cwd=FACE_DETECT_DIR)
    time.sleep(3)

    # 2. Start JARVIS voice chatbot
    jarvis_proc = start_process("VoiceChatbot", JARVIS_CHATBOT, cwd=EXAMPLES_DIR)

    # ── monitor threads ───────────────────────────────────────────────────────
    for name, proc in processes:
        m = threading.Thread(target=monitor_process, args=(name, proc), daemon=True)
        m.start()

    print("\n[JARVIS] All systems online. Press Ctrl+C to shut down.\n")

    try:
        while True:
            if jarvis_proc and jarvis_proc.poll() is not None:
                print("[JARVIS] Voice chatbot stopped. Shutting down...")
                shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()

    #