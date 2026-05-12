#!/usr/bin/env python3
"""
JARVIS Startup Script
Boots the full JARVIS system: sign language pipeline, face detection, and voice chatbot.
Run this script to start everything at once.
"""

import subprocess
import threading
import sys
import os
import time
import signal

# ── paths (relative to this script's location) ──────────────────────────────
# script lives at ~/pidog/start_jarvis.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR    = os.path.join(BASE_DIR, "examples")
SL_PIPELINE_DIR = os.path.join(BASE_DIR, "sl_pipeline")
FACE_DETECT_DIR = os.path.join(BASE_DIR, "face_detect")

JARVIS_CHATBOT   = os.path.join(EXAMPLES_DIR, "jarvis_chatbot.py")
SL_PIPELINE_MAIN = os.path.join(SL_PIPELINE_DIR, "main.py")

# ── process registry ─────────────────────────────────────────────────────────
processes = []

def start_process(name, script_path, cwd=None):
    """Start a Python subprocess and register it."""
    if not os.path.exists(script_path):
        print(f"[JARVIS] WARNING: {name} script not found at {script_path}, skipping.")
        return None

    print(f"[JARVIS] Starting {name}...")
    proc = subprocess.Popen(
        [sys.executable, script_path],
        cwd=cwd or os.path.dirname(script_path),
        # No PIPE — direct terminal output so STT audio loop isn't buffered
    )
    processes.append((name, proc))
    print(f"[JARVIS] {name} started (PID {proc.pid})")
    return proc


def stream_logs(name, proc):
    """Stream subprocess stdout to console with a name prefix."""
    for line in proc.stdout:
        print(f"[{name}] {line}", end="")


def shutdown(signum=None, frame=None):
    """Gracefully shut down all subprocesses."""
    print("\n[JARVIS] Shutting down all systems...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"[JARVIS] Stopping {name} (PID {proc.pid})...")
            proc.terminate()
    # Give them a moment then force kill
    time.sleep(2)
    for name, proc in processes:
        if proc.poll() is None:
            print(f"[JARVIS] Force killing {name}...")
            proc.kill()
    print("[JARVIS] All systems offline. Goodbye.")
    sys.exit(0)


def monitor_process(name, proc):
    """Monitor a process and warn if it dies unexpectedly."""
    proc.wait()
    if proc.returncode != 0:
        print(f"[JARVIS] WARNING: {name} exited with code {proc.returncode}")


# ── register shutdown handler ─────────────────────────────────────────────────
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── main boot sequence ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  J.A.R.V.I.S  —  Booting all systems")
    print("=" * 50)

    # 1. Start sign language pipeline
    #sl_proc = start_process("SignLanguage", SL_PIPELINE_MAIN, cwd=SL_PIPELINE_DIR)

    # Small delay to let SL pipeline initialise before chatbot starts
    #time.sleep(2)

    # 2. Start JARVIS voice chatbot (main process — runs in foreground)
    jarvis_proc = start_process("VoiceChatbot", JARVIS_CHATBOT, cwd=EXAMPLES_DIR)

    # ── start log streaming threads ───────────────────────────────────────────
    threads = []
    for name, proc in processes:
        t = threading.Thread(target=stream_logs, args=(name, proc), daemon=True)
        t.start()
        threads.append(t)

    # ── start monitor threads ─────────────────────────────────────────────────
    for name, proc in processes:
        m = threading.Thread(target=monitor_process, args=(name, proc), daemon=True)
        m.start()

    print("\n[JARVIS] All systems online. Press Ctrl+C to shut down.\n")

    # ── keep main thread alive ────────────────────────────────────────────────
    try:
        while True:
            # Check if the main chatbot process has died
            if jarvis_proc and jarvis_proc.poll() is not None:
                print("[JARVIS] Voice chatbot stopped. Shutting down...")
                shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()