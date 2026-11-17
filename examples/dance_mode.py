#!/usr/bin/env python3
"""
JARVIS Dance Mode — beat-synced choreography.

    sudo python3 dance_mode.py --song ~/music/track.mp3 --bpm 120
    sudo python3 dance_mode.py --bpm 100 --duration 20        # no music, dance in silence

Every beat, JARVIS moves its head and tail; every bar (4 beats) it changes
leg pose and LED colour. Movement buffers are cleared on every beat so the
dog stays locked to the music instead of drifting behind it.

Tip: find a song's BPM by searching "<song name> bpm". If the moves land
slightly late or early, nudge them with --offset (seconds, can be negative).

Stop start_jarvis.py before running: this script needs the servos and GPIO.
"""
import argparse
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass

BEATS_PER_BAR = 4
COLORS = ['magenta', 'cyan', 'yellow', 'green', 'blue', 'pink']

# Leg poses use ActionDict names (pidog/actions_dictionary.py)
LEG_PATTERN = ['stand', 'half_sit', 'stand', 'sit']

# Head positions as [yaw, roll, pitch]; one entry per beat in a bar
HEAD_PATTERNS = {
    'bop':    [[0, 0, -15], [0, 0, 10], [0, 0, -15], [0, 0, 10]],
    'tilt':   [[0, 20, 0], [0, -20, 0], [0, 20, 0], [0, -20, 0]],
    'look':   [[35, 0, 0], [0, 0, 0], [-35, 0, 0], [0, 0, 0]],
    'groove': [[20, 15, -10], [-20, -15, -10], [20, 15, 5], [-20, -15, 5]],
}
HEAD_ORDER = ['bop', 'tilt', 'look', 'groove']

TAIL_PATTERN = [[-30], [30]]


@dataclass
class BeatEvent:
    t: float              # seconds after the song starts
    beat: int             # beat index from 0
    bar: int
    legs: str | None      # leg pose name, only on the first beat of a bar
    head: list            # [yaw, roll, pitch]
    tail: list            # [angle]
    color: str | None     # LED colour, only on the first beat of a bar


def beat_seconds(bpm):
    if not 40 <= bpm <= 220:
        raise ValueError("bpm must be between 40 and 220")
    return 60.0 / bpm


def servo_speed(bpm):
    """Faster songs need faster servos so each move finishes inside its beat."""
    return max(60, min(98, int(40 + bpm * 0.4)))


def build_schedule(bpm, duration, offset=0.0):
    """Pure function: the full list of beat events for a dance."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    step = beat_seconds(bpm)
    events, beat = [], 0
    while beat * step < duration:
        bar, pos = divmod(beat, BEATS_PER_BAR)
        pattern = HEAD_PATTERNS[HEAD_ORDER[(bar // 2) % len(HEAD_ORDER)]]  # change style every 2 bars
        events.append(BeatEvent(
            t=round(offset + beat * step, 4),
            beat=beat,
            bar=bar,
            legs=LEG_PATTERN[bar % len(LEG_PATTERN)] if pos == 0 else None,
            head=pattern[pos],
            tail=TAIL_PATTERN[beat % 2],
            color=COLORS[bar % len(COLORS)] if pos == 0 else None,
        ))
        beat += 1
    return events


def song_duration(path):
    """Length of an mp3 in seconds using ffprobe if available, else None."""
    if not shutil.which('ffprobe'):
        return None
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip())
    except Exception:
        return None


def perform(dog, events, bpm):
    speed = servo_speed(bpm)
    start = time.monotonic()
    for ev in events:
        # Wait for this beat (sleep most of the gap, then spin briefly for accuracy)
        while True:
            remaining = start + ev.t - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.005) if remaining < 0.02 else remaining - 0.015)

        if ev.color:
            dog.rgb_strip.set_mode('boom', ev.color, bps=bpm / 60, brightness=1)
        if ev.legs:
            frames, _ = dog.actions_dict[ev.legs]
            dog.legs_move(frames, immediately=True, speed=speed)
        dog.head_move([ev.head], immediately=True, speed=speed)
        dog.tail_move([ev.tail], immediately=True, speed=speed)
        print(f"\r  bar {ev.bar + 1:>3}  beat {ev.beat % BEATS_PER_BAR + 1}", end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="JARVIS beat-synced dance mode")
    parser.add_argument('--song', help='path to an mp3 (played with mpg123)')
    parser.add_argument('--bpm', type=float, required=True, help='beats per minute of the song')
    parser.add_argument('--duration', type=float, help='seconds to dance (default: song length, or 30)')
    parser.add_argument('--offset', type=float, default=0.0, help='shift moves later (+) or earlier (-), seconds')
    args = parser.parse_args()

    if args.song and not os.path.isfile(args.song):
        parser.error(f"song not found: {args.song}")
    duration = args.duration or (args.song and song_duration(args.song)) or 30
    events = build_schedule(args.bpm, duration, args.offset)

    from pidog import Pidog   # imported here so the schedule logic is testable without hardware
    dog = Pidog()
    music = None
    try:
        time.sleep(0.5)
        dog.do_action('stand', speed=60)
        dog.wait_all_done()
        print(f"=== JARVIS DANCE MODE === {args.bpm:g} BPM, {duration:.0f}s, {len(events)} beats")

        if args.song:
            music = subprocess.Popen(['/usr/bin/mpg123', '-q', args.song],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        perform(dog, events, args.bpm)

        dog.speak('woohoo', volume=80)
        dog.do_action('wag_tail', step_count=4, speed=90)
        dog.wait_all_done()
    except KeyboardInterrupt:
        print("\n[JARVIS] Dance stopped.")
    finally:
        if music and music.poll() is None:
            music.send_signal(signal.SIGTERM)
        dog.body_stop()
        dog.head_move([[0, 0, 0]], immediately=True, speed=60)
        dog.do_action('sit', speed=50)
        dog.wait_all_done()
        dog.rgb_strip.close()
        dog.close()


if __name__ == '__main__':
    main()
