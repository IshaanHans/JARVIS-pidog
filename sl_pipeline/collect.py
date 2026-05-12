#!/usr/bin/env python3
# Headless sign language data collection
# Uses MediaPipe Hands + picamera2
# Saves landmarks in same format as hand_gesture_ref repo

import csv
import copy
import itertools
import time
import argparse
import os
import sys

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# Labels
SIGNS = [
    "HELLO",
    "HELP",
    "YES",
    "NO",
    "STOP",
    "OKAY",
    "SORRY",
    "I_LOVE_YOU",
]

SIGN_TO_INDEX = {sign: i for i, sign in enumerate(SIGNS)}
OUTPUT_CSV = os.path.expanduser('~/hand_gesture_ref/model/keypoint_classifier/keypoint.csv')
SAMPLES_PER_SIGN = 100
COUNTDOWN = 5

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalise
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def collect_sign(picam, sign_label, num_samples):
    sign_idx = SIGN_TO_INDEX[sign_label]
    samples = []

    print(f'\n--- Sign: {sign_label} (index {sign_idx}) ---')
    print(f'Hold the {sign_label} sign clearly in front of the camera.')
    print(f'Recording starts in {COUNTDOWN} seconds...')

    for i in range(COUNTDOWN, 0, -1):
        print(f'  {i}...', flush=True)
        time.sleep(1)

    print(f'RECORDING NOW — hold the sign steady!', flush=True)

    no_detect = 0
    while len(samples) < num_samples:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                samples.append([sign_idx] + pre_processed)
                no_detect = 0

            if len(samples) % 10 == 0:
                print(f'  {len(samples)}/{num_samples} samples collected...', flush=True)
        else:
            no_detect += 1
            if no_detect % 30 == 0:
                print(f'  WARNING: No hand detected — move hand into frame!', flush=True)

    print(f'Done — {len(samples)} samples collected for {sign_label}')
    return samples

def save_samples(samples):
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)
    print(f'Saved {len(samples)} rows to {OUTPUT_CSV}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sign', type=str, default=None,
                        help='Sign to collect (default: all)')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_SIGN)
    args = parser.parse_args()

    signs_to_collect = [args.sign.upper()] if args.sign else SIGNS

    # Validate
    for sign in signs_to_collect:
        if sign not in SIGN_TO_INDEX:
            print(f'ERROR: {sign} not in vocabulary: {SIGNS}')
            return

    # Start camera
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"})
    picam.configure(config)
    picam.start()
    time.sleep(1)
    print('Camera ready.')

    all_samples = []
    for sign in signs_to_collect:
        samples = collect_sign(picam, sign, args.samples)
        all_samples.extend(samples)
        if len(signs_to_collect) > 1:
            print('Next sign in 3 seconds...')
            time.sleep(3)

    picam.stop()
    picam.close()

    if all_samples:
        save_samples(all_samples)
    else:
        print('No data collected.')

if __name__ == '__main__':
    main()
