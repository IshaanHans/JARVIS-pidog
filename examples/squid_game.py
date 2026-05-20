#!/usr/bin/env python3
from pidog import Pidog
from pidog.preset_actions import bark_action
from time import sleep
import cv2
import numpy as np
import random
import subprocess
import sys, os
sys.path.append(os.path.expanduser('~/pidog'))

my_dog = Pidog()
sleep(0.5)

from picamera2 import Picamera2
picam = Picamera2()
picam.configure(picam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}))
picam.start()
sleep(1)

MOTION_THRESHOLD = 5000
GREEN_LIGHT_SOUND = '/home/jarvis/pidog/sounds/squid_game/doll-green-light.mp3'
RED_LIGHT_SOUND   = '/home/jarvis/pidog/sounds/squid_game/doll-red-light.mp3'

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40, detectShadows=False)

def capture_frame():
    frame = picam.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def flush_background(frames=20):
    for _ in range(frames):
        frame = capture_frame()
        bg_subtractor.apply(frame)
        sleep(0.05)

def detect_motion(frame):
    fg_mask = bg_subtractor.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    motion_pixels = cv2.countNonZero(fg_mask)
    return motion_pixels, motion_pixels > MOTION_THRESHOLD

def play_sound(filepath):
    subprocess.run(['/usr/bin/mpg123', '-q', filepath])

def say(text):
    model = os.path.expanduser('~/.piper_models/en_US-ryan-medium.onnx')
    cmd = f'echo "{text}" | piper --model {model} --output-raw | aplay -r 22050 -f S16_LE -t raw -'
    subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

def green_light(duration):
    print('[GREEN LIGHT]')
    my_dog.rgb_strip.set_mode(style='breath', color='green', bps=1, brightness=1)
    my_dog.head_move([[80, 0, 0]], immediately=True, speed=80)
    play_sound(GREEN_LIGHT_SOUND)
    elapsed = 0
    while elapsed < duration:
        frame = capture_frame()
        bg_subtractor.apply(frame)
        sleep(0.1)
        elapsed += 0.1

def red_light():
    print('[RED LIGHT]')
    my_dog.rgb_strip.set_mode(style='monochromatic', color='red', brightness=1)
    my_dog.head_move([[0, 0, 0]], immediately=True, speed=80)
    play_sound(RED_LIGHT_SOUND)
    sleep(1.0)
    flush_background(20)

def caught():
    print('[CAUGHT] Motion detected during red light!')
    my_dog.rgb_strip.set_mode(style='bark', color='red', bps=4, brightness=1)
    say('You moved. You are eliminated.')
    bark_action(my_dog)
    sleep(0.2)
    bark_action(my_dog)

def game_over_win():
    print('[YOU WIN]')
    my_dog.rgb_strip.set_mode(style='boom', color='yellow', bps=2, brightness=1)
    say('You have survived all rounds. Congratulations.')
    my_dog.do_action('wag_tail', step_count=10, speed=80)
    my_dog.wait_all_done()

def squid_game():
    print('=== JARVIS SQUID GAME MODE ===')
    print('Stand in front of the camera.')
    print('Move during GREEN LIGHT, freeze during RED LIGHT!')
    print('Calibrating camera — stand still for 3 seconds...')
    my_dog.do_action('stand', speed=60)
    my_dog.rgb_strip.set_mode(style='breath', color='cyan', bps=1, brightness=0.8)

    for _ in range(30):
        frame = capture_frame()
        bg_subtractor.apply(frame)
        sleep(0.1)

    print('Starting!')
    sleep(1)

    rounds = 5
    for round_num in range(1, rounds + 1):
        print(f'\n--- Round {round_num}/{rounds} ---')

        green_duration = random.uniform(2.0, 4.0)
        green_light(green_duration)

        red_light()
        red_duration = random.uniform(3.0, 6.0)

        caught_moving = False
        check_time = 0
        while check_time < red_duration:
            frame = capture_frame()
            motion_pixels, is_moving = detect_motion(frame)
            print(f'  Motion pixels: {motion_pixels}', end='\r')

            if is_moving:
                caught()
                caught_moving = True
                break

            sleep(0.1)
            check_time += 0.1

        if caught_moving:
            print('\n[GAME OVER] You were caught moving!')
            my_dog.rgb_strip.set_mode(style='monochromatic', color='red', brightness=1)
            say('Game over. You have been eliminated.')
            sleep(2)
            break

        if round_num == rounds:
            game_over_win()

    print('\n[JARVIS] Game over. Thanks for playing!')
    my_dog.rgb_strip.set_mode(style='breath', color='pink', bps=1, brightness=0.5)

if __name__ == '__main__':
    try:
        squid_game()
    except KeyboardInterrupt:
        print('\n[JARVIS] Game interrupted.')
    except Exception as e:
        print(f'\033[31mERROR: {e}\033[m')
    finally:
        picam.stop()
        picam.close()
        my_dog.close()
