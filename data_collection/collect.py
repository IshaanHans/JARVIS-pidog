import argparse, os, sys, time
import cv2
import numpy as np
sys.path.append(os.path.expanduser('~/pidog'))
from model.signs import SIGNS, SIGN_TO_INDEX
from utils.landmark_utils import normalise_landmarks

OUTPUT_PATH = os.path.expanduser('~/pidog/data_collection/training_data.npz')
SAMPLES_PER_SIGN = 150
COUNTDOWN_SECONDS = 5

def collect_sign(model, picam, sign_label, num_samples):
    sign_idx = SIGN_TO_INDEX[sign_label]
    samples = []
    print(f'\n--- Sign: {sign_label} ---')
    print(f'Stand 30-50cm from camera, hold gesture steady.')
    print(f'Recording in {COUNTDOWN_SECONDS} seconds...')
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f'  {i}...', flush=True)
        time.sleep(1)
    print('RECORDING — hold gesture steady!', flush=True)
    no_detect = 0
    while len(samples) < num_samples:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model(frame, imgsz=320, verbose=False)
        vector = None
        if results[0].keypoints is not None:
            kps = results[0].keypoints.data
            if len(kps) > 0:
                kp = kps[0].cpu().numpy()
                h, w = frame.shape[:2]
                kp[:, 0] /= w
                kp[:, 1] /= h
                vector = normalise_landmarks(kp.flatten().astype(np.float32))
        if vector is not None:
            samples.append((vector, sign_idx))
            no_detect = 0
            if len(samples) % 30 == 0:
                print(f'  {len(samples)}/{num_samples} collected...', flush=True)
        else:
            no_detect += 1
            if no_detect % 30 == 0:
                print('  WARNING: No person detected — move into frame!', flush=True)
    print(f'Done — {len(samples)} samples for {sign_label}')
    return samples

def save_data(all_samples):
    X_new = np.array([s[0] for s in all_samples], dtype=np.float32)
    y_new = np.array([s[1] for s in all_samples], dtype=np.int32)
    if os.path.exists(OUTPUT_PATH):
        existing = np.load(OUTPUT_PATH)
        X = np.concatenate([existing['X'], X_new])
        y = np.concatenate([existing['y'], y_new])
        print(f'Merged. Total: {len(X)} samples')
    else:
        X, y = X_new, y_new
        print(f'Saved {len(X)} samples')
    np.savez(OUTPUT_PATH, X=X, y=y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sign', type=str, default=None)
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_SIGN)
    args = parser.parse_args()
    from ultralytics import YOLO
    from picamera2 import Picamera2
    model = YOLO('yolov8n-pose.pt')
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"})
    picam.configure(config)
    picam.start()
    time.sleep(1)
    print('Camera ready.')
    signs_to_collect = [args.sign.upper()] if args.sign else SIGNS
    all_samples = []
    for sign in signs_to_collect:
        if sign not in SIGN_TO_INDEX:
            print(f'WARNING: {sign} not in vocabulary')
            continue
        samples = collect_sign(model, picam, sign, args.samples)
        all_samples.extend(samples)
        if len(signs_to_collect) > 1:
            print('Next sign in 3 seconds...')
            time.sleep(3)
    picam.stop()
    picam.close()
    if all_samples:
        save_data(all_samples)

if __name__ == '__main__':
    main()
