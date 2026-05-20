import argparse, os, sys, time
import cv2
sys.path.append(os.path.expanduser('~/pidog'))
from sl_pipeline.detector import HandDetector
from sl_pipeline.classifier import SignClassifier
from sl_pipeline.tts import Speaker

FONT = cv2.FONT_HERSHEY_SIMPLEX

class JarvisSignPipeline:
    def __init__(self, args):
        self.args = args
        print('Initialising J.A.R.V.I.S Sign Language Pipeline...')
        self.speaker = Speaker(rate=155)
        self.detector = HandDetector()
        try:
            self.classifier = SignClassifier()
        except FileNotFoundError as e:
            print(f'ERROR: {e}')
            sys.exit(1)

    def run(self):
        print('[MODE] Always-on — show your gesture to the camera')
        try:
            while True:
                frame = self.detector.capture_frame()
                frame = cv2.flip(frame, 1)
                vector, annotated = self.detector.process(frame)

                if vector is not None:
                    sign, conf = self.classifier.get_top_prediction(vector)
                    confirmed = self.classifier.predict(vector)

                    if confirmed:
                        print(f'[SIGN] {confirmed}')
                        self.speaker.say(confirmed.replace('_', ' ').lower())

                    if not self.args.no_display and sign:
                        cv2.putText(annotated, f'{sign} {conf:.0%}',
                                    (10, 30), FONT, 0.8, (0, 255, 0), 2)
                else:
                    if not self.args.no_display:
                        cv2.putText(annotated, 'No person detected',
                                    (10, 30), FONT, 0.7, (0, 0, 255), 2)

                if not self.args.no_display:
                    cv2.imshow('JARVIS Sign Language', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print('\nInterrupted.')
        finally:
            cv2.destroyAllWindows()
            self.detector.close()
            self.speaker.wait_until_done()
            self.speaker.close()
            print('Shutdown complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    JarvisSignPipeline(args).run()

if __name__ == '__main__':
    main()
