import threading, queue, subprocess, os

PIPER_MODEL = os.path.expanduser('~/.piper_models/en_US-ryan-low.onnx')

class Speaker:
    def __init__(self, rate=160, volume=1.0):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                cmd = f'echo "{text}" | piper --model {PIPER_MODEL} --output-raw | aplay -r 22050 -f S16_LE -t raw -'
                subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f'[TTS] Error: {e}')
            self._queue.task_done()

    def say(self, text):
        print(f'[TTS] -> {text}')
        self._queue.put(text)

    def wait_until_done(self):
        self._queue.join()

    def close(self):
        self._queue.put(None)
        self._thread.join()
