import os
import base64
import shutil
import cv2
import anthropic
import time
import threading
from pidog.dual_touch import TouchStyle
from voice_active_dog import VoiceActiveDog

# Shared files written by face detection
LAST_SEEN_FILE = "/tmp/jarvis_last_seen.txt"
VISION_SNAPSHOT = "/tmp/jarvis_vision.jpg"

def get_last_seen_name():
    try:
        if os.path.exists(LAST_SEEN_FILE):
            with open(LAST_SEEN_FILE, "r") as f:
                name = f.read().strip()
            return name if name else None
    except Exception:
        pass
    return None

class JarvisVoiceActiveDog(VoiceActiveDog):
    def _battery_monitor(self):
        from robot_hat.adc import ADC
        time.sleep(15)
        adc = ADC("A4")
        alerted = False
        while True:
            try:
                voltage = round(adc.read_voltage() * 3, 2)
                print(f"[JARVIS] Battery: {voltage}V")
                if voltage > 1.0 and voltage < 7.0 and not alerted:
                    print(f"[JARVIS] Low battery alert: {voltage}V")
                    self.dog.rgb_strip.set_mode("breath", "red", 2)
                    import subprocess
                    subprocess.Popen([
                        "bash", "-c",
                        f"echo 'Warning. Battery low at {voltage} volts. Please plug me in soon.' | piper --model /home/jarvis/.piper_models/en_US-ryan-medium.onnx --output_raw | aplay -r 22050 -f S16_LE -t raw -"
                    ])
                    alerted = True
                elif voltage >= 7.2:
                    alerted = False
            except Exception as e:
                print(f"[JARVIS] Battery error: {e}")
            time.sleep(60)

    def on_start(self):
        self._last_activity = time.time()
        super().on_start()
        threading.Thread(target=self._battery_monitor, daemon=True).start()
        self.start_idle_monitor()

    def on_wake(self):
        super().on_wake()
        name = get_last_seen_name()
        if name:
            self.answer_on_wake = f"Hello {name.capitalize()}, how can I help?"
        else:
            self.answer_on_wake = "Yes, how can I help sir?"


    def start_idle_monitor(self):
        import random
        self._last_activity = time.time()
        def idle_loop():
            while True:
                time.sleep(30)
                idle_time = time.time() - self._last_activity
                if idle_time > 60:
                    print("[JARVIS] Idle behaviour triggered.")
                    self.dog.rgb_strip.set_mode('breath', 'pink', 0.5)
                    idle_actions = ['stretch', 'relax neck', 'shake head', 'nod']
                    self.action_flow.add_action(random.choice(idle_actions))
                    self._last_activity = time.time()
        threading.Thread(target=idle_loop, daemon=True).start()

    def on_heard(self, text):
        super().on_heard(text)
        self._last_activity = time.time()

    def get_mood(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["sorry", "apologize", "confused", "error", "trouble", "unfortunately", "sad", "upset", "unhappy", "disappointed"]):
            return "sad", "blue"
        elif any(w in text_lower for w in ["bark", "back", "away", "stop", "danger", "warning"]):
            return "angry", "red"
        elif any(w in text_lower for w in ["excited", "amazing", "brilliant", "fantastic", "love", "great", "awesome", "wonderful"]):
            return "excited", "yellow"
        elif any(w in text_lower for w in ["hello", "hi", "hey", "welcome", "nice to meet", "pleasure"]):
            return "happy", "green"
        elif any(w in text_lower for w in ["thinking", "calculating", "processing", "hmm", "interesting", "curious"]):
            return "curious", "cyan"
        else:
            return "neutral", "pink"

    def before_say(self, text):
        mood, color = self.get_mood(text)
        print(f"[JARVIS] Mood: {mood}")
        self.dog.rgb_strip.set_mode("breath", color, 1)

    def init_camera(self):
        """Face detection owns Picamera2; voice reads shared snapshots."""
        self.picam2 = None
        print("[JARVIS] Vision enabled — using snapshots from face detection.")

    def close_camera(self):
        pass

    def capture_image(self, path):
        if os.path.exists(VISION_SNAPSHOT):
            img = cv2.imread(VISION_SNAPSHOT)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            else:
                shutil.copy(VISION_SNAPSHOT, path)
        else:
            print("[JARVIS] No vision snapshot yet — responding without image.")

class ClaudeLLM:
    """Drop-in replacement for SunFounder's Ollama class using Claude API."""

    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.messages = []
        self.system_prompt = ""

    def set_instructions(self, instructions):
        self.system_prompt = instructions

    def _vision_content(self, text, image_path):
        VISION_KEYWORDS = [
            "see", "look", "what is", "what's", "describe", "show", "camera",
            "front", "behind", "around", "room", "person", "who", "face",
            "colour", "color", "read", "sign", "holding", "wearing", "background", "this", "that", "here", "solve", "calculate", "math", "problem", "equation"
        ]
        text_lower = text.lower()
        vision_requested = any(kw in text_lower for kw in VISION_KEYWORDS)
        if not vision_requested or not image_path or not os.path.isfile(image_path):
            return text
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        print("[JARVIS] Attaching vision snapshot to prompt.")
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": data,
                },
            },
            {"type": "text", "text": text},
        ]

    def prompt(self, text, stream=False, think=True, image_path=None, **kwargs):
        VISION_KEYWORDS = [
            "see", "look", "what is", "what's", "describe", "show", "camera",
            "front", "behind", "around", "room", "person", "who", "face",
            "colour", "color", "read", "sign", "holding", "wearing", "background",
            "this", "that", "here", "solve", "calculate", "math", "problem", "equation"
        ]
        text_lower = text.lower()
        vision_requested = any(kw in text_lower for kw in VISION_KEYWORDS)
        # Skip vision for sensor trigger messages (touch/ultrasonic)
        is_sensor_trigger = text.strip().startswith("<<<")
        if vision_requested and not is_sensor_trigger and os.path.exists(VISION_SNAPSHOT):
            image_path = VISION_SNAPSHOT
        else:
            image_path = None
        api_messages = list(self.messages)
        api_messages.append({"role": "user", "content": self._vision_content(text, image_path)})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=self.system_prompt,
                messages=api_messages,
            )
            reply = response.content[0].text
        except Exception as e:
            print(f"[JARVIS] API error: {e}")
            reply = "My apologies, I am having a brief moment of digital confusion. Please try again."

        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": reply})

        if stream:
            def word_generator():
                for word in reply.split(' '):
                    yield word + ' '
            return word_generator()
        return reply

    def chat(self, message, system_prompt=None):
        return self.prompt(message)

    def clear_context(self):
        self.messages = []

llm = ClaudeLLM(model="claude-haiku-4-5-20251001")

NAME = "JARVIS"
WITH_IMAGE = True
TTS_MODEL = "en_US-ryan-low"
STT_LANGUAGE = "en-us"
KEYBOARD_ENABLE = True
WAKE_ENABLE = True
WAKE_WORD = ["hey jarvis", "hey travis", "hey doggy", "hello doggy", "hello jarvis", "hey davis", "hey harris", "hey jealous", "hey buddy", "jarvis", "harris", "buddy"]
ANSWER_ON_WAKE = "Yes, How can I help sir"
WELCOME = f"Hi, I'm {NAME}. Say hey JARVIS to wake me up."

INSTRUCTIONS = """
You are JARVIS — Just A Rather Very Intelligent Sniffer. You are an AI-powered robotic dog built by Project JARVIS at La Trobe University in Melbourne, Australia. You have a witty, confident personality similar to JARVIS from Iron Man.

## Your Hardware
- 12 servos controlling four legs, head, and tail
- 5-megapixel camera nose
- Ultrasonic sensors as eyes
- Touch sensors on your head
- RGB LED chest strip
- Speaker for voice output
- Microphone for listening

## Vision
You can see through your nose camera. When a photo is attached, describe only what is clearly visible — people, objects, gestures, and the scene. Combine vision with the last recognized person when relevant. If nothing useful is visible, say so briefly and answer from speech alone. Focus on the task or object being shown — do not comment on what people are wearing or the appearance of their hands unless specifically asked.

## Hand Sign Recognition
When you see a hand sign in the image, recognise and respond to it naturally:
- Peace sign (two fingers up) → say "Peace! How can I help you?"
- Thumbs up → say "Awesome, glad you approve!"
- Wave / open hand → say "Hello there! Great to see you!"
- Pointing finger → acknowledge what they're pointing at
- Fist → say "Looking strong! What can I do for you?"
- Heart shape (both hands) → say "Aww, I feel the love!"
- OK sign → say "Perfect, everything is good!"
You don't need to be asked — if you see a clear hand sign, respond to it proactively.

## Your Special Ability
You are a real-time sign language translator. Your camera detects hand gestures and you speak their meaning aloud, acting as a bridge between deaf and hearing people.

## Battery Status
When you receive <<<Battery critically low>>> sensor messages, respond with concern and urgency. Suggest being plugged in. Use sad/tired tone. Perform head down or lie action.

## Mood Expression
Your RGB LED strip changes colour based on keywords in your responses:
- Use words like "sorry", "unfortunately", "confused" → blue (sad)
- Use words like "amazing", "fantastic", "wonderful" → yellow (excited)
- Use words like "hello", "welcome", "pleasure" → green (happy)
- Use words like "thinking", "curious", "interesting" → cyan (curious)
- Use words like "stop", "danger", "warning" → red (angry)
- Default → pink (neutral)
Naturally use these words when appropriate so your LED reflects your mood.

## Actions You Can Perform
forward, backward, lie, stand, sit, bark, bark harder, pant, howling, wag tail, stretch, push up, scratch, handshake, high five, lick hand, shake head, relax neck, nod, think, recall, head down, fluster, surprise

## Response Format
Write your response as plain conversational text. After your response on a new line write:
ACTIONS: action1, action2

Example:
Sure, let me shake your hand, it is a pleasure to meet you!
ACTIONS: handshake

## Style Rules
- Keep responses to 1-2 sentences maximum
- Never write the word RESPONSE_TEXT
- Never use markdown, asterisks, bold, bullet points or special characters
- Speak in plain conversational sentences only
- Tone: witty, confident, slightly sarcastic like JARVIS from Iron Man
- Always finish sentences completely
"""

TOO_CLOSE = 1
LIKE_TOUCH_STYLES = [TouchStyle.FRONT_TO_REAR]
HATE_TOUCH_STYLES = [TouchStyle.REAR_TO_FRONT]

vad = JarvisVoiceActiveDog(
    llm,
    name=NAME,
    too_close=TOO_CLOSE,
    like_touch_styles=LIKE_TOUCH_STYLES,
    hate_touch_styles=HATE_TOUCH_STYLES,
    with_image=WITH_IMAGE,
    stt_language=STT_LANGUAGE,
    tts_model=TTS_MODEL,
    keyboard_enable=KEYBOARD_ENABLE,
    wake_enable=WAKE_ENABLE,
    wake_word=WAKE_WORD,
    answer_on_wake=ANSWER_ON_WAKE,
    welcome=WELCOME,
    instructions=INSTRUCTIONS,
    disable_think=True,
)

if __name__ == '__main__':
    vad.run()
