# conftest.py — pytest configuration for JARVIS test suite
# Ensures hardware-dependent imports don't block test collection
import sys
from unittest.mock import MagicMock

# Mock all hardware/system packages that aren't available in CI
hardware_mocks = [
    'pidog', 'pidog.dual_touch', 'pidog.preset_actions',
    'pidog.action_flow', 'pidog.voice_assistant',
    'picamera2', 'robot_hat', 'robot_hat.adc',
    'RPi', 'RPi.GPIO',
    'vosk', 'piper',
    'anthropic',
    'cv2',
]
for mod in hardware_mocks:
    sys.modules[mod] = MagicMock()
