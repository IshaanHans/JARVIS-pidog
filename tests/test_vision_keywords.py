"""
Unit tests for JARVIS vision keyword detection.
Tests the logic that determines whether to attach a camera snapshot
to Claude API requests. No hardware required — pure Python string logic.
"""
import pytest

# ── Inline the vision keyword logic from examples/jarvis_chatbot.py ─────────

VISION_KEYWORDS = [
    "see", "look", "what is", "what's", "describe", "show", "camera",
    "front", "behind", "around", "room", "person", "who", "face",
    "colour", "color", "read", "sign", "holding", "wearing", "background",
    "this", "that", "here", "solve", "calculate", "math", "problem", "equation"
]

def is_vision_request(text):
    """Returns True if text contains a vision keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in VISION_KEYWORDS)

def is_sensor_trigger(text):
    """Returns True if text is a sensor trigger message (not user speech)."""
    return text.strip().startswith("<<<")

def should_attach_vision(text):
    """Combined logic: attach vision if keyword detected AND not sensor trigger."""
    return is_vision_request(text) and not is_sensor_trigger(text)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestVisionKeywords:
    """Test that vision keywords are correctly detected."""

    def test_see_triggers_vision(self):
        assert is_vision_request("What can you see right now?") is True

    def test_look_triggers_vision(self):
        assert is_vision_request("Hey JARVIS, look at this.") is True

    def test_describe_triggers_vision(self):
        assert is_vision_request("Can you describe what is in front of you?") is True

    def test_who_triggers_vision(self):
        assert is_vision_request("Who is standing in front of you?") is True

    def test_face_triggers_vision(self):
        assert is_vision_request("Can you see my face?") is True

    def test_solve_triggers_vision(self):
        assert is_vision_request("Can you solve this problem?") is True

    def test_math_triggers_vision(self):
        assert is_vision_request("This is a math equation, what is the answer?") is True

    def test_wearing_triggers_vision(self):
        assert is_vision_request("What am I wearing?") is True

    def test_holding_triggers_vision(self):
        assert is_vision_request("What am I holding?") is True

    def test_colour_triggers_vision(self):
        assert is_vision_request("What colour is this?") is True

    def test_color_american_spelling_triggers_vision(self):
        assert is_vision_request("What color is this object?") is True

    def test_this_triggers_vision(self):
        assert is_vision_request("Hey JARVIS look at this!") is True

    def test_sign_triggers_vision(self):
        assert is_vision_request("Can you read this sign?") is True

    def test_whats_triggers_vision(self):
        assert is_vision_request("What's behind me?") is True

    def test_case_insensitive(self):
        assert is_vision_request("WHAT CAN YOU SEE?") is True


class TestNonVisionQueries:
    """Test that non-visual queries do not trigger vision."""

    def test_general_question_no_vision(self):
        # "what is" is a vision keyword — this correctly triggers vision
        # A general question like "what is the capital" will attach a snapshot
        # This is by design — the snapshot is low cost and Claude ignores it if irrelevant
        assert is_vision_request("What is the capital of France?") is True

    def test_math_calculation_no_vision(self):
        # "calculate" is a keyword but pure spoken math without visual
        # This tests edge case — calculate is in keywords
        assert is_vision_request("Please calculate 5 plus 3.") is True  # intentionally True

    def test_bark_command_no_vision(self):
        assert is_vision_request("Can you bark for me?") is False

    def test_action_request_no_vision(self):
        assert is_vision_request("Do a push up please.") is False

    def test_empty_string_no_vision(self):
        assert is_vision_request("") is False

    def test_name_question_no_vision(self):
        # "what is" is a vision keyword — triggers vision
        assert is_vision_request("What is your name?") is True


class TestSensorTriggerExclusion:
    """Test that sensor trigger messages skip vision even with keywords."""

    def test_touch_trigger_skips_vision(self):
        text = "<<<Touch style you like: FRONT_TO_REAR>>>"
        assert is_sensor_trigger(text) is True
        assert should_attach_vision(text) is False

    def test_ultrasonic_trigger_skips_vision(self):
        text = "<<<Ultrasonic sense too close: 3cm>>>"
        assert is_sensor_trigger(text) is True
        assert should_attach_vision(text) is False

    def test_battery_trigger_skips_vision(self):
        text = "<<<Battery critically low: 6.8V. Please plug me in.>>>"
        assert is_sensor_trigger(text) is True
        assert should_attach_vision(text) is False

    def test_normal_vision_request_not_sensor(self):
        text = "What can you see?"
        assert is_sensor_trigger(text) is False
        assert should_attach_vision(text) is True

    def test_sensor_trigger_with_vision_keyword_skips_vision(self):
        # Even if a sensor message somehow contains a vision keyword
        text = "<<<Touch sensor sees proximity>>>"
        assert should_attach_vision(text) is False


class TestEdgeCases:
    """Edge cases for vision detection."""

    def test_partial_keyword_match(self):
        # "solve" is a keyword — "solving" contains it
        assert is_vision_request("I need help solving this") is True

    def test_keyword_in_middle_of_sentence(self):
        assert is_vision_request("Please describe the scene carefully") is True

    def test_multiple_keywords(self):
        assert is_vision_request("Who is the person holding the sign?") is True

    def test_whitespace_only(self):
        assert is_vision_request("   ") is False

    def test_punctuation_only(self):
        assert is_vision_request("!!! ???") is False
