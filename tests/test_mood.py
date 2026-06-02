"""
Unit tests for JARVIS mood detection (get_mood method).
Tests the keyword matching logic that drives RGB LED colour transitions.
No hardware required — pure Python string logic.
"""
import pytest
import sys
import os

# ── Inline the get_mood logic so tests run without hardware ──────────────────
def get_mood(text):
    """Extracted from JarvisVoiceActiveDog.get_mood() in examples/jarvis_chatbot.py"""
    text_lower = text.lower()
    if any(w in text_lower for w in [
        "sorry", "apologize", "confused", "error", "trouble",
        "unfortunately", "sad", "upset", "unhappy", "disappointed"
    ]):
        return "sad", "blue"
    elif any(w in text_lower for w in [
        "bark", "back", "away", "stop", "danger", "warning"
    ]):
        return "angry", "red"
    elif any(w in text_lower for w in [
        "excited", "amazing", "brilliant", "fantastic", "love",
        "great", "awesome", "wonderful"
    ]):
        return "excited", "yellow"
    elif any(w in text_lower for w in [
        "hello", "hi", "hey", "welcome", "nice to meet", "pleasure"
    ]):
        return "happy", "green"
    elif any(w in text_lower for w in [
        "thinking", "calculating", "processing", "hmm", "interesting", "curious"
    ]):
        return "curious", "cyan"
    else:
        return "neutral", "pink"


# ── Tests ────────────────────────────────────────────────────────────────────

class TestMoodSad:
    def test_sorry_triggers_sad(self):
        mood, color = get_mood("I'm sorry, I made an error.")
        assert mood == "sad"
        assert color == "blue"

    def test_unfortunately_triggers_sad(self):
        mood, color = get_mood("Unfortunately I cannot do that.")
        assert mood == "sad"
        assert color == "blue"

    def test_confused_triggers_sad(self):
        mood, color = get_mood("I'm confused about that request.")
        assert mood == "sad"
        assert color == "blue"

    def test_disappointed_triggers_sad(self):
        mood, color = get_mood("I'm disappointed I couldn't help more.")
        assert mood == "sad"
        assert color == "blue"

    def test_case_insensitive_sad(self):
        mood, color = get_mood("SORRY for the confusion!")
        assert mood == "sad"
        assert color == "blue"


class TestMoodAngry:
    def test_danger_triggers_angry(self):
        mood, color = get_mood("Danger! Stop right there.")
        assert mood == "angry"
        assert color == "red"

    def test_warning_triggers_angry(self):
        mood, color = get_mood("Warning: battery low.")
        assert mood == "angry"
        assert color == "red"

    def test_stop_triggers_angry(self):
        mood, color = get_mood("Stop! You are too close.")
        assert mood == "angry"
        assert color == "red"


class TestMoodExcited:
    def test_amazing_triggers_excited(self):
        mood, color = get_mood("That's amazing! What a fantastic result!")
        assert mood == "excited"
        assert color == "yellow"

    def test_wonderful_triggers_excited(self):
        mood, color = get_mood("Wonderful! I love that idea.")
        assert mood == "excited"
        assert color == "yellow"

    def test_awesome_triggers_excited(self):
        mood, color = get_mood("Awesome work on that problem!")
        assert mood == "excited"
        assert color == "yellow"


class TestMoodHappy:
    def test_hello_triggers_happy(self):
        mood, color = get_mood("Hello! How can I help you today?")
        assert mood == "happy"
        assert color == "green"

    def test_welcome_triggers_happy(self):
        mood, color = get_mood("Welcome! It's a pleasure to meet you.")
        assert mood == "happy"
        assert color == "green"

    def test_hey_triggers_happy(self):
        mood, color = get_mood("Hey there! Nice to meet you!")
        assert mood == "happy"
        assert color == "green"


class TestMoodCurious:
    def test_thinking_triggers_curious(self):
        mood, color = get_mood("I'm calculating the answer carefully.")
        assert mood == "curious"
        assert color == "cyan"

    def test_interesting_triggers_curious(self):
        mood, color = get_mood("That's an interesting question!")
        assert mood == "curious"
        assert color == "cyan"

    def test_hmm_triggers_curious(self):
        mood, color = get_mood("Hmm, let me consider that.")
        assert mood == "curious"
        assert color == "cyan"

    def test_calculating_triggers_curious(self):
        mood, color = get_mood("I'm calculating the result now.")
        assert mood == "curious"
        assert color == "cyan"


class TestMoodNeutral:
    def test_plain_response_is_neutral(self):
        mood, color = get_mood("The capital of France is Paris.")
        assert mood == "neutral"
        assert color == "pink"

    def test_empty_string_is_neutral(self):
        mood, color = get_mood("")
        assert mood == "neutral"
        assert color == "pink"

    def test_number_response_is_neutral(self):
        mood, color = get_mood("The answer is 42.")
        assert mood == "neutral"
        assert color == "pink"

    def test_actions_line_is_neutral(self):
        mood, color = get_mood("ACTIONS: nod, wag tail")
        assert mood == "neutral"
        assert color == "pink"


class TestMoodPriority:
    """Test that mood detection follows the correct priority order."""
    def test_sad_takes_priority_over_excited(self):
        # "sorry" + "amazing" — sad should win (checked first)
        mood, color = get_mood("I'm sorry I can't do that amazing thing.")
        assert mood == "sad"

    def test_angry_takes_priority_over_happy(self):
        # "danger" + "hello" — angry should win
        mood, color = get_mood("Hello, danger is approaching!")
        assert mood == "angry"
