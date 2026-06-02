"""
Unit tests for ClaudeLLM class.
Uses unittest.mock to patch the Anthropic API client so no real
API calls are made and no API key is required.
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch, mock_open


# ── Minimal ClaudeLLM implementation (extracted from jarvis_chatbot.py) ──────
# We recreate it here to avoid hardware imports (pidog, picamera2, etc.)

class ClaudeLLM:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.model = model
        self.messages = []
        self.system_prompt = ""
        self.client = None  # set by tests

    def set_instructions(self, instructions):
        self.system_prompt = instructions

    def clear_context(self):
        self.messages = []

    VISION_KEYWORDS = [
        "see", "look", "what is", "what's", "describe", "show", "camera",
        "front", "behind", "around", "room", "person", "who", "face",
        "colour", "color", "read", "sign", "holding", "wearing", "background",
        "this", "that", "here", "solve", "calculate", "math", "problem", "equation"
    ]

    def _vision_content(self, text, image_path):
        text_lower = text.lower()
        vision_requested = any(kw in text_lower for kw in self.VISION_KEYWORDS)
        if not vision_requested or not image_path or not os.path.isfile(image_path):
            return text
        import base64
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": text},
        ]

    def prompt(self, text, stream=False, think=True, image_path=None, **kwargs):
        is_sensor_trigger = text.strip().startswith("<<<")
        text_lower = text.lower()
        vision_requested = any(kw in text_lower for kw in self.VISION_KEYWORDS)

        if vision_requested and not is_sensor_trigger and image_path and os.path.exists(image_path):
            content = self._vision_content(text, image_path)
        else:
            content = text
            image_path = None

        api_messages = list(self.messages)
        api_messages.append({"role": "user", "content": content})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=self.system_prompt,
                messages=api_messages,
            )
            reply = response.content[0].text
        except Exception as e:
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


# ── Helper ────────────────────────────────────────────────────────────────────

def make_llm_with_mock(reply_text="Hello! How can I help?"):
    """Create a ClaudeLLM instance with mocked Anthropic client."""
    llm = ClaudeLLM()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=reply_text)]
    mock_client.messages.create.return_value = mock_response
    llm.client = mock_client
    return llm, mock_client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestClaudeLLMInit:
    def test_default_model(self):
        llm = ClaudeLLM()
        assert llm.model == "claude-haiku-4-5-20251001"

    def test_custom_model(self):
        llm = ClaudeLLM(model="claude-opus-4-6")
        assert llm.model == "claude-opus-4-6"

    def test_empty_message_history(self):
        llm = ClaudeLLM()
        assert llm.messages == []

    def test_empty_system_prompt(self):
        llm = ClaudeLLM()
        assert llm.system_prompt == ""


class TestSetInstructions:
    def test_set_instructions_stores_prompt(self):
        llm = ClaudeLLM()
        llm.set_instructions("You are JARVIS.")
        assert llm.system_prompt == "You are JARVIS."

    def test_set_instructions_overwrite(self):
        llm = ClaudeLLM()
        llm.set_instructions("First prompt.")
        llm.set_instructions("Second prompt.")
        assert llm.system_prompt == "Second prompt."


class TestPrompt:
    def test_prompt_returns_reply(self):
        llm, mock_client = make_llm_with_mock("I am JARVIS!")
        reply = llm.prompt("Who are you?")
        assert reply == "I am JARVIS!"

    def test_prompt_appends_to_history(self):
        llm, _ = make_llm_with_mock("Sure!")
        llm.prompt("Can you bark?")
        assert len(llm.messages) == 2
        assert llm.messages[0]["role"] == "user"
        assert llm.messages[1]["role"] == "assistant"

    def test_prompt_history_accumulates(self):
        llm, _ = make_llm_with_mock("Sure!")
        llm.prompt("Question 1")
        llm.prompt("Question 2")
        assert len(llm.messages) == 4

    def test_prompt_uses_system_prompt(self):
        llm, mock_client = make_llm_with_mock("Response")
        llm.set_instructions("You are JARVIS.")
        llm.prompt("Hello")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are JARVIS."

    def test_prompt_sends_correct_model(self):
        llm, mock_client = make_llm_with_mock("Response")
        llm.prompt("Test")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_prompt_max_tokens_300(self):
        llm, mock_client = make_llm_with_mock("Response")
        llm.prompt("Test")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 300

    def test_api_error_returns_fallback(self):
        llm = ClaudeLLM()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        llm.client = mock_client
        reply = llm.prompt("Hello")
        assert "digital confusion" in reply.lower()

    def test_stream_returns_generator(self):
        llm, _ = make_llm_with_mock("Hello there friend")
        result = llm.prompt("Hi", stream=True)
        words = list(result)
        assert len(words) == 3
        assert "Hello " in words[0]

    def test_sensor_trigger_skips_vision(self):
        llm, mock_client = make_llm_with_mock("Response")
        sensor_text = "<<<Touch style you like: FRONT_TO_REAR>>>"
        llm.prompt(sensor_text, image_path="/tmp/jarvis_vision.jpg")
        call_kwargs = mock_client.messages.create.call_args[1]
        # Content should be plain text, not a list with image
        user_content = call_kwargs["messages"][-1]["content"]
        assert isinstance(user_content, str)


class TestClearContext:
    def test_clear_context_empties_history(self):
        llm, _ = make_llm_with_mock("Hi!")
        llm.prompt("Hello")
        llm.prompt("How are you?")
        assert len(llm.messages) == 4
        llm.clear_context()
        assert llm.messages == []

    def test_prompt_after_clear_starts_fresh(self):
        llm, _ = make_llm_with_mock("Fresh start!")
        llm.prompt("Old message")
        llm.clear_context()
        llm.prompt("New message")
        assert len(llm.messages) == 2


class TestChat:
    def test_chat_is_alias_for_prompt(self):
        llm, _ = make_llm_with_mock("Chat response")
        result = llm.chat("Hello")
        assert result == "Chat response"

    def test_chat_appends_to_history(self):
        llm, _ = make_llm_with_mock("Response")
        llm.chat("Message")
        assert len(llm.messages) == 2
