"""
Unit tests for JARVIS inter-process communication (IPC) file operations.
Tests the file-based communication between face_detect.py and jarvis_chatbot.py.
No hardware required — pure file I/O.
"""
import pytest
import os
import tempfile
import time


# ── Inline IPC functions from face_detect.py and jarvis_chatbot.py ───────────

def write_last_seen(name, filepath):
    """Write recognised person name to IPC file (from face_detect.py)."""
    try:
        with open(filepath, "w") as f:
            f.write(name)
        return True
    except Exception:
        return False


def get_last_seen_name(filepath):
    """Read last recognised person name (from jarvis_chatbot.py)."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                name = f.read().strip()
            return name if name else None
    except Exception:
        pass
    return None


def write_vision_snapshot(frame_data, filepath):
    """Write vision snapshot bytes (from face_detect.py)."""
    try:
        with open(filepath, "wb") as f:
            f.write(frame_data)
        return True
    except Exception:
        return False


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_last_seen(tmp_path):
    """Provide a temp file path for last_seen IPC tests."""
    return str(tmp_path / "jarvis_last_seen.txt")


@pytest.fixture
def tmp_vision(tmp_path):
    """Provide a temp file path for vision snapshot IPC tests."""
    return str(tmp_path / "jarvis_vision.jpg")


class TestLastSeenIPC:
    """Tests for /tmp/jarvis_last_seen.txt read/write."""

    def test_write_name_creates_file(self, tmp_last_seen):
        write_last_seen("ishaan", tmp_last_seen)
        assert os.path.exists(tmp_last_seen)

    def test_write_and_read_name(self, tmp_last_seen):
        write_last_seen("ishaan", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "ishaan"

    def test_write_multiple_names_returns_last(self, tmp_last_seen):
        write_last_seen("ishaan", tmp_last_seen)
        write_last_seen("luke", tmp_last_seen)
        write_last_seen("suhansa", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "suhansa"

    def test_read_nonexistent_file_returns_none(self, tmp_last_seen):
        result = get_last_seen_name(tmp_last_seen)
        assert result is None

    def test_write_empty_string_returns_none(self, tmp_last_seen):
        write_last_seen("", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result is None

    def test_write_whitespace_returns_none(self, tmp_last_seen):
        write_last_seen("   ", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result is None

    def test_name_is_stripped_of_whitespace(self, tmp_last_seen):
        write_last_seen("  parth  \n", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "parth"

    def test_all_team_members_writeable(self, tmp_last_seen):
        team = ["ishaan", "luke", "suhansa", "parth", "dhil"]
        for member in team:
            write_last_seen(member, tmp_last_seen)
            result = get_last_seen_name(tmp_last_seen)
            assert result == member

    def test_write_returns_true_on_success(self, tmp_last_seen):
        result = write_last_seen("ishaan", tmp_last_seen)
        assert result is True

    def test_capitalised_name_preserved(self, tmp_last_seen):
        write_last_seen("Ishaan", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "Ishaan"


class TestVisionSnapshotIPC:
    """Tests for /tmp/jarvis_vision.jpg read/write."""

    def test_write_snapshot_creates_file(self, tmp_vision):
        dummy_jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # fake JPEG header
        write_vision_snapshot(dummy_jpeg, tmp_vision)
        assert os.path.exists(tmp_vision)

    def test_write_snapshot_correct_content(self, tmp_vision):
        dummy_jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        write_vision_snapshot(dummy_jpeg, tmp_vision)
        with open(tmp_vision, "rb") as f:
            content = f.read()
        assert content == dummy_jpeg

    def test_overwrite_snapshot(self, tmp_vision):
        frame1 = b'\xff\xd8\xff\xe0' + b'\x01' * 100
        frame2 = b'\xff\xd8\xff\xe0' + b'\x02' * 100
        write_vision_snapshot(frame1, tmp_vision)
        write_vision_snapshot(frame2, tmp_vision)
        with open(tmp_vision, "rb") as f:
            content = f.read()
        assert content == frame2

    def test_snapshot_file_size_reasonable(self, tmp_vision):
        # Real snapshot is 320x240 JPEG at ~15-30KB
        dummy = b'\x00' * 20000
        write_vision_snapshot(dummy, tmp_vision)
        size = os.path.getsize(tmp_vision)
        assert size == 20000

    def test_write_returns_true_on_success(self, tmp_vision):
        result = write_vision_snapshot(b'\x00' * 100, tmp_vision)
        assert result is True


class TestIPCRaceCondition:
    """Test IPC file operations under rapid write/read cycles."""

    def test_rapid_writes_readable(self, tmp_last_seen):
        """Simulate face detection writing at ~10 FPS."""
        names = ["ishaan", "luke", "ishaan", "ishaan", "suhansa"]
        for name in names:
            write_last_seen(name, tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "suhansa"

    def test_read_after_rapid_writes(self, tmp_last_seen):
        """Ensure chatbot reads correct value after rapid face detection updates."""
        for _ in range(50):
            write_last_seen("ishaan", tmp_last_seen)
        result = get_last_seen_name(tmp_last_seen)
        assert result == "ishaan"
