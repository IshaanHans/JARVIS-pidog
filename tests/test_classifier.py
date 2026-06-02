"""
Unit tests for the JARVIS sign language gesture classifier.
Tests the SignClassifier smoothing buffer and prediction logic.
No hardware required — uses numpy arrays as dummy keypoint data.
"""
import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock, patch


# ── Inline SignClassifier logic from sl_pipeline/classifier.py ──────────────

class SignClassifier:
    """
    Extracted from sl_pipeline/classifier.py.
    Wraps a RandomForest model with temporal smoothing.
    """
    def __init__(self, model_path=None,
                 confidence_threshold=0.45,
                 buffer_size=20,
                 confirm_threshold=8,
                 cooldown_frames=50):
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.confirm_threshold = confirm_threshold
        self.cooldown_frames = cooldown_frames
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_counter = 0
        self.model = None  # loaded separately

    def reset_buffer(self):
        self.buffer.clear()
        self.cooldown_counter = 0

    def predict(self, keypoints):
        """
        keypoints: 51-float numpy array (17 joints × 3: x, y, conf)
        Returns: (sign_name, confidence) or (None, 0.0)
        """
        if self.model is None:
            return None, 0.0

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None, 0.0

        proba = self.model.predict_proba([keypoints])[0]
        best_idx = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        if best_conf >= self.confidence_threshold:
            self.buffer.append(best_idx)

        if len(self.buffer) >= self.confirm_threshold:
            counts = {}
            for idx in self.buffer:
                counts[idx] = counts.get(idx, 0) + 1
            most_common = max(counts, key=counts.get)
            if counts[most_common] >= self.confirm_threshold:
                self.buffer.clear()
                self.cooldown_counter = self.cooldown_frames
                classes = self.model.classes_
                return classes[most_common], best_conf

        return None, 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────

SIGNS = ["HELP", "NO"]

def make_classifier():
    clf = SignClassifier()
    mock_model = MagicMock()
    mock_model.classes_ = np.array(SIGNS)
    clf.model = mock_model
    return clf, mock_model


def mock_proba(sign, confidence=0.9):
    """Build a probability array for a given sign."""
    proba = np.zeros(len(SIGNS))
    idx = SIGNS.index(sign)
    proba[idx] = confidence
    proba[1 - idx] = 1.0 - confidence
    return proba


class TestSignClassifierInit:
    def test_default_thresholds(self):
        clf = SignClassifier()
        assert clf.confidence_threshold == 0.45
        assert clf.buffer_size == 20
        assert clf.confirm_threshold == 8
        assert clf.cooldown_frames == 50

    def test_custom_thresholds(self):
        clf = SignClassifier(confidence_threshold=0.7, buffer_size=10, confirm_threshold=5)
        assert clf.confidence_threshold == 0.7
        assert clf.buffer_size == 10
        assert clf.confirm_threshold == 5

    def test_empty_buffer_on_init(self):
        clf = SignClassifier()
        assert len(clf.buffer) == 0

    def test_zero_cooldown_on_init(self):
        clf = SignClassifier()
        assert clf.cooldown_counter == 0


class TestPredictNoModel:
    def test_predict_without_model_returns_none(self):
        clf = SignClassifier()
        keypoints = np.zeros(51)
        sign, conf = clf.predict(keypoints)
        assert sign is None
        assert conf == 0.0


class TestPredictWithModel:
    def test_single_prediction_below_confirm_threshold(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        sign, conf = clf.predict(keypoints)
        # Not enough frames to confirm yet
        assert sign is None

    def test_confirm_after_enough_frames(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        sign = None
        # Feed confirm_threshold frames
        for _ in range(clf.confirm_threshold):
            sign, conf = clf.predict(keypoints)
        assert sign == "HELP"

    def test_cooldown_prevents_immediate_redetection(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        # Trigger detection
        for _ in range(clf.confirm_threshold):
            clf.predict(keypoints)
        # Immediately after, should be in cooldown
        sign, conf = clf.predict(keypoints)
        assert sign is None
        assert clf.cooldown_counter == clf.cooldown_frames - 1

    def test_low_confidence_not_added_to_buffer(self):
        clf, mock_model = make_classifier()
        # Below confidence_threshold
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.3)]
        keypoints = np.zeros(51)
        for _ in range(20):
            clf.predict(keypoints)
        assert len(clf.buffer) == 0

    def test_no_sign_detected_returns_none(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        # Only 3 frames — not enough
        for _ in range(3):
            sign, conf = clf.predict(keypoints)
        assert sign is None

    def test_both_signs_detectable(self):
        for target_sign in SIGNS:
            clf, mock_model = make_classifier()
            mock_model.predict_proba.return_value = [mock_proba(target_sign, 0.9)]
            keypoints = np.zeros(51)
            sign = None
            for _ in range(clf.confirm_threshold):
                sign, conf = clf.predict(keypoints)
            assert sign == target_sign

    def test_buffer_cleared_after_detection(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        for _ in range(clf.confirm_threshold):
            clf.predict(keypoints)
        assert len(clf.buffer) == 0


class TestResetBuffer:
    def test_reset_clears_buffer(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        for _ in range(5):
            clf.predict(keypoints)
        clf.reset_buffer()
        assert len(clf.buffer) == 0

    def test_reset_clears_cooldown(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.zeros(51)
        for _ in range(clf.confirm_threshold):
            clf.predict(keypoints)
        clf.reset_buffer()
        assert clf.cooldown_counter == 0


class TestKeypointFormat:
    def test_51_float_vector_accepted(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("HELP", 0.9)]
        keypoints = np.random.rand(51).astype(np.float32)
        sign, conf = clf.predict(keypoints)
        mock_model.predict_proba.assert_called_once()

    def test_keypoints_passed_as_list_to_model(self):
        clf, mock_model = make_classifier()
        mock_model.predict_proba.return_value = [mock_proba("NO", 0.8)]
        keypoints = np.zeros(51)
        clf.predict(keypoints)
        call_args = mock_model.predict_proba.call_args[0][0]
        assert len(call_args) == 1
        assert len(call_args[0]) == 51
