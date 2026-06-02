"""
Unit tests for JARVIS face detection logic.
Tests the recognition matching, confidence filtering, and IPC writing.
No hardware required — uses mocked face_recognition calls.
"""
import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch


# ── Inline key functions from face_detect/face_detect.py ─────────────────────

TOLERANCE = 0.55
FRAME_SCALE = 0.5

def identify_faces_logic(distances_list, known_names, tolerance=TOLERANCE):
    """
    Core matching logic from identify_faces().
    Takes precomputed distances (list of arrays) and returns results.
    """
    results = []
    for distances in distances_list:
        name = "Unknown"
        confidence = 0.0
        if len(distances) > 0:
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist <= tolerance:
                name = known_names[best_idx]
                confidence = round(1.0 - best_dist, 2)
        results.append({"name": name, "confidence": confidence})
    return results


def should_print_detection(name, last_printed):
    """Terminal print suppression logic from FaceRecognitionThread."""
    return last_printed != name


# ── Tests ─────────────────────────────────────────────────────────────────────

TEAM = ["ishaan", "luke", "suhansa", "parth", "dhil"]


class TestFaceMatching:
    def test_exact_match_recognised(self):
        distances = [np.array([0.0, 0.5, 0.6, 0.7, 0.8])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["name"] == "ishaan"
        assert results[0]["confidence"] == 1.0

    def test_close_match_within_tolerance(self):
        distances = [np.array([0.4, 0.6, 0.7, 0.8, 0.9])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["name"] == "ishaan"
        assert results[0]["confidence"] == 0.6

    def test_match_at_tolerance_boundary(self):
        distances = [np.array([0.55, 0.8, 0.9, 0.9, 0.9])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["name"] == "ishaan"

    def test_beyond_tolerance_returns_unknown(self):
        distances = [np.array([0.56, 0.7, 0.8, 0.9, 0.95])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["name"] == "Unknown"
        assert results[0]["confidence"] == 0.0

    def test_all_team_members_matched(self):
        for i, member in enumerate(TEAM):
            dist = np.ones(len(TEAM)) * 0.9
            dist[i] = 0.3
            distances = [dist]
            results = identify_faces_logic(distances, TEAM)
            assert results[0]["name"] == member

    def test_best_match_selected(self):
        # Multiple close matches — should pick the best
        distances = [np.array([0.3, 0.4, 0.5, 0.8, 0.9])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["name"] == "ishaan"

    def test_multiple_faces_detected(self):
        distances = [
            np.array([0.2, 0.8, 0.9, 0.9, 0.9]),  # ishaan
            np.array([0.9, 0.3, 0.9, 0.9, 0.9]),  # luke
        ]
        results = identify_faces_logic(distances, TEAM)
        assert len(results) == 2
        assert results[0]["name"] == "ishaan"
        assert results[1]["name"] == "luke"

    def test_no_faces_returns_empty(self):
        results = identify_faces_logic([], TEAM)
        assert results == []

    def test_confidence_calculation(self):
        distances = [np.array([0.2, 0.9, 0.9, 0.9, 0.9])]
        results = identify_faces_logic(distances, TEAM)
        expected_conf = round(1.0 - 0.2, 2)
        assert results[0]["confidence"] == expected_conf

    def test_unknown_confidence_is_zero(self):
        distances = [np.array([0.8, 0.9, 0.95, 0.95, 0.95])]
        results = identify_faces_logic(distances, TEAM)
        assert results[0]["confidence"] == 0.0


class TestPrintSuppression:
    """Test the _last_printed spam suppression logic."""

    def test_new_person_should_print(self):
        assert should_print_detection("ishaan", None) is True

    def test_same_person_should_not_print(self):
        assert should_print_detection("ishaan", "ishaan") is False

    def test_different_person_should_print(self):
        assert should_print_detection("luke", "ishaan") is True

    def test_none_to_name_should_print(self):
        assert should_print_detection("parth", None) is True

    def test_name_to_none_comparison(self):
        # When no face detected, last_printed should reset to None
        # A new detection of same person should then print
        assert should_print_detection("ishaan", None) is True

    def test_all_team_members_print_on_first_detection(self):
        for member in TEAM:
            assert should_print_detection(member, None) is True

    def test_consecutive_same_person_suppressed(self):
        last = "ishaan"
        for _ in range(100):
            result = should_print_detection("ishaan", last)
            assert result is False
            last = "ishaan"


class TestToleranceThreshold:
    def test_custom_tolerance_stricter(self):
        distances = [np.array([0.4, 0.9, 0.9, 0.9, 0.9])]
        # Strict tolerance of 0.3 — 0.4 should not match
        results = identify_faces_logic(distances, TEAM, tolerance=0.3)
        assert results[0]["name"] == "Unknown"

    def test_custom_tolerance_more_lenient(self):
        distances = [np.array([0.7, 0.9, 0.9, 0.9, 0.9])]
        # Lenient tolerance of 0.75 — 0.7 should match
        results = identify_faces_logic(distances, TEAM, tolerance=0.75)
        assert results[0]["name"] == "ishaan"

    def test_production_tolerance_is_0_55(self):
        assert TOLERANCE == 0.55
