"""Tests for examples/trick_library.py (voice-taught tricks)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from trick_library import (  # noqa: E402
    MAX_STEPS, TrickLibrary, make_operation, normalise_name, parse_directives,
)

BUILTIN = ["forward", "backward", "turn left", "turn right", "stop", "lie", "stand", "sit",
           "bark", "bark harder", "pant", "wag tail", "shake head", "stretch", "doze off",
           "push up", "howling", "twist body", "scratch", "handshake", "high five",
           "lick hand", "waiting", "feet shake", "relax neck", "nod", "think", "recall",
           "fluster", "surprise"]


@pytest.fixture
def lib(tmp_path):
    return TrickLibrary(str(tmp_path / "tricks.json"), BUILTIN)


class TestParseDirectives:
    def test_plain_reply_untouched(self):
        text = "Sure thing!\nACTIONS: nod"
        assert parse_directives(text) == (text, [], [])

    def test_learn_line_extracted_and_removed(self):
        clean, learns, forgets = parse_directives(
            "Trick learned!\nLEARN: victory = sit, high five, howling\nACTIONS: victory")
        assert clean == "Trick learned!\nACTIONS: victory"
        assert learns == [("victory", ["sit", "high five", "howling"])]
        assert forgets == []

    def test_forget_line(self):
        clean, learns, forgets = parse_directives("Gone.\nFORGET: Victory\nACTIONS: nod")
        assert forgets == ["victory"]
        assert "FORGET" not in clean

    def test_case_and_spacing_tolerant(self):
        _, learns, _ = parse_directives("learn:  Happy  Dance =Wag Tail ,  Twist Body ")
        assert learns == [("happy dance", ["wag tail", "twist body"])]

    def test_learn_after_actions_line_still_removed(self):
        clean, learns, _ = parse_directives("Done!\nACTIONS: nod\nLEARN: x = sit")
        assert clean == "Done!\nACTIONS: nod"
        assert learns == [("x", ["sit"])]

    def test_learn_without_equals_ignored(self):
        _, learns, _ = parse_directives("LEARN: victory sit nod")
        assert learns == []

    def test_empty_steps_dropped(self):
        _, learns, _ = parse_directives("LEARN: a = sit, , nod,")
        assert learns == [("a", ["sit", "nod"])]


class TestNormaliseName:
    def test_strips_punctuation(self):
        assert normalise_name("Victory!!") == "victory"

    def test_collapses_spaces(self):
        assert normalise_name("  big   bow ") == "big bow"


class TestTrickLibrary:
    def test_learn_valid_trick(self, lib):
        ok, err = lib.learn("victory", ["sit", "high five"])
        assert ok and err is None
        assert lib.tricks["victory"] == ["sit", "high five"]

    def test_learn_persists_to_disk(self, lib):
        lib.learn("victory", ["sit"])
        with open(lib.path) as f:
            assert json.load(f) == {"victory": ["sit"]}

    def test_reload_from_disk(self, lib):
        lib.learn("victory", ["sit", "nod"])
        again = TrickLibrary(lib.path, BUILTIN)
        assert again.tricks == {"victory": ["sit", "nod"]}

    def test_reject_unknown_action(self, lib):
        ok, err = lib.learn("bad", ["sit", "head down"])
        assert not ok and "head down" in err

    def test_reject_builtin_name(self, lib):
        ok, err = lib.learn("sit", ["nod"])
        assert not ok and "built-in" in err

    def test_reject_too_many_steps(self, lib):
        ok, err = lib.learn("long", ["nod"] * (MAX_STEPS + 1))
        assert not ok

    def test_max_steps_allowed(self, lib):
        ok, _ = lib.learn("long", ["nod"] * MAX_STEPS)
        assert ok

    def test_reject_empty_name(self, lib):
        assert lib.learn("", ["sit"])[0] is False

    def test_reject_no_steps(self, lib):
        assert lib.learn("x", [])[0] is False

    def test_stop_is_not_a_valid_step(self, lib):
        assert lib.learn("x", ["stop"])[0] is False

    def test_relearn_overwrites(self, lib):
        lib.learn("v", ["sit"])
        lib.learn("v", ["stand"])
        assert lib.tricks["v"] == ["stand"]

    def test_forget(self, lib):
        lib.learn("v", ["sit"])
        assert lib.forget("v") is True
        assert "v" not in lib.tricks

    def test_forget_unknown(self, lib):
        assert lib.forget("nope") is False

    def test_invalid_entries_dropped_on_load(self, tmp_path):
        path = tmp_path / "tricks.json"
        path.write_text(json.dumps({"good": ["sit"], "bad": ["moonwalk"]}))
        assert TrickLibrary(str(path), BUILTIN).tricks == {"good": ["sit"]}

    def test_reload_when_tricks_already_registered_as_actions(self, lib):
        # e.g. re-init in the same process after OPERATIONS already has the trick
        lib.learn("victory", ["sit"])
        again = TrickLibrary(lib.path, BUILTIN + ["victory"])
        assert again.tricks == {"victory": ["sit"]}

    def test_corrupt_file_starts_empty(self, tmp_path):
        path = tmp_path / "tricks.json"
        path.write_text("{not json")
        assert TrickLibrary(str(path), BUILTIN).tricks == {}

    def test_prompt_section_empty_when_no_tricks(self, lib):
        assert lib.prompt_section() == ""

    def test_prompt_section_lists_tricks(self, lib):
        lib.learn("victory", ["sit", "howling"])
        section = lib.prompt_section()
        assert "## Learned Tricks" in section
        assert "victory: sit, howling" in section

    def test_prompt_section_has_no_braces(self, lib):
        # VoiceAssistant runs .format() on the base prompt; keep ours brace-free too
        lib.learn("victory", ["sit"])
        assert "{" not in lib.prompt_section() and "}" not in lib.prompt_section()


class TestMakeOperation:
    def test_runs_steps_in_order(self):
        ran = []

        class FakeFlow:
            def run(self, action):
                ran.append(action)

        make_operation(["sit", "high five", "howling"])["function"](FakeFlow())
        assert ran == ["sit", "high five", "howling"]

    def test_has_no_posture(self):
        # Posture is handled by each step, not the composite
        assert "poseture" not in make_operation(["sit"])
