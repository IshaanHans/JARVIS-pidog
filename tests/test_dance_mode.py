"""Tests for examples/dance_mode.py choreography and timing."""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
import dance_mode as dm  # noqa: E402


class TestBeatTiming:
    def test_120_bpm_is_half_second(self):
        assert dm.beat_seconds(120) == 0.5

    @pytest.mark.parametrize("bpm", [39, 221, 0, -10])
    def test_bpm_out_of_range(self, bpm):
        with pytest.raises(ValueError):
            dm.beat_seconds(bpm)

    def test_servo_speed_clamped(self):
        assert dm.servo_speed(40) == 60
        assert dm.servo_speed(220) == 98

    def test_servo_speed_increases_with_bpm(self):
        assert dm.servo_speed(140) > dm.servo_speed(90)


class TestSchedule:
    def test_beat_count(self):
        # 120 BPM for 10s = 20 beats
        assert len(dm.build_schedule(120, 10)) == 20

    def test_beats_evenly_spaced(self):
        ev = dm.build_schedule(100, 6)
        gaps = {round(b.t - a.t, 4) for a, b in zip(ev, ev[1:])}
        assert gaps == {0.6}

    def test_offset_shifts_all_beats(self):
        base = dm.build_schedule(120, 4)
        shifted = dm.build_schedule(120, 4, offset=0.25)
        assert all(round(s.t - b.t, 4) == 0.25 for b, s in zip(base, shifted))

    def test_legs_and_color_only_on_bar_start(self):
        for ev in dm.build_schedule(120, 8):
            on_downbeat = ev.beat % dm.BEATS_PER_BAR == 0
            assert (ev.legs is not None) == on_downbeat
            assert (ev.color is not None) == on_downbeat

    def test_every_beat_moves_head_and_tail(self):
        for ev in dm.build_schedule(120, 8):
            assert len(ev.head) == 3 and len(ev.tail) == 1

    def test_tail_alternates(self):
        ev = dm.build_schedule(120, 4)
        assert ev[0].tail != ev[1].tail and ev[0].tail == ev[2].tail

    def test_head_style_changes_every_two_bars(self):
        ev = dm.build_schedule(120, 16)
        bar_style = lambda bar: tuple(map(tuple, (e.head for e in ev if e.bar == bar)))
        assert bar_style(0) == bar_style(1)
        assert bar_style(1) != bar_style(2)

    def test_leg_poses_are_valid_actiondict_names(self):
        valid = {'stand', 'sit', 'lie', 'half_sit', 'stretch', 'push_up'}
        assert set(dm.LEG_PATTERN) <= valid

    def test_head_positions_within_servo_limits(self):
        # Pidog limits: yaw ±90, roll ±70, pitch -45..30
        for pattern in dm.HEAD_PATTERNS.values():
            for yaw, roll, pitch in pattern:
                assert -90 <= yaw <= 90 and -70 <= roll <= 70 and -45 <= pitch <= 30

    def test_head_patterns_fill_a_bar(self):
        assert all(len(p) == dm.BEATS_PER_BAR for p in dm.HEAD_PATTERNS.values())

    def test_zero_duration_rejected(self):
        with pytest.raises(ValueError):
            dm.build_schedule(120, 0)


class TestPerform:
    def test_issues_moves_with_immediately_true(self):
        dog = MagicMock()
        dog.actions_dict = {'stand': ([[0] * 8], 'legs'), 'half_sit': ([[1] * 8], 'legs'),
                            'sit': ([[2] * 8], 'legs')}
        events = dm.build_schedule(220, 2)   # fast so the test is quick
        dm.perform(dog, events, 220)

        assert dog.head_move.call_count == len(events)
        assert dog.tail_move.call_count == len(events)
        assert dog.legs_move.call_count == sum(1 for e in events if e.legs)
        # immediately=True clears the buffer each beat so the dog never drifts behind
        for call in dog.head_move.call_args_list + dog.legs_move.call_args_list:
            assert call.kwargs["immediately"] is True

    def test_led_changes_each_bar(self):
        dog = MagicMock()
        dog.actions_dict = {n: ([[0] * 8], 'legs') for n in dm.LEG_PATTERN}
        events = dm.build_schedule(220, 2)
        dm.perform(dog, events, 220)
        assert dog.rgb_strip.set_mode.call_count == sum(1 for e in events if e.color)
