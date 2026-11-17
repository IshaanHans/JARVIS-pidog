"""
Trick learning for JARVIS.

Lets the user teach JARVIS new tricks by voice, e.g.
    "JARVIS, learn a trick called victory: sit, high five, then howl."

Claude replies with a directive line that this module parses:
    LEARN: victory = sit, high five, howling
    FORGET: victory

Learned tricks are saved to tricks.json, registered as new ActionFlow
operations (so they work in the normal ACTIONS line), and listed in the
system prompt so Claude knows they exist.

This module has no hardware imports so it can be unit tested anywhere.
"""
import json
import os
import re

MAX_STEPS = 8
MAX_NAME_LEN = 30

_DIRECTIVE = re.compile(r'^\s*(LEARN|FORGET)\s*:\s*(.*?)\s*$', re.IGNORECASE)


def normalise_name(name):
    """Lowercase, keep letters/digits/spaces, collapse whitespace."""
    name = re.sub(r'[^a-z0-9 ]', ' ', name.lower())
    return re.sub(r'\s+', ' ', name).strip()


def parse_directives(text):
    """
    Split LEARN/FORGET directive lines out of a Claude reply.

    Returns (clean_text, learns, forgets) where learns is a list of
    (name, [steps]) and forgets is a list of names. clean_text has the
    directive lines removed so they are never spoken or parsed as actions.
    """
    kept, learns, forgets = [], [], []
    for line in text.splitlines():
        m = _DIRECTIVE.match(line)
        if not m:
            kept.append(line)
            continue
        kind, body = m.group(1).upper(), m.group(2)
        if kind == 'LEARN' and '=' in body:
            name, steps = body.split('=', 1)
            step_list = [normalise_name(s) for s in steps.split(',')]
            learns.append((normalise_name(name), [s for s in step_list if s]))
        elif kind == 'FORGET':
            forgets.append(normalise_name(body))
    return '\n'.join(kept).strip(), learns, forgets


class TrickLibrary:
    """Stores learned tricks and validates them against built-in actions."""

    def __init__(self, path, builtin_actions):
        self.path = path
        self.builtin = {a for a in builtin_actions if a != 'stop'}
        self.tricks = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    data = json.load(f)
                # Learned tricks may already be registered as actions; they aren't built-ins
                self.builtin -= set(data)
                # Drop anything that is no longer valid (e.g. SDK changed)
                self.tricks = {n: s for n, s in data.items() if self.validate(n, s) is None}
        except Exception as e:
            print(f"[TRICKS] Could not load {self.path}: {e}")
            self.tricks = {}

    def save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.tricks, f, indent=2)
        os.replace(tmp, self.path)   # atomic, so a crash can't corrupt the file

    # ── validation / editing ─────────────────────────────────────────────────
    def validate(self, name, steps):
        """Return an error string, or None if the trick is valid."""
        if not name:
            return "trick needs a name"
        if len(name) > MAX_NAME_LEN:
            return "trick name is too long"
        if name in self.builtin:
            return f"'{name}' is already a built-in action"
        if not steps:
            return "trick needs at least one step"
        if len(steps) > MAX_STEPS:
            return f"trick can have at most {MAX_STEPS} steps"
        unknown = [s for s in steps if s not in self.builtin]
        if unknown:
            return f"unknown actions: {', '.join(unknown)}"
        return None

    def learn(self, name, steps):
        error = self.validate(name, steps)
        if error:
            return False, error
        self.tricks[name] = list(steps)
        self.save()
        return True, None

    def forget(self, name):
        if name in self.tricks:
            del self.tricks[name]
            self.save()
            return True
        return False

    # ── integration helpers ──────────────────────────────────────────────────
    def prompt_section(self):
        """Text appended to the system prompt so Claude knows learned tricks."""
        if not self.tricks:
            return ''
        lines = [f"- {n}: {', '.join(s)}" for n, s in sorted(self.tricks.items())]
        return ("\n\n## Learned Tricks\nYou can perform these learned tricks by "
                "putting their name in ACTIONS:\n" + '\n'.join(lines) + '\n')


def make_operation(steps):
    """
    Build an ActionFlow OPERATIONS entry that runs each step in order.
    ActionFlow.run() handles posture changes and waits for each step.
    """
    def run_steps(action_flow):
        for step in steps:
            action_flow.run(step)
    return {"function": run_steps}
