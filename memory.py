import json
import os
from datetime import datetime

MEMORY_FILE = "jarvis_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {
        "people": {},
        "facts": [],
        "last_seen": {},
        "conversations": []
    }

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def remember_person(name, fact):
    """Store a fact about a person."""
    memory = load_memory()
    if name not in memory["people"]:
        memory["people"][name] = []
    if fact not in memory["people"][name]:
        memory["people"][name].append(fact)
    save_memory(memory)
    print(f"[MEMORY] Remembered about {name}: {fact}")

def remember_fact(fact):
    """Store a general fact."""
    memory = load_memory()
    if fact not in memory["facts"]:
        memory["facts"].append(fact)
    save_memory(memory)
    print(f"[MEMORY] Remembered fact: {fact}")

def update_last_seen(name):
    """Update when a person was last seen."""
    memory = load_memory()
    memory["last_seen"][name] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_memory(memory)

def log_conversation(role, message):
    """Store last 20 conversation exchanges."""
    memory = load_memory()
    memory["conversations"].append({
        "role": role,
        "message": message,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    # Keep only last 20
    memory["conversations"] = memory["conversations"][-20:]
    save_memory(memory)

def get_memory_context():
    """Build a memory summary string to inject into Claude's system prompt."""
    memory = load_memory()
    lines = []

    if memory["people"]:
        lines.append("## People I know:")
        for name, facts in memory["people"].items():
            last = memory["last_seen"].get(name, "unknown")
            lines.append(f"- {name}: {', '.join(facts)} (last seen: {last})")

    if memory["facts"]:
        lines.append("## Things I remember:")
        for fact in memory["facts"]:
            lines.append(f"- {fact}")

    if memory["conversations"]:
        lines.append("## Recent conversation:")
        for c in memory["conversations"][-5:]:
            lines.append(f"- {c['role']}: {c['message']}")

    return "\n".join(lines) if lines else ""

def forget(name=None):
    """Wipe memory for a person or everything."""
    memory = load_memory()
    if name:
        memory["people"].pop(name, None)
        memory["last_seen"].pop(name, None)
        print(f"[MEMORY] Forgot everything about {name}")
    else:
        memory = {"people": {}, "facts": [], "last_seen": {}, "conversations": []}
        print("[MEMORY] Memory wiped.")
    save_memory(memory)