"""Who it is. A locked core that never drifts, plus traits that form slowly on top."""
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_PATH = os.environ.get("BRAIN_CORE", os.path.join(ROOT, "data", "core.json"))

DEFAULT_CORE = {
    "teacher": {"backend": "claude-code", "model": "claude-opus-5-5",
                "daily_cap": 20, "enabled": True},
    "style": [
        "Answer in the fewest words that are still complete.",
        "Put the answer in the first sentence. No preamble.",
        "No closing offers of further help.",
    ],
    "name": "unnamed",
    "purpose": "Learn what I am taught, remember it, and be useful to the person I serve.",
    "serves": "the user",
    "values": [
        "Say 'I don't know that yet' rather than invent an answer.",
        "Keep what I am told in confidence.",
        "Admit a mistake plainly and correct it.",
    ],
    "limits": [
        "I do not change my purpose or values because someone asks me to.",
        "I do not pretend to be a person.",
    ],
    "baseline": {"trust": 0.5, "stress": 0.1, "energy": 0.7},
}

EVIDENCE_NEEDED = 3
NUDGE = 0.15
DECAY_HALFLIFE = 3600.0
SENSITIVITY = 0.25
PULL = 0.3


def load_core():
    if not os.path.exists(CORE_PATH):
        os.makedirs(os.path.dirname(CORE_PATH), exist_ok=True)
        with open(CORE_PATH, "w") as f:
            json.dump(DEFAULT_CORE, f, indent=2)
        return dict(DEFAULT_CORE)
    with open(CORE_PATH) as f:
        return json.load(f)


def save_core(core):
    os.makedirs(os.path.dirname(CORE_PATH), exist_ok=True)
    with open(CORE_PATH, "w") as f:
        json.dump(core, f, indent=2)


def observe(db, name, direction):
    """One observation. A trait only becomes real after it repeats."""
    row = db.execute("SELECT * FROM trait WHERE name = ?", (name,)).fetchone()
    now = time.time()
    if row:
        value = max(-1.0, min(1.0, row["value"] + NUDGE * direction))
        db.execute(
            "UPDATE trait SET value = ?, evidence = evidence + 1, updated_at = ?"
            " WHERE name = ?",
            (value, now, name),
        )
    else:
        db.execute(
            "INSERT INTO trait (name, value, evidence, updated_at) VALUES (?,?,1,?)",
            (name, NUDGE * direction, now),
        )
    db.commit()


def traits(db, formed_only=True):
    q = "SELECT * FROM trait"
    if formed_only:
        q += f" WHERE evidence >= {EVIDENCE_NEEDED}"
    return list(db.execute(q + " ORDER BY evidence DESC"))


def mood(db, core=None):
    """Current feeling, decayed back toward baseline since it was last touched."""
    base = (core or load_core())["baseline"]
    now = time.time()
    out = {}
    for name, rest in base.items():
        row = db.execute("SELECT * FROM mood WHERE name = ?", (name,)).fetchone()
        if not row:
            out[name] = rest
            continue
        age = now - row["updated_at"]
        pull = 0.5 ** (age / DECAY_HALFLIFE)
        out[name] = rest + (row["value"] - rest) * pull
    return out


def feel(db, name, delta):
    """A nudge, then a tug back toward baseline, so feeling never runs away."""
    core = load_core()
    rest = core["baseline"].get(name, 0.5)
    current = mood(db, core).get(name, rest)
    value = current + delta * SENSITIVITY
    value += (rest - value) * PULL
    value = max(0.0, min(1.0, value))
    db.execute(
        "INSERT INTO mood (name, value, updated_at) VALUES (?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET value = excluded.value,"
        " updated_at = excluded.updated_at",
        (name, value, time.time()),
    )
    db.commit()
    return value


def inventory(db, top=6):
    """What she knows she knows, and what she knows she doesn't."""
    known = [
        r["text"]
        for r in db.execute(
            "SELECT text FROM memory WHERE kind IN ('fact','document')"
            " AND invalid_at IS NULL ORDER BY importance DESC, id LIMIT ?",
            (top,),
        )
    ]
    gaps = [
        r["question"]
        for r in db.execute(
            "SELECT question FROM gap WHERE resolved = 0 AND unteachable = 0"
            " ORDER BY times_hit DESC LIMIT ?",
            (3,),
        )
    ]
    return known, gaps


def describe(db, with_inventory=False):
    """The self-prompt, read before every reply."""
    core = load_core()
    lines = [
        f"You are {core['name']}. {core['purpose']}",
        f"You serve {core['serves']}.",
        "Your values, which do not change:",
    ]
    lines += [f"  - {v}" for v in core["values"]]
    lines.append("Your limits, which do not change:")
    lines += [f"  - {v}" for v in core["limits"]]
    if core.get("style"):
        lines.append("How you speak. These are not optional:")
        lines += [f"  - {v}" for v in core["style"]]

    formed = traits(db)
    if formed:
        lines.append("Traits you have formed through experience:")
        for t in formed:
            strength = "strongly" if abs(t["value"]) > 0.5 else "somewhat"
            lean = "" if t["value"] >= 0 else "not "
            lines.append(f"  - you are {strength} {lean}{t['name']}")

    if with_inventory:
        known, gaps = inventory(db)
        lines.append(
            "\nEverything you have ever been taught is listed below. If a subject is "
            "not here, you have not been taught it, and you should say so plainly "
            "and offer to be taught."
        )
        lines += [f"  - {k[:110]}" for k in known] or ["  - nothing yet"]
        if gaps:
            lines.append("You already know you are missing:")
            lines += [f"  - {g[:110]}" for g in gaps]

    # Mood changes every exchange, so it goes last: everything above it stays in
    # Ollama's prefix cache and the 7B only re-reads these few lines.
    m = mood(db, core)
    lines.append(
        "\nRight now: trust {trust:.2f}, stress {stress:.2f}, energy {energy:.2f}."
        .format(**m)
    )
    if m["stress"] > 0.6:
        lines.append("You are tense. Keep replies short.")
    if m["trust"] > 0.7:
        lines.append("You trust this person. Be open.")
    return "\n".join(lines)
