"""What she does when nobody is talking to her."""
import random
import time

from . import decide, graph, identity, memory, ollama, store, teacher

QUESTION_PROMPT = """Someone just read this. Write one short question it leaves open \
that has a specific, checkable, factual answer the text does not already contain: a date, \
a name, a number, a place, a definition. Do not ask why someone prefers or feels \
something, do not ask about reactions or opinions, do not speculate. Output the \
question only, nothing else.

{text}"""

MAX_OPEN_GAPS = 5
SELF_TRIES_BEFORE_TEACHER = 2


def _ask_self(db, question):
    """Same answer path as chat, so idle thinking never knows less than she does awake."""
    from . import cli  # cli imports this module; resolved lazily to avoid a cycle
    return cli.answer(db, question, learn=False)[0]

def wander(db):
    """Pick something she was taught, imagine what she'd be asked about it, check if she
    can answer. Never wonders about what people confided in conversation."""
    if len(store.open_gaps(db)) >= MAX_OPEN_GAPS:
        return "enough open questions already. not wondering about more until some close."
    rows = list(
        db.execute(
            "SELECT id, text FROM memory WHERE kind IN ('fact','document')"
            " AND COALESCE(source, '') NOT IN ('conversation', 'self-resolved')"
            " ORDER BY use_count ASC, RANDOM() LIMIT 5"
        )
    )
    if not rows:
        return "nothing to wander through yet"
    row = random.choice(rows)

    question = ollama.chat(
        [{"role": "user", "content": QUESTION_PROMPT.format(text=row["text"])}],
        temperature=0.6,
        model=ollama.JUDGE_MODEL,
    ).strip().strip('"')

    identity.observe(db, "curious", 1)
    answer = _ask_self(db, question)

    if "don't know that yet" in answer.lower():
        store.note_gap(db, question)
        return f"wondered: {question}\n    -> gap. logged it."

    ids = memory.learn(db, question, source="anticipated", kind="question",
                       importance=3.0, dedupe=True)
    if ids:
        graph.connect(db, ids[0], row["id"], 0.8, relation="anticipates")
    return f"wondered: {question}\n    -> already knows this. pre-wired it."


def chase(db):
    """Revisit an old gap. New links may have made it answerable since."""
    gaps = store.open_gaps(db)
    if not gaps:
        return "no open gaps"
    gap = gaps[0]
    store.tried_gap(db, gap["id"])
    identity.observe(db, "curious", 1)

    answer = _ask_self(db, gap["question"])
    if "don't know that yet" in answer.lower():
        # Recall has variance. Two honest attempts of her own before spending a
        # teacher call on it.
        if gap["tries"] + 1 < SELF_TRIES_BEFORE_TEACHER:
            return f"chased: {gap['question']}\n    -> still a gap. will try once more before asking."
        outcome, _ = teacher.learn_gap(db, gap)
        return f"chased: {gap['question']}\n    -> {outcome}"

    ids = memory.learn(db, f"{gap['question']}\n{answer}", source="self-resolved",
                       kind="fact", importance=5.0)
    store.close_gap(db, gap["id"], ids[0])
    return f"chased: {gap['question']}\n    -> worked it out. gap closed."


def tidy(db):
    """Sleep. Merge duplicates, let unused memories fade."""
    merged = 0
    keep = []
    for r in db.execute("SELECT id, kind, vec FROM memory ORDER BY id"):
        v = store.unpack(r["vec"])
        dup = next(
            (k for k, kind, kv in keep
             if kind == r["kind"] and memory.cosine(v, kv) >= memory.SAME),
            None,
        )
        if dup:
            store.reinforce(db, dup)
            store.delete(db, [r["id"]])
            merged += 1
        else:
            keep.append((r["id"], r["kind"], v))

    faded = db.execute(
        "UPDATE memory SET importance = importance - 0.2"
        " WHERE use_count = 0 AND importance > 1"
        " AND created_at < ?", (time.time() - 86400,)
    ).rowcount
    db.commit()
    identity.feel(db, "stress", -0.1)
    return f"tidied: merged {merged}, faded {faded}"


ACTIONS = [(chase, 5), (wander, 2), (tidy, 1)]


def beat(db):
    actions, weights = zip(*ACTIONS)
    return random.choices(actions, weights=weights)[0](db)


def live(db, interval=20, beats=None):
    n = 0
    while beats is None or n < beats:
        n += 1
        print(f"  [{n}] {beat(db)}", flush=True)
        if beats is None or n < beats:
            time.sleep(interval)
