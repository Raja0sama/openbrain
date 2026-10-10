import math
import re

from . import decide, graph, ollama, store

SAME = 0.93
RELATED = 0.60
LINK_AT = 0.55
LINKS_PER_MEMORY = 4

# Who may overrule whom. A human and the teacher can correct anything; her own guesses
# (self-resolved, anticipated) can only correct other guesses.
GUESS_SOURCES = {"self-resolved", "anticipated"}

# Superseded memories still surface in recall, as history, at this fraction of their score.
OUTDATED = 0.6
STOP = set("a an the is are was were be been being of in on at to for and or but not now "
           "has have had s his her their its it this that".split())


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def chunk(text, size=900, overlap=150):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p).strip() if overlap and buf else p
            while len(buf) > size:
                chunks.append(buf[:size])
                buf = buf[size - overlap:]
    if buf:
        chunks.append(buf)
    return chunks


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9)


def ranked(db, vec, kinds=None, live_only=True):
    return sorted(
        ((cosine(vec, v), i) for i, v in store.all_vectors(db, kinds, live_only)),
        reverse=True,
    )


def auto_link(db, mem_id, vec):
    for score, other in ranked(db, vec)[: LINKS_PER_MEMORY + 1]:
        if other != mem_id and score >= LINK_AT:
            graph.connect(db, mem_id, other, score)


def _can_overrule(new_source, old_source):
    return new_source not in GUESS_SOURCES or old_source in GUESS_SOURCES


def reconcile(db, text, vec, source):
    """Before storing a fact, ask what it does to the facts already held.
    Returns (same_id, [ids to supersede])."""
    near = [(i, s) for s, i in ranked(db, vec, ["fact"])[:4] if s >= RELATED]
    if not near:
        return None, []
    if near[0][1] >= SAME:
        return near[0][0], []
    rows = {r["id"]: r for r in store.get(db, [i for i, _ in near])}
    # A restatement that adds no word the old memory lacks is the same fact. Decided
    # here, not by the model, so a short rewording can never replace a richer memory.
    for i, _ in near:
        if _words(text) <= _words(rows[i]["text"]):
            return i, []
    verdict = decide.compare(text, [(i, rows[i]["text"]) for i, _ in near])
    if verdict["same"]:
        return verdict["same"][0], []
    return None, [i for i in verdict["replaces"]
                  if _can_overrule(source, rows[i]["source"])]


def learn(db, text, source=None, kind="episode", importance=5.0,
          dedupe=True, confidence=0.7, correct=True):
    """correct=False stores a fact without letting it overrule any existing one:
    for facts distilled from her own answers, which are restatements, not news."""
    chunks = chunk(text) if len(text) > 900 else [text]
    ids = []
    for c, v in zip(chunks, ollama.embed(chunks)):
        replaces = []
        if dedupe and kind == "fact":
            same, replaces = reconcile(db, c, v, source)
            if not correct:
                replaces = []
            if same:
                store.reinforce(db, same, importance)
                ids.append(same)
                continue
        elif dedupe:
            hits = ranked(db, v, [kind])
            if hits and hits[0][0] >= SAME:
                store.reinforce(db, hits[0][1], importance)
                ids.append(hits[0][1])
                continue
        mem_id = store.add(db, kind, c, v, source, importance, confidence)
        auto_link(db, mem_id, v)
        for old in replaces:
            store.supersede(db, old, mem_id)
            graph.connect(db, mem_id, old, 0.9, relation="supersedes")
        ids.append(mem_id)
    return ids


def used(db, hits):
    """Mark a recall as actually used: touch the memories, wire them together."""
    ids = [r["id"] for _, r in hits]
    store.touch(db, ids)
    graph.strengthen(db, ids)


def recall(db, query, k=5, kinds=None, seeds=3, touch=True):
    """What she knows about this, live facts first. Superseded facts come along as
    history, marked, so she can say what used to be true without believing it still is."""
    qvec = ollama.embed(query)[0]
    direct = ranked(db, qvec, kinds, live_only=False)
    live = {i for i, _ in store.all_vectors(db, kinds)}
    entry = {i: s for s, i in direct[:seeds] if s > 0 and i in live}
    if not entry:
        return []

    activation = graph.spread(db, entry)
    direct_scores = {i: s for s, i in direct}
    combined = {
        i: max(direct_scores.get(i, 0.0), a * 0.9) * (1 if i in live else OUTDATED)
        for i, a in {**{i: 0 for i in direct_scores}, **activation}.items()
    }
    best = sorted(combined.items(), key=lambda kv: -kv[1])[:k]

    rows = {r["id"]: r for r in store.get(db, [i for i, _ in best])}
    best = [(i, s) for i, s in best if i in rows]
    hits = [(s, rows[i]) for i, s in best]
    if touch:
        used(db, hits)
    return hits
