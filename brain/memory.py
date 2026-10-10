import math
import re

from . import graph, ollama, store

SAME = 0.93
LINK_AT = 0.55
LINKS_PER_MEMORY = 4


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


def ranked(db, vec, kinds=None):
    return sorted(
        ((cosine(vec, v), i) for i, v in store.all_vectors(db, kinds)), reverse=True
    )


def auto_link(db, mem_id, vec):
    for score, other in ranked(db, vec)[: LINKS_PER_MEMORY + 1]:
        if other != mem_id and score >= LINK_AT:
            graph.connect(db, mem_id, other, score)


def learn(db, text, source=None, kind="episode", importance=5.0,
          dedupe=True, confidence=0.7):
    chunks = chunk(text) if len(text) > 900 else [text]
    ids = []
    for c, v in zip(chunks, ollama.embed(chunks)):
        if dedupe:
            hits = ranked(db, v, [kind])
            if hits and hits[0][0] >= SAME:
                store.reinforce(db, hits[0][1], importance)
                ids.append(hits[0][1])
                continue
        mem_id = store.add(db, kind, c, v, source, importance, confidence)
        auto_link(db, mem_id, v)
        ids.append(mem_id)
    return ids


def recall(db, query, k=5, kinds=None, seeds=3):
    qvec = ollama.embed(query)[0]
    direct = ranked(db, qvec, kinds)
    entry = {i: s for s, i in direct[:seeds] if s > 0}
    if not entry:
        return []

    activation = graph.spread(db, entry)
    direct_scores = {i: s for s, i in direct}
    combined = {
        i: max(direct_scores.get(i, 0.0), a * 0.9) for i, a in activation.items()
    }
    best = sorted(combined.items(), key=lambda kv: -kv[1])[:k]

    ids = [i for i, _ in best]
    store.touch(db, ids)
    graph.strengthen(db, ids)
    return list(zip([s for _, s in best], store.get(db, ids)))
