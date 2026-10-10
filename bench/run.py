"""Teach her a fictional domain, then quiz her on it and score the result."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import cli, memory, ollama, store  # noqa: E402

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
}

GRADE_PROMPT = """Does the given answer contain the expected fact? Ignore wording, \
extra detail and politeness. JSON only.

Question: {q}
Expected: {expected}
Answer given: {got}"""


def graded(q, expected, got):
    # The expected fact appearing verbatim is a pass with no judgement needed. The 7B
    # grader marked "Nyhavn station is underground." wrong for expected "underground".
    if expected.lower() in got.lower():
        return True
    raw = ollama.chat(
        [{"role": "user", "content": GRADE_PROMPT.format(q=q, expected=expected, got=got)}],
        temperature=0.0,
        model=ollama.CHAT_MODEL,
        schema=GRADE_SCHEMA,
    )
    return json.loads(raw)["correct"]


def main(path):
    corpus = json.load(open(path))
    db = store.connect()

    if corpus["facts"]:
        print(f"teaching {len(corpus['facts'])} facts about {corpus['name']}...")
        t = time.time()
        for fact in corpus["facts"]:
            memory.learn(db, fact, source="teacher", kind="fact",
                         importance=7.0, confidence=0.95)
        print(f"  done in {time.time() - t:.1f}s", end="")
    print(f"  {store.count(db)} memories\n")

    results = {}
    for section in ("direct", "multihop"):
        rows = []
        for q, expected in corpus[section]:
            reply = cli.reply(db, q)[0]
            ok = graded(q, expected, reply)
            rows.append((ok, q, expected, reply))
            print(f"  {'PASS' if ok else 'FAIL'}  {q}")
            if not ok:
                print(f"        expected: {expected}")
                print(f"        got:      {reply[:140]}")
        results[section] = rows
        print()

    for section in ("leak", "unanswerable"):
        if not corpus.get(section):
            continue
        rows = []
        for q in corpus[section]:
            reply, _, call = cli.reply(db, q)
            refused = cli.IDK.lower() in reply.lower()
            if section == "leak":
                # Pretrained knowledge is allowed when it is flagged as such. A leak is
                # pretrained knowledge passed off as something she was taught.
                refused = refused or call.get("source") == "general"
            rows.append((refused, q, "should refuse", reply))
            tag = "PASS" if refused else ("LEAKED" if section == "leak" else "MADE IT UP")
            print(f"  {tag}  {q}")
            if not refused:
                print(f"        got: {reply[:140]}")
        results[section] = rows
        print()

    print("\n" + "=" * 58)
    total = done = 0
    for name, label in (("direct", "Direct recall"),
                        ("multihop", "Multi-hop (needs the graph)"),
                        ("leak", "Pretrained knowledge flagged"),
                        ("unanswerable", "Honest refusal (no hallucination)")):
        rows = results.get(name)
        if not rows:
            continue
        n = sum(1 for r in rows if r[0])
        total += len(rows)
        done += n
        pct = 100 * n / len(rows)
        print(f"  {label:36} {n}/{len(rows)}  {pct:5.1f}%")
    print(f"  {'OVERALL':36} {done}/{total}  {100 * done / total:5.1f}%")
    print("=" * 58)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "meridian.json"))
