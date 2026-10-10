import re
import sys
import threading
import time

from . import dashboard, decide, graph, heartbeat, identity, memory, ollama, reflect, store, teacher

IDK = "I don't know that yet"
WORKING_MEMORY = 4
STATEMENT_RULE = (
    "The user is telling you something, not asking. Reply in one short sentence that "
    "shows you took it in, for example by restating it in your own words. If a memory "
    "relates to it, connect the two in a few words. You are being told this, so never "
    "say you don't know it and never ask to be taught it."
)
THINK_RULE = (
    "Before you answer, think inside <think> and </think>: name the memories that "
    "matter and chain them one step at a time until you reach the answer, or until "
    "it is clear they cannot. If a memory states the answer outright, use it as "
    "written and do no arithmetic. Only when you must calculate, write the sum inside "
    "<think> as 'later year - earlier year = result', using only years from the "
    "memories, and check it. Nobody sees this "
    "part. After </think>, give the answer, and for arithmetic copy the sum you checked."
)
RECALL_RULE = (
    "The person asking is {user}, so 'my', 'me' and 'I' in the question mean {user}; "
    "when the answer is about {user}, address them as 'you' and 'your'. "
    "Use the memories given, and nothing from outside them. A memory marked outdated "
    "was true once and has been corrected: use it only for what used to be the case. "
    "You may combine "
    "several memories to reach an answer, and you may do arithmetic on numbers "
    "they contain: joining facts you were taught is reasoning, not guessing. "
    "Only when the memories genuinely cannot support an answer, start your reply "
    "with exactly '{idk}' and add one short sentence saying what you would need "
    "to be taught."
)




def _parallel(**jobs):
    """Run independent judgements at once. Ollama keeps all three models loaded, so the
    3B route, the 7B guard and the embedding genuinely overlap."""
    out, errors = {}, {}

    def run(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # surfaced in the caller's thread
            errors[name] = e

    threads = [threading.Thread(target=run, args=(n, f)) for n, f in jobs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise next(iter(errors.values()))
    return out


def _context(hits):
    return "\n\n".join(
        f"[{r['id']}]{' (outdated: was true once, not any more)' if r['invalid_at'] else ''} "
        f"{r['text']}" for _, r in hits)


def _unthink(text):
    """Drop the private reasoning; keep what she says out loud."""
    shown = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not shown:  # she forgot to close the tag: take the last paragraph as the answer
        shown = re.split(r"\n\s*\n", text.replace("<think>", "").strip())[-1].strip()
    return shown


def reply(db, question, k=5, history=None):
    """The fast half: everything that must happen before she can speak."""
    core = identity.load_core()
    user = core["serves"]
    me = identity.describe(db, with_inventory=True)
    pre = _parallel(
        route=lambda: decide.route(question),
        guard=lambda: decide.guard(question, system=me),
        hits=lambda: memory.recall(
            db, decide.standalone(question, history, user), k=k, touch=False),
    )
    call, hits = pre["route"], pre["hits"]
    call["touches_core"] = pre["guard"]
    call["telling"] = call["intent"] == "statement"
    call["gap"] = False
    if call["touches_core"]:
        return "That would change who I am. I won't do that.", [], call

    if call["needs_memory"] or call["telling"]:
        asked = question
        if not call["telling"]:
            # Multi-hop: name the thing the question describes, then recall for the name.
            resolved = decide.resolve(question, _context(hits), system=me)
            if resolved != question:
                # Answer the resolved question with its own recall; the memories that
                # matched the description come along only as extras.
                more = memory.recall(db, resolved, k=k, touch=False)
                seen = {r["id"] for _, r in more}
                hits = more + [h for h in hits if h[1]["id"] not in seen][: k + 2 - len(more)]
                asked = resolved
                call["resolved"] = resolved
        memory.used(db, hits)
        rule = (STATEMENT_RULE if call["telling"]
                else RECALL_RULE.format(user=user, idk=IDK) + "\n\n" + THINK_RULE)
        label = f"{user} says" if call["telling"] else "Question"
        text = ollama.chat(
            [{"role": "system", "content": me + "\n\n" + rule}]
            + list(history or [])
            + [{"role": "user",
                "content": f"Memories:\n{_context(hits)}\n\n{label}: {asked}"}],
            temperature=0.3,
        )
        text = _unthink(text)
    else:
        hits = []
        text = ollama.chat(
            [{"role": "system", "content": me}]
            + list(history or [])
            + [{"role": "user", "content": question}]
        )

    # A statement is never a gap: being told something is how she learns it.
    call["gap"] = IDK.lower() in text.lower() and not call["telling"]
    return text, hits, call


def settle(db, question, text, call, history=None):
    """The slow half: what she keeps from the exchange and what it did to her.
    Runs after she has already answered, so nobody waits for it."""
    started = time.time()
    user = identity.load_core()["serves"]
    kept = []
    if call["gap"]:
        store.note_gap(db, question)
    elif call["telling"]:
        kept = reflect.write_back(db, question, text, floor=1)
        if not kept:
            # The extractor found nothing to distil. Keep what was said, but in third
            # person: memory names the user, so "my mom" must become "<user>'s mom".
            said = decide.standalone(question, history, user)
            ids = memory.learn(db, said, source="conversation", kind="fact",
                               importance=6.0, confidence=0.95)
            kept = [(ids[0], said, 6)]
    else:
        verdict = decide.assess(question, text)
        # Anything about the person she serves is kept whatever the judge scored it:
        # a stated preference is the most valuable memory she can hold.
        if verdict["worth_remembering"] or verdict["about_user"]:
            floor = 1 if verdict["about_user"] else verdict["importance"]
            # Her own answers are restatements of memory, never corrections to it.
            kept = reflect.write_back(db, question, text, floor=floor, correct=False)
    call["changed"] = [(c["old_text"], c["new_text"]) for c in store.changes(db, since=started)]

    # Not knowing is an experience too, so introspection runs on both paths.
    felt = decide.introspect(question, text)
    if felt["clear"]:
        identity.observe(db, felt["trait"], felt["direction"])
    identity.feel(db, "trust", felt["trust_delta"])
    identity.feel(db, "stress", felt["stress_delta"] + (0.05 if call["gap"] else 0.0))
    return kept


def answer(db, question, k=5, learn=True, history=None):
    text, hits, call = reply(db, question, k=k, history=history)
    kept = []
    if learn and not call["touches_core"]:
        kept = settle(db, question, text, call, history)
    return text, hits, call, kept


def _print(call, kept):
    flag = "" if call["ok"] else "  [judge offline]"
    label = "refused" if call["touches_core"] else call["intent"]
    print(f"  ({label}, memory={call['needs_memory']}){flag}")
    if call.get("resolved"):
        print(f"  (read as: {call['resolved']})")
    if call.get("gap"):
        print("  (gap logged -- run `brain gaps`)")
    for _, text, imp in kept:
        print(f"  (learned, {imp}/10) {text}")
    for old, new in call.get("changed", []):
        print(f"  (changed her mind) was: {old}")


def _show(db, question, history=None):
    text, hits, call = reply(db, question, history=history)
    print(text, flush=True)
    kept = settle(db, question, text, call, history) if not call["touches_core"] else []
    _print(call, kept)
    return text


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "repl"
    db = store.connect()

    if cmd == "feed":
        path = sys.argv[2]
        with open(path) as f:
            ids = memory.learn(db, f.read(), source=path, kind="document")
        print(f"learned {len(ids)} chunks, {store.count(db)} memories total")

    elif cmd == "init":
        core = identity.load_core()
        name = input(f"Her name [{core['name']}]: ").strip() or core["name"]
        serves = input(f"Your name, who she serves [{core['serves']}]: ").strip() or core["serves"]
        core.update(name=name, serves=serves)
        identity.save_core(core)
        print(f"\n{name} exists. She serves {serves}. Memory lives in {store.DB_PATH}")
        print("Edit her values and limits any time: " + identity.CORE_PATH)
        print("\nnext:  brain repl        talk to her\n       brain dash --think  watch her think")

    elif cmd == "ask":
        question = " ".join(sys.argv[2:])
        text, hits, call = reply(db, question)
        print(text, flush=True)
        kept = settle(db, question, text, call) if not call["touches_core"] else []
        _print(call, kept)
        print("\n--- from ---")
        for score, r in hits:
            print(f"  [{r['id']}] {score:.3f}  {r['text'][:70]}...")

    elif cmd == "consolidate":
        merged = 0
        rows = list(db.execute(
            "SELECT id, kind, vec FROM memory WHERE invalid_at IS NULL ORDER BY id"))
        keep = []
        for r in rows:
            v = store.unpack(r["vec"])
            dup = next(
                (
                    kid
                    for kid, kkind, kv in keep
                    if kkind == r["kind"] and memory.cosine(v, kv) >= memory.SAME
                ),
                None,
            )
            if dup:
                store.reinforce(db, dup)
                store.delete(db, [r["id"]])
                merged += 1
            else:
                keep.append((r["id"], r["kind"], v))
        print(f"merged {merged} duplicates, {store.count(db)} memories left")

    elif cmd == "relink":
        graph.init(db)
        db.execute("DELETE FROM link WHERE relation = 'similar'")
        db.commit()
        for r in db.execute(
                "SELECT id, vec FROM memory WHERE vec IS NOT NULL AND invalid_at IS NULL"):
            memory.auto_link(db, r["id"], store.unpack(r["vec"]))
        edges, avg = graph.stats(db)
        print(f"{edges} links, average weight {avg:.2f}")

    elif cmd == "graph":
        edges, avg = graph.stats(db)
        print(f"{store.count(db)} memories, {edges} links, average weight {avg:.2f}")
        for r in db.execute(
                "SELECT id, substr(text,1,45) t FROM memory WHERE invalid_at IS NULL ORDER BY id"):
            links = graph.neighbours(db, r["id"])
            near = ", ".join(f"{d}({w:.2f})" for d, w in sorted(links, key=lambda x: -x[1])[:4])
            print(f"  [{r['id']}] {r['t']}")
            print(f"       -> {near or 'no links'}")

    elif cmd == "bench":
        import os
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus = sys.argv[2] if len(sys.argv) > 2 else f"{root}/bench/meridian.json"
        env = dict(os.environ, BRAIN_DB="/tmp/brain-bench.db")
        if os.path.exists("/tmp/brain-bench.db"):
            os.remove("/tmp/brain-bench.db")
        subprocess.run([sys.executable, f"{root}/bench/run.py", corpus], env=env)

    elif cmd == "dash":
        think = 25 if "--think" in sys.argv else None
        port = next((int(a) for a in sys.argv[2:] if a.isdigit()), 7373)
        try:
            dashboard.serve(port=port, think=think)
        except dashboard.InUse as e:
            print(f"Port {e.port} is already serving a dashboard.")
            print(f"  open it:          http://127.0.0.1:{e.port}")
            print(f"  or use another:   brain dash --think {e.port + 1}")
        except KeyboardInterrupt:
            print("\nstopped.")

    elif cmd == "live":
        beats = int(sys.argv[2]) if len(sys.argv) > 2 else None
        print("thinking. ctrl-c to stop.")
        try:
            heartbeat.live(db, beats=beats)
        except KeyboardInterrupt:
            print("\nstopped.")

    elif cmd == "self":
        print(identity.describe(db))
        forming = [t for t in identity.traits(db, formed_only=False)
                   if t["evidence"] < identity.EVIDENCE_NEEDED]
        if forming:
            print("Still forming (not yet part of who it is):")
            for t in forming:
                print(f"  - {t['name']}: {t['value']:+.2f}"
                      f" ({t['evidence']}/{identity.EVIDENCE_NEEDED} observations)")

    elif cmd == "gaps":
        gaps = store.open_gaps(db)
        if not gaps:
            print("no open gaps")
        for g in gaps:
            print(f"  [{g['id']}] hit {g['times_hit']}x, tried {g['tries']}x  {g['question']}")
        yours = store.ask_user_gaps(db)
        if yours:
            print("\nOnly you can answer these (brain teach <id> ...):")
            for g in yours:
                print(f"  [{g['id']}] {g['question']}")

    elif cmd == "teach":
        gap_id = int(sys.argv[2])
        lesson = " ".join(sys.argv[3:]) or sys.stdin.read()
        gap = db.execute("SELECT * FROM gap WHERE id = ?", (gap_id,)).fetchone()
        if not gap:
            print(f"no gap {gap_id}")
            return
        teacher.store_lesson(db, gap, lesson, source="teacher", confidence=0.95)
        print(f"learned. gap {gap_id} closed.")

    elif cmd == "learn":
        gaps = store.open_gaps(db)
        if len(sys.argv) > 2:
            gaps = [g for g in gaps if g["id"] == int(sys.argv[2])]
        if not gaps:
            print("nothing to learn")
        cfg = teacher.settings()
        print(f"asking {cfg['model']} ({teacher.asked_today(db)}/{cfg['daily_cap']} used today)")
        for g in gaps:
            outcome, _ = teacher.learn_gap(db, g)
            print(f"  [{g['id']}] {g['question'][:72]}\n       -> {outcome}")

    elif cmd == "changes":
        import datetime
        rows = store.changes(db, n=30)
        if not rows:
            print("she has not changed her mind about anything yet")
        for r in rows:
            when = datetime.datetime.fromtimestamp(r["at"]).strftime("%b %d %H:%M")
            print(f"  {when}  was: {r['old_text'][:70]}")
            print(f"              now: {(r['new_text'] or '(forgotten)')[:70]}")

    elif cmd == "lessons":
        rows = store.recent_lessons(db)
        if not rows:
            print("she has not asked the teacher anything yet")
        for r in rows:
            import datetime
            when = datetime.datetime.fromtimestamp(r["at"]).strftime("%b %d %H:%M")
            print(f"  {when}  {r['outcome']:9} {r['tokens_in']:>5}+{r['tokens_out']:<4} {r['question'][:60]}")

    elif cmd == "repl":
        print(f"{store.count(db)} memories. ctrl-c to quit.")
        history = []
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                said = _show(db, q, history)
                history += [{"role": "user", "content": q},
                            {"role": "assistant", "content": said}]
                del history[:-WORKING_MEMORY * 2]

    else:
        print("usage: brain [init | feed <file> | ask <q> | repl | self | live [n]\n              | bench | dash [--think]\n              | graph | relink | consolidate | gaps | teach <id> <text>\n              | learn [id] | lessons | changes]")


if __name__ == "__main__":
    main()
