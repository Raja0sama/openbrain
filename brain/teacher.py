"""When she hits a gap she cannot close herself, she asks a frontier model. Once.

One gap, one question, one lesson. Every call is logged and capped per day, so this
stays a curious learner asking a knowledgeable friend, never a scraper.
"""
import json
import os
import shutil
import subprocess
import time

import anthropic

from . import decide, graph, identity, memory, store

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
DEFAULTS = {"backend": "claude-code", "model": "claude-opus-5-5",
            "daily_cap": 20, "enabled": True}
CLI_TIMEOUT = 180
NO_LESSON = "NO_LESSON"
ASK_USER = "ASK_USER"

SYSTEM = """You are the teacher for {name}, a small local assistant that keeps everything \
it knows in an external memory. {name} asks you one question because its memory could \
not answer it.

Reply with a lesson she can store forever: two to five plain sentences of durable fact, \
each able to stand alone. Use full names and dates rather than pronouns. No preamble, \
no second person, no offer of further help.

If only {name}'s own user could know the answer (their preferences, their life, their \
reasons, what {name} is for), reply with exactly: ASK_USER
If the question has no factual answer at all, reply with exactly: NO_LESSON"""


def _load_env():
    """A key in the environment wins; <repo>/.env fills in when it is absent."""
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()


class Unavailable(Exception):
    """The teacher could not be reached. The gap stays open."""


class Declined(Exception):
    """The teacher will not teach this. reason is "ask_user" or "no_lesson"."""

    def __init__(self, reason, msg):
        super().__init__(msg)
        self.reason = reason


def settings():
    return {**DEFAULTS, **identity.load_core().get("teacher", {})}


def has_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def asked_today(db):
    return store.lessons_since(db, time.time() - 86400)


def ask(db, question, context=""):
    cfg = settings()
    if not cfg["enabled"]:
        raise Unavailable("teacher is switched off in core.json")
    if asked_today(db) >= cfg["daily_cap"]:
        raise Unavailable(f"daily cap of {cfg['daily_cap']} questions reached")

    core = identity.load_core()
    system = SYSTEM.format(name=core["name"])
    prompt = f"Question: {question}"
    if context:
        prompt = f"What {core['name']} already knows nearby:\n{context}\n\n{prompt}"

    if cfg["backend"] == "claude-code":
        text, model, usage = _ask_cli(prompt, cfg, system)
    else:
        text, model, usage = _ask_api(prompt, cfg, system)

    if text is None:
        store.log_lesson(db, question, None, model, usage, "refused")
        raise Declined("no_lesson", "teacher refused")
    if text.startswith(ASK_USER):
        store.log_lesson(db, question, None, model, usage, "ask_user")
        raise Declined("ask_user", f"only {core['serves']} can answer this")
    if not text or text.startswith(NO_LESSON):
        store.log_lesson(db, question, None, model, usage, "no_lesson")
        raise Declined("no_lesson", "nothing factual to teach")

    store.log_lesson(db, question, text, model, usage, "learned")
    return text, model


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _ask_cli(prompt, cfg, system):
    """One-shot `claude -p`. Uses the signed-in Claude Code account, no API key."""
    if not shutil.which("claude"):
        raise Unavailable("claude CLI not found on PATH")
    # A child `claude` refuses to start when it thinks it is nested inside another.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
           "--tools", "", "--system-prompt", system, prompt]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise Unavailable(f"claude CLI took longer than {CLI_TIMEOUT}s") from None
    if run.returncode != 0:
        raise Unavailable(f"claude CLI exit {run.returncode}: {run.stderr.strip()[:160]}")
    try:
        out = json.loads(run.stdout)
    except ValueError:
        raise Unavailable(f"claude CLI gave non-JSON output: {run.stdout[:120]!r}") from None

    model = next(iter(out.get("modelUsage") or {}), "claude-code")
    u = out.get("usage") or {}
    usage = _Usage(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                   + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0))
    if out.get("stop_reason") == "refusal" or out.get("is_error"):
        return None, model, usage
    return (out.get("result") or "").strip(), model, usage


def _ask_api(prompt, cfg, system):
    """Direct Anthropic API. Needs ANTHROPIC_API_KEY in the env or <repo>/.env."""
    if not has_key():
        raise Unavailable(f"no ANTHROPIC_API_KEY (put it in {ENV_PATH})")
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=cfg["model"],
            max_tokens=2048,  # a lesson is a few sentences; this is a ceiling, not a target
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise Unavailable("API key rejected") from None
    except anthropic.RateLimitError as e:
        wait = e.response.headers.get("retry-after", "?")
        raise Unavailable(f"rate limited, retry after {wait}s") from None
    except anthropic.APIStatusError as e:
        raise Unavailable(f"API error {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError:
        raise Unavailable("could not reach the API") from None

    if resp.stop_reason == "refusal":
        return None, resp.model, resp.usage
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.model, resp.usage

def store_lesson(db, gap, lesson, source, confidence):
    """The lesson stands alone so its embedding reflects what it says, not the question
    that surfaced it. The question is kept beside it as an anticipated route in."""
    ids = memory.learn(db, lesson, source=source, kind="fact",
                       importance=7.0, confidence=confidence)
    qids = memory.learn(db, gap["question"], source=source, kind="question",
                        importance=3.0)
    if ids and qids:
        graph.connect(db, qids[0], ids[0], 0.8, relation="anticipates")
    store.close_gap(db, gap["id"], ids[0])
    return ids[0]


PRIVATE_SOURCES = ("conversation", "self-resolved")


def learn_gap(db, gap):
    """Try to close one gap with the teacher. Returns (what happened, learned?)."""
    # Nothing about the user leaves this machine. Decided locally, before any call.
    core = identity.load_core()
    if decide.personal(gap["question"], core["serves"], core["name"]):
        store.give_up_gap(db, gap["id"], "ask_user")
        return (f"that's about {core['serves']} or me. not sending it anywhere; "
                "kept it for them.", False)

    # Context for the teacher comes only from taught or documented knowledge, never
    # from what was said in conversation.
    nearby = "\n".join(
        f"- {r['text'][:200]}"
        for _, r in memory.recall(db, gap["question"], k=5)
        if r["kind"] != "episode" and (r["source"] or "") not in PRIVATE_SOURCES
    )
    try:
        lesson, model = ask(db, gap["question"], nearby)
    except Unavailable as e:
        return f"still a gap. teacher unavailable: {e}", False
    except Declined as e:
        store.give_up_gap(db, gap["id"], e.reason)
        if e.reason == "ask_user":
            return f"{e}. kept it for him.", False
        return f"let it go. {e}.", False

    store_lesson(db, gap, lesson, source=f"teacher:{model}", confidence=0.9)
    identity.observe(db, "curious", 1)
    identity.feel(db, "trust", 0.05)
    return f"asked the teacher. learned it: {lesson[:90]}", True
