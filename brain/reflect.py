import json
import re

from . import memory, ollama

PROMPT = """You just had this exchange. Extract only durable facts worth remembering forever.

Keep anything the user states about themselves: preferences, habits, their work, \
their name, how they want to be spoken to. Those are durable and always worth keeping.
Skip pleasantries and anything true only right now.
Rate importance 1-10: 1 is trivial, 10 is core to who the user is.

Exchange:
User: {question}
You: {answer}

Return JSON only, no other text:
{{"facts": [{{"text": "<one self-contained fact>", "importance": <1-10>}}]}}
If nothing is worth keeping, return {{"facts": []}}."""

THRESHOLD = 4


def extract(question, answer):
    raw = ollama.chat(
        [{"role": "user", "content": PROMPT.format(question=question, answer=answer)}],
        temperature=0.1,
    )
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return []
    try:
        return json.loads(match.group())["facts"]
    except (ValueError, KeyError):
        return []


def write_back(db, question, answer, source="conversation", floor=THRESHOLD):
    memory.learn(db, f"Q: {question}\nA: {answer}", source=source, kind="episode")
    kept = []
    for fact in extract(question, answer):
        if fact.get("importance", 0) >= min(floor, THRESHOLD):
            ids = memory.learn(db, fact["text"], source=source, kind="fact",
                               importance=float(fact["importance"]))
            kept.append((ids[0], fact["text"], fact["importance"]))
    return kept
