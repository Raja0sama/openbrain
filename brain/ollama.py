import json
import urllib.request

HOST = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"


def _post(path, payload):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return _post("/api/embed", {"model": EMBED_MODEL, "input": texts})["embeddings"]


def chat(messages, temperature=0.7, model=None, schema=None):
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if schema:
        payload["format"] = schema
    return _post("/api/chat", payload)["message"]["content"]


def stream(messages, temperature=0.7, model=None):
    """Yield the reply as it is written, chunk by chunk."""
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.strip():
                part = json.loads(line)
                yield part.get("message", {}).get("content", "")
                if part.get("done"):
                    return
