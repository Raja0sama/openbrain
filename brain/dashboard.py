"""Local dashboard. Serves the live SQLite state as JSON and renders it."""
import errno
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import cli, graph, heartbeat, identity, store

HERE = os.path.dirname(__file__)
THOUGHTS = deque(maxlen=40)
CHAT = deque(maxlen=60)
BUSY = threading.Lock()
THINKING = {"on": False}


def snapshot(db):
    rows = list(
        db.execute(
            "SELECT id, kind, text, source, use_count, importance, confidence,"
            " created_at, invalid_at FROM memory ORDER BY id"
        )
    )
    nodes = [
        {
            "id": r["id"],
            "kind": r["kind"],
            "text": r["text"],
            "source": r["source"] or "",
            "uses": r["use_count"],
            "importance": r["importance"],
            "confidence": r["confidence"],
            "dead": r["invalid_at"] is not None,
        }
        for r in rows
    ]
    graph.init(db)
    links = [
        {"source": r["src"], "target": r["dst"], "weight": r["w"], "relation": r["rel"]}
        for r in db.execute(
            "SELECT src, dst, MAX(weight) w,"
            " (SELECT relation FROM link l2 WHERE l2.src=link.src AND l2.dst=link.dst"
            "  ORDER BY weight DESC LIMIT 1) rel"
            " FROM link WHERE src < dst GROUP BY src, dst"
        )
    ]
    core = identity.load_core()
    return {
        "core": core,
        "nodes": nodes,
        "links": links,
        "traits": [
            {
                "name": t["name"],
                "value": t["value"],
                "evidence": t["evidence"],
                "formed": t["evidence"] >= identity.EVIDENCE_NEEDED,
            }
            for t in identity.traits(db, formed_only=False)
        ],
        "needed": identity.EVIDENCE_NEEDED,
        "mood": identity.mood(db, core),
        "baseline": core["baseline"],
        "gaps": [
            {"id": g["id"], "question": g["question"], "hits": g["times_hit"],
             "tries": g["tries"]}
            for g in store.open_gaps(db)
        ],
        "ask_user": [
            {"id": g["id"], "question": g["question"]} for g in store.ask_user_gaps(db)
        ],
        "thoughts": list(THOUGHTS),
        "chat": list(CHAT),
        "thinking": THINKING["on"],
        "at": time.time(),
    }


def think_forever(interval):
    db = store.connect()
    THINKING["on"] = True
    while True:
        try:
            line = heartbeat.beat(db)
        except Exception as e:
            line = f"stumbled: {e}"
        THOUGHTS.appendleft({"at": time.time(), "text": line})
        time.sleep(interval)


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/api/say"):
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0))
        text = (json.loads(self.rfile.read(n) or b"{}").get("text") or "").strip()
        if not text:
            return self.send_error(400)

        CHAT.append({"role": "user", "text": text, "at": time.time()})
        with BUSY:
            db = store.connect()
            history = []
            for m in list(CHAT)[-(cli.WORKING_MEMORY * 2 + 1):-1]:
                history.append(
                    {"role": "user" if m["role"] == "user" else "assistant",
                     "content": m["text"]}
                )
            try:
                reply, _, call, kept = cli.answer(db, text, history=history)
            except Exception as e:
                reply, call, kept = f"something broke: {e}", {"intent": "error"}, []
        CHAT.append({
            "role": "assistant", "text": reply, "at": time.time(),
            "intent": "refused" if call.get("touches_core") else call.get("intent", ""),
            "learned": [{"text": t, "importance": i} for _, t, i in kept],
            "changed": [{"was": o, "now": n} for o, n in call.get("changed", [])],
            "gap": cli.IDK.lower() in reply.lower(),
        })
        self._send(json.dumps({"ok": True}).encode(), "application/json")

    def do_GET(self):
        if self.path.startswith("/api/state"):
            db = store.connect()
            self._send(json.dumps(snapshot(db)).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


def serve(port=7373, think=None):
    try:
        httpd = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            raise InUse(port) from None
        raise
    print(f"http://127.0.0.1:{port}" + ("  (thinking while you watch)" if think else ""))
    if think:
        threading.Thread(target=think_forever, args=(think,), daemon=True).start()
    httpd.serve_forever()


class InUse(Exception):
    def __init__(self, port):
        self.port = port
