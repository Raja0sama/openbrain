"""Local dashboard. One WebSocket pushes changes; nothing is polled or resent unchanged."""
import errno
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import cli, graph, heartbeat, identity, store, ws

HERE = os.path.dirname(__file__)
THOUGHTS = deque(maxlen=12)
CHAT = deque(maxlen=40)
BUSY = threading.Lock()
THINKING = {"on": False}
LAST_CHAT = {"at": 0.0}
QUIET_AFTER_CHAT = 60  # she stops wandering while you are talking to her
CLIENTS = set()
CLIENTS_LOCK = threading.Lock()
KINDS = ["fact", "episode", "question", "document"]


def broadcast(event):
    with CLIENTS_LOCK:
        targets = list(CLIENTS)
    for c in targets:
        try:
            c.send(event)
        except ws.Closed:
            with CLIENTS_LOCK:
                CLIENTS.discard(c)


def graph_signature(db):
    graph.init(db)
    m = db.execute("SELECT COUNT(*), MAX(id), SUM(use_count),"
                   " SUM(invalid_at IS NOT NULL) FROM memory").fetchone()
    links = db.execute("SELECT COUNT(*), SUM(weight) FROM link").fetchone()
    return tuple(m) + tuple(links)


def graph_payload(db):
    """Compact on purpose: arrays, no memory text. Text is fetched one node at a time."""
    nodes = [
        [r["id"], KINDS.index(r["kind"]) if r["kind"] in KINDS else 3, r["use_count"],
         1 if r["invalid_at"] else 0]
        for r in db.execute("SELECT id, kind, use_count, invalid_at FROM memory ORDER BY id")
    ]
    links = [
        [r["src"], r["dst"], round(r["w"], 2), r["rel"]]
        for r in db.execute(
            "SELECT src, dst, MAX(weight) w, MAX(relation) rel FROM link"
            " WHERE src < dst GROUP BY src, dst")
    ]
    return {"type": "graph", "kinds": KINDS, "nodes": nodes, "links": links}


def state_payload(db):
    core = identity.load_core()
    edges, _ = graph.stats(db)
    return {
        "type": "state",
        "core": {"name": core["name"], "serves": core["serves"]},
        "memories": store.count(db),
        "links": edges,
        "traits": [
            {"name": t["name"], "value": round(t["value"], 2), "evidence": t["evidence"],
             "formed": t["evidence"] >= identity.EVIDENCE_NEEDED}
            for t in identity.traits(db, formed_only=False)
        ],
        "needed": identity.EVIDENCE_NEEDED,
        "mood": {k: round(v, 2) for k, v in identity.mood(db, core).items()},
        "gaps": [{"id": g["id"], "question": g["question"][:140], "hits": g["times_hit"],
                  "tries": g["tries"]} for g in store.open_gaps(db)[:7]],
        "ask_user": [{"id": g["id"], "question": g["question"][:140]}
                     for g in store.ask_user_gaps(db)[:5]],
        "thinking": THINKING["on"],
    }


def watch(interval=2.0):
    """Push the graph or the state only when they actually changed."""
    db = store.connect()
    last_sig, last_state = None, None
    while True:
        try:
            sig = graph_signature(db)
            if sig != last_sig:
                last_sig = sig
                broadcast(graph_payload(db))
            st = state_payload(db)
            if st != last_state:
                last_state = st
                broadcast(st)
        except Exception:
            pass
        time.sleep(interval)


def think_forever(interval):
    db = store.connect()
    THINKING["on"] = True
    while True:
        time.sleep(interval)
        if time.time() - LAST_CHAT["at"] < QUIET_AFTER_CHAT:
            continue
        with BUSY:
            try:
                line = heartbeat.beat(db)
            except Exception as e:
                line = f"stumbled: {e}"
        thought = {"at": time.time(), "text": line}
        THOUGHTS.appendleft(thought)
        broadcast({"type": "thought", **thought})


def converse(text):
    LAST_CHAT["at"] = time.time()
    mine = {"id": f"u{time.time_ns()}", "role": "user", "text": text}
    CHAT.append(mine)
    broadcast({"type": "chat", **mine})
    rid = f"a{time.time_ns()}"
    broadcast({"type": "start", "id": rid})
    if BUSY.locked():
        broadcast({"type": "waiting", "id": rid})

    with BUSY:
        LAST_CHAT["at"] = time.time()
        db = store.connect()
        history = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["text"]}
                   for m in list(CHAT)[-(cli.WORKING_MEMORY * 2 + 1):-1]]
        try:
            said, _, call = cli.reply(
                db, text, history=history,
                on_token=lambda t: broadcast({"type": "token", "id": rid, "text": t}))
        except Exception as e:
            said, call = f"something broke: {e}", {"intent": "error", "touches_core": True}
        entry = {
            "id": rid, "role": "assistant", "text": said,
            "intent": "refused" if call.get("touches_core") else call.get("intent", ""),
            "general": call.get("source") == "general",
            "gap": cli.IDK.lower() in said.lower(), "learned": [], "changed": [],
        }
        CHAT.append(entry)
        broadcast({"type": "chat", **entry})
        if call.get("touches_core"):
            return
        # The reply is on screen. Now she keeps what matters from it.
        try:
            kept = cli.settle(db, text, said, call, history)
        except Exception:
            return
    entry["learned"] = [{"text": t, "importance": i} for _, t, i in kept]
    entry["changed"] = [{"was": o, "now": n} for o, n in call.get("changed", [])]
    broadcast({"type": "chat", **entry})


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def _socket(self):
        sock = ws.Socket(self)
        db = store.connect()
        sock.send(graph_payload(db))
        sock.send(state_payload(db))
        sock.send({"type": "history", "chat": list(CHAT), "thoughts": list(THOUGHTS)})
        with CLIENTS_LOCK:
            CLIENTS.add(sock)
        try:
            while True:
                msg = sock.recv()
                if msg.get("type") == "say" and (msg.get("text") or "").strip():
                    threading.Thread(target=converse, args=(msg["text"].strip(),),
                                     daemon=True).start()
        except (ws.Closed, OSError, ValueError):
            pass
        finally:
            with CLIENTS_LOCK:
                CLIENTS.discard(sock)
            self.close_connection = True

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            return self._socket()
        if url.path == "/api/node":
            db = store.connect()
            rows = store.get(db, [int(parse_qs(url.query).get("id", ["0"])[0])])
            if not rows:
                return self._json({"error": "no such memory"}, 404)
            r = rows[0]
            return self._json({"id": r["id"], "kind": r["kind"], "text": r["text"][:400],
                               "uses": r["use_count"], "confidence": r["confidence"],
                               "dead": r["invalid_at"] is not None})
        if url.path == "/api/table":
            db = store.connect()
            return self._json([
                [r["id"], r["kind"], r["text"][:150], r["use_count"], round(r["confidence"], 2),
                 1 if r["invalid_at"] else 0]
                for r in db.execute("SELECT * FROM memory ORDER BY id DESC LIMIT 300")])
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                return self._send(f.read(), "text/html; charset=utf-8")
        self.send_error(404)


def serve(port=7373, think=None):
    try:
        httpd = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            raise InUse(port) from None
        raise
    print(f"http://127.0.0.1:{port}" + ("  (thinking while you watch)" if think else ""))
    threading.Thread(target=watch, daemon=True).start()
    if think:
        threading.Thread(target=think_forever, args=(think,), daemon=True).start()
    httpd.serve_forever()


class InUse(Exception):
    def __init__(self, port):
        self.port = port
