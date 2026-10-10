import array
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("BRAIN_DB", os.path.join(ROOT, "data", "brain.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    source     TEXT,
    created_at REAL NOT NULL,
    last_used  REAL,
    use_count  INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 5.0,
    vec        BLOB
);
CREATE INDEX IF NOT EXISTS memory_kind ON memory(kind);
"""


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    migrate(db)
    return db


def pack(vec):
    return array.array("f", vec).tobytes()


def unpack(blob):
    return array.array("f", blob)


def add(db, kind, text, vec, source=None, importance=5.0, confidence=0.7):
    now = time.time()
    cur = db.execute(
        "INSERT INTO memory (kind, text, source, created_at, valid_from, importance,"
        " confidence, vec) VALUES (?,?,?,?,?,?,?,?)",
        (kind, text, source, now, now, importance, confidence, pack(vec)),
    )
    db.commit()
    return cur.lastrowid


def all_vectors(db, kinds=None, live_only=True):
    q = "SELECT id, vec FROM memory WHERE vec IS NOT NULL"
    if live_only:
        q += " AND invalid_at IS NULL"
    args = ()
    if kinds:
        q += " AND kind IN (%s)" % ",".join("?" * len(kinds))
        args = tuple(kinds)
    return [(r["id"], unpack(r["vec"])) for r in db.execute(q, args)]


def get(db, ids):
    if not ids:
        return []
    q = "SELECT * FROM memory WHERE id IN (%s)" % ",".join("?" * len(ids))
    rows = {r["id"]: r for r in db.execute(q, tuple(ids))}
    return [rows[i] for i in ids if i in rows]


def touch(db, ids):
    db.executemany(
        "UPDATE memory SET use_count = use_count + 1, last_used = ? WHERE id = ?",
        [(time.time(), i) for i in ids],
    )
    db.commit()


def count(db):
    return db.execute("SELECT COUNT(*) FROM memory WHERE invalid_at IS NULL").fetchone()[0]


def supersede(db, old_id, new_id):
    """The old memory stops being true now. It stays on disk as history."""
    db.execute(
        "UPDATE memory SET invalid_at = ?, superseded_by = ? WHERE id = ? AND invalid_at IS NULL",
        (time.time(), new_id, old_id),
    )
    db.commit()


def changes(db, since=0.0, n=20):
    """Facts she stopped believing, newest first, paired with what replaced them."""
    return list(db.execute(
        "SELECT old.id old_id, old.text old_text, old.invalid_at at,"
        " new.id new_id, new.text new_text, new.source new_source"
        " FROM memory old LEFT JOIN memory new ON new.id = old.superseded_by"
        " WHERE old.invalid_at > ? ORDER BY old.invalid_at DESC LIMIT ?",
        (since, n),
    ))


GAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS gap (
    id         INTEGER PRIMARY KEY,
    question   TEXT NOT NULL,
    asked_at   REAL NOT NULL,
    times_hit  INTEGER NOT NULL DEFAULT 1,
    resolved   INTEGER NOT NULL DEFAULT 0,
    learned_id INTEGER
);
"""


def note_gap(db, question):
    db.executescript(GAP_SCHEMA)
    row = db.execute(
        "SELECT id FROM gap WHERE question = ? AND resolved = 0", (question,)
    ).fetchone()
    if row:
        db.execute("UPDATE gap SET times_hit = times_hit + 1 WHERE id = ?", (row["id"],))
        db.commit()
        return row["id"]
    cur = db.execute(
        "INSERT INTO gap (question, asked_at) VALUES (?,?)", (question, time.time())
    )
    db.commit()
    return cur.lastrowid


def open_gaps(db):
    db.executescript(GAP_SCHEMA)
    cols = {r["name"] for r in db.execute("PRAGMA table_info(gap)")}
    if "last_tried" not in cols:
        db.execute("ALTER TABLE gap ADD COLUMN last_tried REAL NOT NULL DEFAULT 0")
        db.execute("ALTER TABLE gap ADD COLUMN tries INTEGER NOT NULL DEFAULT 0")
        db.commit()
    return list(
        db.execute(
            "SELECT * FROM gap WHERE resolved = 0 AND unteachable = 0"
            " ORDER BY last_tried ASC, times_hit DESC, id"
        )
    )


def tried_gap(db, gap_id):
    db.execute(
        "UPDATE gap SET last_tried = ?, tries = tries + 1 WHERE id = ?",
        (time.time(), gap_id),
    )
    db.commit()


def close_gap(db, gap_id, learned_id):
    db.execute(
        "UPDATE gap SET resolved = 1, learned_id = ? WHERE id = ?", (learned_id, gap_id)
    )
    db.commit()


def reinforce(db, mem_id, importance=None):
    db.execute(
        "UPDATE memory SET use_count = use_count + 1, last_used = ?,"
        " importance = MAX(importance, COALESCE(?, importance)) WHERE id = ?",
        (time.time(), importance, mem_id),
    )
    db.commit()


def delete(db, ids):
    db.executemany("DELETE FROM memory WHERE id = ?", [(i,) for i in ids])
    db.commit()


SELF_SCHEMA = """
CREATE TABLE IF NOT EXISTS trait (
    name       TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    evidence   INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mood (
    name       TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def migrate(db):  # noqa: F821  (GAP_SCHEMA/SELF_SCHEMA defined below, resolved at call time)
    cols = {r["name"] for r in db.execute("PRAGMA table_info(memory)")}
    if "confidence" not in cols:
        db.execute("ALTER TABLE memory ADD COLUMN confidence REAL NOT NULL DEFAULT 0.7")
        db.execute("UPDATE memory SET confidence = 0.95 WHERE source = 'teacher'")
        db.execute("UPDATE memory SET confidence = 0.5 WHERE source = 'conversation'")
    if "invalid_at" not in cols:
        # A memory is true from valid_from until invalid_at. NULL means still true.
        db.execute("ALTER TABLE memory ADD COLUMN valid_from REAL")
        db.execute("ALTER TABLE memory ADD COLUMN invalid_at REAL")
        db.execute("ALTER TABLE memory ADD COLUMN superseded_by INTEGER")
        db.execute("UPDATE memory SET valid_from = created_at")
    db.executescript(SELF_SCHEMA)
    db.executescript(GAP_SCHEMA)
    db.executescript(LESSON_SCHEMA)
    cols = {r["name"] for r in db.execute("PRAGMA table_info(gap)")}
    if "unteachable" not in cols:
        db.execute("ALTER TABLE gap ADD COLUMN unteachable INTEGER NOT NULL DEFAULT 0")
    if "last_tried" not in cols:
        db.execute("ALTER TABLE gap ADD COLUMN last_tried REAL NOT NULL DEFAULT 0")
        db.execute("ALTER TABLE gap ADD COLUMN tries INTEGER NOT NULL DEFAULT 0")
    db.commit()


LESSON_SCHEMA = """
CREATE TABLE IF NOT EXISTS lesson (
    id         INTEGER PRIMARY KEY,
    question   TEXT NOT NULL,
    lesson     TEXT,
    model      TEXT,
    tokens_in  INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    outcome    TEXT NOT NULL,
    at         REAL NOT NULL
);
"""


def log_lesson(db, question, lesson, model, usage, outcome):
    db.execute(
        "INSERT INTO lesson (question, lesson, model, tokens_in, tokens_out, outcome, at)"
        " VALUES (?,?,?,?,?,?,?)",
        (question, lesson, model,
         getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0,
         outcome, time.time()),
    )
    db.commit()


def lessons_since(db, since):
    return db.execute("SELECT COUNT(*) FROM lesson WHERE at > ?", (since,)).fetchone()[0]


def recent_lessons(db, n=20):
    return list(db.execute("SELECT * FROM lesson ORDER BY at DESC LIMIT ?", (n,)))


DEAD, ASK_USER = 1, 2


def give_up_gap(db, gap_id, reason="no_lesson"):
    """1 = nobody can answer this; 2 = only the user can, so keep it visible to them."""
    flag = ASK_USER if reason == "ask_user" else DEAD
    db.execute("UPDATE gap SET unteachable = ? WHERE id = ?", (flag, gap_id))
    db.commit()


def ask_user_gaps(db):
    return list(db.execute(
        "SELECT * FROM gap WHERE resolved = 0 AND unteachable = ? ORDER BY times_hit DESC, id",
        (ASK_USER,),
    ))
