import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS link (
    src        INTEGER NOT NULL,
    dst        INTEGER NOT NULL,
    relation   TEXT NOT NULL DEFAULT 'similar',
    weight     REAL NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS link_src ON link(src);
CREATE INDEX IF NOT EXISTS link_dst ON link(dst);
"""

HEBBIAN = 0.05
MAX_WEIGHT = 1.0


def init(db):
    db.executescript(SCHEMA)


def connect(db, a, b, weight, relation="similar"):
    if a == b:
        return
    init(db)
    for src, dst in ((a, b), (b, a)):
        db.execute(
            "INSERT INTO link (src, dst, relation, weight, created_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(src, dst, relation) DO UPDATE SET"
            " weight = MIN(?, MAX(weight, excluded.weight))",
            (src, dst, relation, weight, time.time(), MAX_WEIGHT),
        )
    db.commit()


def strengthen(db, ids):
    """Fired together, wired together."""
    init(db)
    now = time.time()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for src, dst in ((a, b), (b, a)):
                db.execute(
                    "INSERT INTO link (src, dst, relation, weight, created_at)"
                    " VALUES (?,?,'recalled',?,?)"
                    " ON CONFLICT(src, dst, relation) DO UPDATE SET"
                    " weight = MIN(?, weight + ?)",
                    (src, dst, HEBBIAN, now, MAX_WEIGHT, HEBBIAN),
                )
    db.commit()


def neighbours(db, mem_id):
    init(db)
    rows = db.execute(
        "SELECT dst, MAX(weight) w FROM link WHERE src = ? GROUP BY dst", (mem_id,)
    )
    return [(r["dst"], r["w"]) for r in rows]


def spread(db, seeds, hops=2, decay=0.6, floor=0.05):
    """Light up the seeds, let activation flow outward a couple of steps."""
    activation = dict(seeds)
    frontier = dict(seeds)
    for _ in range(hops):
        nxt = {}
        for mem_id, energy in frontier.items():
            for other, weight in neighbours(db, mem_id):
                passed = energy * weight * decay
                if passed < floor:
                    continue
                if passed > activation.get(other, 0):
                    activation[other] = passed
                    nxt[other] = passed
        if not nxt:
            break
        frontier = nxt
    return activation


def stats(db):
    init(db)
    row = db.execute(
        "SELECT COUNT(*) c, AVG(weight) w FROM link WHERE src < dst"
    ).fetchone()
    return row["c"] or 0, row["w"] or 0.0
