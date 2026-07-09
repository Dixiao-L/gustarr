"""The store: one SQLite database, WAL mode, shared by every command.

This is the only coupling point between gustarr's atomic commands —
collectors write events, enrich fills items, embed fills embeddings,
train/rank read them all, apply flips recommendation status. Nothing
talks to anything else except through these tables.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id          TEXT PRIMARY KEY,          -- canonical: domain:ns:key (see ids.py)
  domain      TEXT NOT NULL,             -- movie|series|artist|album|track
  title       TEXT,
  year        INTEGER,
  ids         TEXT NOT NULL DEFAULT '{}',-- json: {"tmdb":603,"imdb":"tt0133093","mbid":"..."}
  meta        TEXT NOT NULL DEFAULT '{}',-- json: genres, tags, overview, language, popularity
  enriched_at TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_domain ON items(domain);

-- Every taste signal, append-only. weight is the label contribution
-- (see signals.py); one row per (ts,item,kind,source,dedup) so re-syncs
-- are idempotent. dedup disambiguates genuinely distinct same-second
-- events (e.g. two tracks by one artist scrobbled in the same second
-- both mirror to the artist item) — collectors pass a stable
-- discriminator like the originating track id.
CREATE TABLE IF NOT EXISTS events (
  id      INTEGER PRIMARY KEY,
  ts      TEXT NOT NULL,                 -- ISO8601 UTC
  item_id TEXT NOT NULL REFERENCES items(id),
  kind    TEXT NOT NULL,                 -- play|complete|scrobble|loved|favorite|library_add|approve|reject|abandon
  weight  REAL NOT NULL,
  source  TEXT NOT NULL,                 -- jellyfin|lastfm|listenbrainz|arr|user
  dedup   TEXT NOT NULL DEFAULT '',
  meta    TEXT NOT NULL DEFAULT '{}',
  UNIQUE(ts, item_id, kind, source, dedup)
);
CREATE INDEX IF NOT EXISTS idx_events_item ON events(item_id);

-- Fallback → canonical id mappings recorded by merge_item, so a
-- collector re-minting a name-keyed id after enrich merged it away
-- gets transparently redirected instead of resurrecting the fallback
-- row (and re-running the whole merge every night).
CREATE TABLE IF NOT EXISTS item_aliases (
  alias_id     TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL
);

-- What the *arrs already manage. Never recommend anything in here.
CREATE TABLE IF NOT EXISTS library (
  item_id  TEXT PRIMARY KEY REFERENCES items(id),
  arr      TEXT NOT NULL,                -- sonarr|radarr|lidarr
  arr_id   INTEGER,
  status   TEXT,                         -- monitored|unmonitored|missing...
  added_at TEXT,
  meta     TEXT NOT NULL DEFAULT '{}'
);

-- The pool rank scores. Sources keep refreshing their own rows;
-- (item,source) is the identity so one item found by several sources
-- keeps all its provenance.
CREATE TABLE IF NOT EXISTS candidates (
  item_id        TEXT NOT NULL REFERENCES items(id),
  source         TEXT NOT NULL,          -- tmdb_similar|tmdb_discover|lastfm_similar|listenbrainz_cf|...
  seed_item_id   TEXT,                   -- which liked item produced this (for "why")
  external_score REAL,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  meta           TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (item_id, source)
);

CREATE TABLE IF NOT EXISTS recommendations (
  id       INTEGER PRIMARY KEY,
  run_id   TEXT NOT NULL,
  ts       TEXT NOT NULL,
  domain   TEXT NOT NULL,
  item_id  TEXT NOT NULL REFERENCES items(id),
  score    REAL NOT NULL,
  why      TEXT NOT NULL DEFAULT '{}',   -- json: {"neighbors":[...],"sources":[...],"exploration":bool}
  status   TEXT NOT NULL DEFAULT 'proposed',
           -- proposed → approved|rejected (user) → added|failed (apply)
           -- proposed → auto_added (music auto mode) | expired (TTL)
  acted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_recs_status ON recommendations(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_recs_open_item
  ON recommendations(item_id) WHERE status IN ('proposed', 'approved');

CREATE TABLE IF NOT EXISTS embeddings (
  item_id    TEXT NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,              -- float16 little-endian, length = dim*2
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, model)
);

-- Sync cursors, model metadata, anything key-value.
CREATE TABLE IF NOT EXISTS state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


# ── items ────────────────────────────────────────────────────────────


def canonical_id(conn: sqlite3.Connection, item_id: str) -> str:
    """Resolve an id through recorded merges. Collectors call this (or
    rely on upsert_item/add_event doing it) so merged fallback ids never
    come back to life."""
    row = conn.execute(
        "SELECT canonical_id FROM item_aliases WHERE alias_id=?", (item_id,)).fetchone()
    return row["canonical_id"] if row else item_id


def upsert_item(
    conn: sqlite3.Connection,
    item_id: str,
    domain: str,
    title: str | None = None,
    year: int | None = None,
    ids: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    enriched: bool = False,
) -> str:
    """Insert or non-destructively update: never blanks an existing field,
    merges ids/meta dicts key-wise (new keys win). Returns the effective
    id, which differs from item_id when a merge redirected it."""
    ts = now()
    item_id = canonical_id(conn, item_id)
    row = conn.execute("SELECT ids, meta FROM items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO items (id, domain, title, year, ids, meta, enriched_at,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, domain, title, year, json.dumps(ids or {}), json.dumps(meta or {}),
             ts if enriched else None, ts, ts),
        )
        return
    merged_ids = {**json.loads(row["ids"]), **(ids or {})}
    merged_meta = {**json.loads(row["meta"]), **(meta or {})}
    conn.execute(
        "UPDATE items SET title=COALESCE(?, title), year=COALESCE(?, year),"
        " ids=?, meta=?, enriched_at=CASE WHEN ? THEN ? ELSE enriched_at END,"
        " updated_at=? WHERE id=?",
        (title, year, json.dumps(merged_ids), json.dumps(merged_meta),
         enriched, ts, ts, item_id),
    )


def merge_item(conn: sqlite3.Connection, fallback_id: str, canonical_id: str) -> None:
    """Repoint every reference from a name-keyed fallback item to its
    resolved canonical item, then drop the fallback row. Called by enrich
    when it upgrades e.g. artist:lastfm:radiohead → artist:mbid:a74b1b7f-...
    """
    if fallback_id == canonical_id:
        return
    fb = conn.execute("SELECT * FROM items WHERE id=?", (fallback_id,)).fetchone()
    if fb is None:
        return
    upsert_item(conn, canonical_id, fb["domain"], fb["title"], fb["year"],
                json.loads(fb["ids"]), json.loads(fb["meta"]))
    # Record the redirect (and re-point any older aliases at the new
    # canonical id) so future writes under the fallback id land here.
    conn.execute("UPDATE item_aliases SET canonical_id=? WHERE canonical_id=?",
                 (canonical_id, fallback_id))
    conn.execute(
        "INSERT INTO item_aliases (alias_id, canonical_id) VALUES (?,?)"
        " ON CONFLICT(alias_id) DO UPDATE SET canonical_id=excluded.canonical_id",
        (fallback_id, canonical_id))
    # events: fallback rows may collide with existing canonical rows on the
    # uniqueness key (same scrobble synced under both ids) — drop dupes.
    conn.execute(
        "UPDATE OR IGNORE events SET item_id=? WHERE item_id=?", (canonical_id, fallback_id))
    conn.execute("DELETE FROM events WHERE item_id=?", (fallback_id,))
    conn.execute(
        "UPDATE OR IGNORE candidates SET item_id=? WHERE item_id=?", (canonical_id, fallback_id))
    conn.execute("DELETE FROM candidates WHERE item_id=?", (fallback_id,))
    conn.execute("UPDATE candidates SET seed_item_id=? WHERE seed_item_id=?",
                 (canonical_id, fallback_id))
    conn.execute(
        "UPDATE OR IGNORE library SET item_id=? WHERE item_id=?", (canonical_id, fallback_id))
    conn.execute("DELETE FROM library WHERE item_id=?", (fallback_id,))
    conn.execute("UPDATE OR IGNORE recommendations SET item_id=? WHERE item_id=?",
                 (canonical_id, fallback_id))
    conn.execute("DELETE FROM recommendations WHERE item_id=?", (fallback_id,))
    conn.execute("DELETE FROM embeddings WHERE item_id=?", (fallback_id,))
    conn.execute("DELETE FROM items WHERE id=?", (fallback_id,))


# ── events ───────────────────────────────────────────────────────────


def add_event(
    conn: sqlite3.Connection,
    ts: str,
    item_id: str,
    kind: str,
    weight: float,
    source: str,
    meta: dict[str, Any] | None = None,
    dedup: str = "",
) -> bool:
    """Returns False when the event already existed (idempotent re-sync)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO events (ts, item_id, kind, weight, source, dedup, meta)"
        " VALUES (?,?,?,?,?,?,?)",
        (ts, canonical_id(conn, item_id), kind, weight, source, dedup,
         json.dumps(meta or {})),
    )
    return cur.rowcount > 0


# ── state (sync cursors etc.) ────────────────────────────────────────


def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO state (key, value) VALUES (?,?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# ── embeddings ───────────────────────────────────────────────────────


def put_embedding(conn, item_id: str, model: str, vec: bytes, dim: int) -> None:
    conn.execute(
        "INSERT INTO embeddings (item_id, model, dim, vec, created_at) VALUES (?,?,?,?,?)"
        " ON CONFLICT(item_id, model) DO UPDATE SET vec=excluded.vec, dim=excluded.dim,"
        " created_at=excluded.created_at",
        (item_id, model, dim, vec, now()),
    )


def iter_embeddings(conn, model: str, item_ids: Iterable[str] | None = None):
    """Yields (item_id, dim, vec_bytes)."""
    if item_ids is None:
        yield from conn.execute(
            "SELECT item_id, dim, vec FROM embeddings WHERE model=?", (model,))
        return
    ids = list(item_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" * len(chunk))
        yield from conn.execute(
            f"SELECT item_id, dim, vec FROM embeddings WHERE model=? AND item_id IN ({marks})",
            (model, *chunk),
        )
