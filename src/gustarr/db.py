"""The store: one SQLite database, WAL mode, shared by every command.

This is the only coupling point between gustarr's atomic commands —
collectors write events, enrich fills items, embed fills embeddings,
train/rank read them all, apply flips recommendation status. Nothing
talks to anything else except through these tables.

Identity (schema v3): items carry a surrogate integer id; every external
name for an item — tmdb/tvdb/imdb ids, MusicBrainz ids, Jellyfin ids,
and each spelling of a name (incl. MusicBrainz aliases) — is one row in
``identities`` pointing at that item. All writes resolve through
``resolve_item``: a spelling seen before lands on the same item on
arrival, and attaching a newly-learned authoritative id either extends
the item or reveals that two items were one, in which case
``merge_items`` repoints the children once. Key normalization is
lookup-time policy — tightening it re-normalizes one small table, never
event history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import ids as ids_mod

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id          INTEGER PRIMARY KEY,       -- surrogate — stable across renames/aliases
  domain      TEXT NOT NULL,             -- movie|series|artist|album|track
  title       TEXT,
  year        INTEGER,
  meta        TEXT NOT NULL DEFAULT '{}',-- json: genres, tags, overview, language, popularity
  enriched_at TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_domain ON items(domain);

-- Every name an item goes by, external ids and spellings alike. Keys are
-- stored normalized (ids.normalize_key) — ns 'name' covers human names
-- from any source plus MusicBrainz aliases, so romaji/kana/width
-- variants of one artist all resolve to one row set → one item.
CREATE TABLE IF NOT EXISTS identities (
  domain  TEXT NOT NULL,
  ns      TEXT NOT NULL,                 -- tmdb|tvdb|imdb|mbid|jellyfin|name
  key     TEXT NOT NULL,
  item_id INTEGER NOT NULL REFERENCES items(id),
  PRIMARY KEY (domain, ns, key)
);
CREATE INDEX IF NOT EXISTS idx_identities_item ON identities(item_id);

-- Every taste signal, append-only, owned by a profile. weight is the
-- label contribution (see signals.py) — one row per
-- (profile,ts,item,kind,source,dedup) so re-syncs are idempotent.
-- dedup disambiguates genuinely distinct same-second events — collectors
-- pass a stable discriminator like the originating track id.
CREATE TABLE IF NOT EXISTS events (
  id      INTEGER PRIMARY KEY,
  profile TEXT NOT NULL DEFAULT 'default',
  ts      TEXT NOT NULL,                 -- ISO8601 UTC
  item_id INTEGER NOT NULL REFERENCES items(id),
  kind    TEXT NOT NULL,                 -- play|complete|scrobble|loved|favorite|library_add|approve|reject|abandon
  weight  REAL NOT NULL,
  source  TEXT NOT NULL,                 -- jellyfin|lastfm|listenbrainz|arr|user
  dedup   TEXT NOT NULL DEFAULT '',
  meta    TEXT NOT NULL DEFAULT '{}',
  UNIQUE(profile, ts, item_id, kind, source, dedup)
);
CREATE INDEX IF NOT EXISTS idx_events_item ON events(profile, item_id);

-- What the *arrs already manage. Never recommend anything in here.
CREATE TABLE IF NOT EXISTS library (
  item_id  INTEGER PRIMARY KEY REFERENCES items(id),
  arr      TEXT NOT NULL,                -- sonarr|radarr|lidarr
  arr_id   INTEGER,
  status   TEXT,                         -- monitored|unmonitored|missing...
  added_at TEXT,
  meta     TEXT NOT NULL DEFAULT '{}'
);

-- The pool rank scores. Sources keep refreshing their own rows —
-- (profile,item,source) is the identity so one item found by several
-- sources keeps all its provenance.
CREATE TABLE IF NOT EXISTS candidates (
  profile        TEXT NOT NULL DEFAULT 'default',
  item_id        INTEGER NOT NULL REFERENCES items(id),
  source         TEXT NOT NULL,          -- tmdb_similar|tmdb_discover|lastfm_similar|listenbrainz_cf|...
  seed_item_id   INTEGER,                -- which liked item produced this (for "why")
  external_score REAL,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  meta           TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (profile, item_id, source)
);

CREATE TABLE IF NOT EXISTS recommendations (
  id       INTEGER PRIMARY KEY,
  profile  TEXT NOT NULL DEFAULT 'default',
  run_id   TEXT NOT NULL,
  ts       TEXT NOT NULL,
  domain   TEXT NOT NULL,
  item_id  INTEGER NOT NULL REFERENCES items(id),
  score    REAL NOT NULL,
  why      TEXT NOT NULL DEFAULT '{}',   -- json: {"neighbors":[...],"sources":[...],"exploration":bool}
  status   TEXT NOT NULL DEFAULT 'proposed',
           -- proposed → approved|rejected|snoozed (user) → added|failed (apply)
           -- proposed → auto_added (music auto mode) | expired (TTL)
  acted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_recs_status ON recommendations(profile, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_recs_open_item
  ON recommendations(profile, item_id) WHERE status IN ('proposed', 'approved');

CREATE TABLE IF NOT EXISTS embeddings (
  item_id    INTEGER NOT NULL REFERENCES items(id),
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

SCHEMA_VERSION = "3"

# state keys that belong to a profile (cursors, taste models); everything
# else in state — settings, arr inventory, API caches — is operator/global.
_PROFILE_STATE_PREFIXES = ("model:", "centroid:", "lastfm:", "jellyfin:")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _migrate_v2(conn)
    _migrate_v3(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    _exec_schema(conn)
    conn.execute(
        "INSERT INTO state (key, value) VALUES ('schema_version', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (SCHEMA_VERSION,))
    conn.commit()
    return conn


def _exec_schema(conn: sqlite3.Connection) -> None:
    # statement-by-statement instead of executescript: executescript
    # force-commits any open transaction, which would silently break the
    # migrations' atomicity. complete_statement() reassembles fragments a
    # naive split breaks on ';' inside comments or strings.
    buf = ""
    for chunk in SCHEMA.split(";"):
        buf += chunk + ";"
        if sqlite3.complete_statement(buf):
            if buf.strip(" \n;"):
                conn.execute(buf)
            buf = ""


def _atomic_migration(conn: sqlite3.Connection, work) -> None:
    """Run a migration as ONE transaction, DDL included. Python's sqlite3
    legacy transaction control autocommits DDL (ALTER/CREATE/DROP run
    outside the implicit transaction), so `with conn:` alone leaves a
    crashed migration half-committed — renamed tables, missing data, and
    a store the version detection then misreads as healthy. Explicit
    BEGIN IMMEDIATE under isolation_level=None makes SQLite's
    transactional DDL actually transactional, takes the write lock up
    front (a concurrent connect blocks, then sees the finished version),
    and a crash at any point rolls back to the pre-migration store."""
    prev = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        try:
            work()
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass  # the journal rolls back on next open
            raise
    finally:
        conn.isolation_level = prev


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2 (profile columns); see git history for the v1 shape. Kept so
    a pre-profile store still upgrades along the full chain."""
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "events" not in tables:
        return
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    if "profile" in cols:
        return

    def work() -> None:
        event_cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
        if "dedup" not in event_cols:  # pre-dedup v1 stores
            conn.execute("ALTER TABLE events ADD COLUMN dedup TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE events ADD COLUMN profile TEXT NOT NULL DEFAULT 'default'")
        conn.execute(
            "ALTER TABLE candidates ADD COLUMN profile TEXT NOT NULL DEFAULT 'default'")
        conn.execute(
            "ALTER TABLE recommendations ADD COLUMN profile TEXT NOT NULL DEFAULT 'default'")
        for prefix in _PROFILE_STATE_PREFIXES:
            conn.execute(
                "UPDATE OR REPLACE state SET key = 'p:default:' || key"
                " WHERE key LIKE ? AND key NOT LIKE 'p:%'", (prefix + "%",))

    _atomic_migration(conn, work)


def _migrate_v3(conn: sqlite3.Connection) -> None:
    """v2 → v3: surrogate integer item ids + the identities table.

    Every old TEXT item id (domain:ns:key), every entry in the old
    items.ids JSON, the Jellyfin id v2 kept in meta, and every
    item_aliases row becomes an identities row pointing at the new
    integer id. Two v2 items claiming the same identity are resolved by
    the runtime attach rule — an authoritative-key collision proves one
    entity and the rows merge; a name collision between two
    authoritative items keeps them apart (the first writer keeps the
    key). Everything, DDL included, runs in one genuine transaction
    (_atomic_migration): a crash at any point leaves v2 intact.
    """
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "items_v2" in tables:
        raise RuntimeError(
            "interrupted v2→v3 migration detected (items_v2 tables present);"
            " restore the store from a backup before upgrading")
    if "items" not in tables:
        return  # fresh database: SCHEMA creates v3 directly
    items_cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)")}
    if "ids" not in items_cols:
        return  # already v3

    def work() -> None:
        for t in ("items", "events", "candidates", "recommendations",
                  "library", "embeddings"):
            conn.execute(f"ALTER TABLE {t} RENAME TO {t}_v2")
        for idx in ("idx_items_domain", "idx_events_item", "idx_recs_status",
                    "idx_recs_open_item"):
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
        _exec_schema(conn)

        # ── every v2 item's identity claims, in creation order ──────
        rows = conn.execute(
            "SELECT * FROM items_v2 ORDER BY created_at, id").fetchall()
        rowmap = {row["id"]: row for row in rows}
        order = [row["id"] for row in rows]
        pos = {old: i for i, old in enumerate(order)}
        claims: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            mine: list[tuple[str, str]] = []

            def claim(ns: str, key: Any, mine: list[tuple[str, str]] = mine) -> None:
                ns = "name" if ns == "lastfm" else ns
                k = ids_mod.normalize_key(str(key))
                if k and (ns, k) not in mine:
                    mine.append((ns, k))

            try:
                _d, ns0, key0 = row["id"].split(":", 2)
                claim(ns0, key0)
            except ValueError:
                pass
            for ns, key in json.loads(row["ids"] or "{}").items():
                # artist_mbid is a relation to another item, not a name
                if ns != "artist_mbid" and key is not None and str(key).strip():
                    claim(ns, key)
            jf = json.loads(row["meta"] or "{}").get("jellyfin_id")
            if jf:
                claim("jellyfin", jf)
            claims[row["id"]] = mine

        # ── union same-entity claims (the attach_identity rule) ─────
        parent = {old: old for old in order}
        strong = {old: any(ns not in ("name", "jellyfin") for ns, _ in claims[old])
                  for old in order}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        owner: dict[tuple[str, str, str], str] = {}
        for old in order:
            domain = rowmap[old]["domain"]
            for ns, key in claims[old]:
                first = owner.setdefault((domain, ns, key), old)
                a, b = find(first), find(old)
                if a == b:
                    continue
                # a shared name/jellyfin key between two authoritative
                # items proves difference: the first writer keeps the key
                if ns in ("name", "jellyfin") and strong[a] and strong[b]:
                    continue
                root, gone = (a, b) if strong[a] or not strong[b] else (b, a)
                parent[gone] = root
                strong[root] = strong[root] or strong[gone]

        groups: dict[str, list[str]] = {}
        for old in order:
            groups.setdefault(find(old), []).append(old)

        # ── one v3 item per group; fields from the best member ──────
        id_map: dict[str, int] = {}
        for members in groups.values():
            members.sort(key=pos.__getitem__)
            rep = next(
                (m for m in members
                 if any(ns not in ("name", "jellyfin") for ns, _ in claims[m])),
                members[0])
            meta: dict[str, Any] = {}
            for m in [rep] + [m for m in members if m != rep]:
                for k, v in json.loads(rowmap[m]["meta"] or "{}").items():
                    meta.setdefault(k, v)
            r = rowmap[rep]
            cur = conn.execute(
                "INSERT INTO items (domain, title, year, meta, enriched_at,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (r["domain"], r["title"], r["year"], json.dumps(meta),
                 r["enriched_at"], rowmap[members[0]]["created_at"], r["updated_at"]))
            for m in members:
                id_map[m] = cur.lastrowid
        conn.executemany(
            "INSERT INTO identities (domain, ns, key, item_id) VALUES (?,?,?,?)",
            [(d, ns, key, id_map[find(first)])
             for (d, ns, key), first in owner.items()])

        alias_map: dict[str, int] = {}
        if "item_aliases" in tables:
            # v2 alias redirects: same-entity assertions already acted on
            # (events moved at merge time). A key some live item owns
            # stays with that item — v2's conflict-skip semantics.
            for row in conn.execute("SELECT * FROM item_aliases").fetchall():
                target = id_map.get(row["canonical_id"])
                if target is None:
                    continue
                alias_map[row["alias_id"]] = target
                try:
                    d, ns, key = row["alias_id"].split(":", 2)
                except ValueError:
                    continue
                ns = "name" if ns == "lastfm" else ns
                norm = ids_mod.normalize_key(key)
                if norm:
                    conn.execute(
                        "INSERT OR IGNORE INTO identities (domain, ns, key, item_id)"
                        " VALUES (?,?,?,?)", (d, ns, norm, target))
            conn.execute("DROP TABLE item_aliases")

        # SQLite has no int-cast join shortcut for dict maps: build a temp
        # mapping table once and rewrite children with joins, which stays
        # fast at any store size.
        conn.execute("CREATE TEMP TABLE idmap (old TEXT PRIMARY KEY, new INTEGER)")
        conn.executemany("INSERT INTO idmap VALUES (?,?)", id_map.items())
        # alias redirects too: a dedup can reference a track that was
        # merged away in v2, reachable only through its alias row
        conn.executemany("INSERT OR IGNORE INTO idmap VALUES (?,?)", alias_map.items())
        # dedup values that were v2 item ids (lastfm artist events carry
        # the originating track id) follow the items to their new ints —
        # otherwise the first post-upgrade sync re-emits that history
        # under the new dedup format and every loved row doubles.
        conn.execute(
            "INSERT OR IGNORE INTO events (id, profile, ts, item_id, kind, weight,"
            " source, dedup, meta)"
            " SELECT e.id, e.profile, e.ts, m.new, e.kind, e.weight, e.source,"
            "        COALESCE(CAST(d.new AS TEXT), e.dedup), e.meta"
            " FROM events_v2 e JOIN idmap m ON m.old = e.item_id"
            " LEFT JOIN idmap d ON d.old = e.dedup")
        conn.execute(
            "INSERT OR IGNORE INTO candidates (profile, item_id, source, seed_item_id,"
            " external_score, first_seen, last_seen, meta)"
            " SELECT c.profile, m.new, c.source, ms.new, c.external_score,"
            "        c.first_seen, c.last_seen, c.meta"
            " FROM candidates_v2 c JOIN idmap m ON m.old = c.item_id"
            " LEFT JOIN idmap ms ON ms.old = c.seed_item_id")
        # approved rows first: when a union folds two open recommendations
        # onto one item, the partial unique index keeps the first — the
        # user's verdict must be the survivor, not the newer proposal.
        conn.execute(
            "INSERT OR IGNORE INTO recommendations (id, profile, run_id, ts, domain,"
            " item_id, score, why, status, acted_at)"
            " SELECT r.id, r.profile, r.run_id, r.ts, r.domain, m.new, r.score,"
            "        r.why, r.status, r.acted_at"
            " FROM recommendations_v2 r JOIN idmap m ON m.old = r.item_id"
            " ORDER BY CASE r.status WHEN 'approved' THEN 0 ELSE 1 END, r.id")
        conn.execute(
            "INSERT OR IGNORE INTO library (item_id, arr, arr_id, status, added_at, meta)"
            " SELECT m.new, l.arr, l.arr_id, l.status, l.added_at, l.meta"
            " FROM library_v2 l JOIN idmap m ON m.old = l.item_id")
        conn.execute(
            "INSERT OR IGNORE INTO embeddings (item_id, model, dim, vec, created_at)"
            " SELECT m.new, e.model, e.dim, e.vec, e.created_at"
            " FROM embeddings_v2 e JOIN idmap m ON m.old = e.item_id")
        conn.execute("DROP TABLE idmap")

        # taste models store exemplar item ids as learned; the TEXT ids
        # mean nothing now — drop them and let the next train rebuild.
        conn.execute(
            "DELETE FROM state WHERE key LIKE 'p:%:model:%' OR key LIKE 'p:%:centroid:%'")
        # v3 keys series cursors by Jellyfin id (merge-proof); carry the
        # counted progress over so the first post-upgrade sync neither
        # re-emits history nor swallows genuinely new plays.
        for row in conn.execute(
                "SELECT key, value FROM state"
                " WHERE key LIKE 'p:%:jellyfin:series_played:%'").fetchall():
            prefix, _, old_tail = row["key"].partition(":jellyfin:series_played:")
            new_item = id_map.get(old_tail)
            jf = None
            if new_item is not None:
                r = conn.execute(
                    "SELECT key FROM identities WHERE ns='jellyfin' AND item_id=?"
                    " ORDER BY rowid", (new_item,)).fetchone()
                jf = r["key"] if r else None
            conn.execute("DELETE FROM state WHERE key=?", (row["key"],))
            if jf:
                conn.execute(
                    "INSERT OR REPLACE INTO state (key, value) VALUES (?,?)",
                    (f"{prefix}:jellyfin:series_played:{jf}", row["value"]))
        # v3 derives series completeness from event existence, not state
        conn.execute("DELETE FROM state WHERE key LIKE 'p:%:jellyfin:series_complete:%'")

        for t in ("events_v2", "candidates_v2", "recommendations_v2",
                  "library_v2", "embeddings_v2", "items_v2"):
            conn.execute(f"DROP TABLE {t}")

    _atomic_migration(conn, work)


# ── identity resolution: THE write path ─────────────────────────────


def lookup_item(conn: sqlite3.Connection, domain: str, ns: str, key: str) -> int | None:
    row = conn.execute(
        "SELECT item_id FROM identities WHERE domain=? AND ns=? AND key=?",
        (domain, ns, ids_mod.normalize_key(str(key)))).fetchone()
    return row["item_id"] if row else None


def resolve_item(
    conn: sqlite3.Connection,
    domain: str,
    ns: str,
    key: str,
    title: str | None = None,
    year: int | None = None,
    meta: dict[str, Any] | None = None,
) -> int:
    """The single write-path entry: any spelling or external id seen before
    lands on the same item; unseen ones create it. Title/year/meta are
    applied non-destructively either way. A key that normalizes to nothing
    is refused — a shared '' identity would fuse unrelated items."""
    if not ids_mod.normalize_key(str(key)):
        raise ValueError(f"empty identity key for {domain}:{ns}")
    item_id = lookup_item(conn, domain, ns, key)
    if item_id is None:
        ts = now()
        cur = conn.execute(
            "INSERT INTO items (domain, title, year, meta, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (domain, title, year, json.dumps(meta or {}), ts, ts))
        item_id = cur.lastrowid
        conn.execute(
            "INSERT INTO identities (domain, ns, key, item_id) VALUES (?,?,?,?)",
            (domain, ns, ids_mod.normalize_key(str(key)), item_id))
        return item_id
    if title is not None or year is not None or meta:
        upsert_item_fields(conn, item_id, title=title, year=year, meta=meta)
    return item_id


def attach_identity(conn: sqlite3.Connection, item_id: int, ns: str, key: str) -> int:
    """Teach an item another of its names. When the identity already points
    at a DIFFERENT item, they merge — *if* the collision proves they are one
    entity — and the surviving id is returned (callers must continue with it).

    A collision on an authoritative key (tmdb/tvdb/imdb/mbid) is proof:
    those namespaces map one key to one real-world entity. A collision on a
    'name' key is not — MusicBrainz alias lists legitimately carry OTHER
    entities' names (former band names, personas, tributes), so when both
    items hold their own authoritative identity a name collision proves
    they are DIFFERENT, and the attach is refused: the key stays with its
    current owner and item_id comes back unchanged. Distinguish refusal
    from no-op with lookup_item when it matters (conflict stats).

    A 'jellyfin' key is a pointer to a library entry, not an immutable
    identity: when both items hold their own authoritative ids, the entry
    was re-identified in Jellyfin (a retag) and the pointer MOVES to the
    item the entry now resolves to — old events stay where they were
    earned, new ones follow the corrected identification."""
    domain = conn.execute(
        "SELECT domain FROM items WHERE id=?", (item_id,)).fetchone()["domain"]
    norm = ids_mod.normalize_key(str(key))
    if not norm:
        raise ValueError(f"empty identity key for {domain}:{ns}")
    row = conn.execute(
        "SELECT item_id FROM identities WHERE domain=? AND ns=? AND key=?",
        (domain, ns, norm)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO identities (domain, ns, key, item_id) VALUES (?,?,?,?)",
            (domain, ns, norm, item_id))
        return item_id
    other = row["item_id"]
    if other == item_id:
        return item_id
    if ns in ("name", "jellyfin") \
            and _authority(conn, other) > 0 and _authority(conn, item_id) > 0:
        if ns == "jellyfin":
            conn.execute(
                "UPDATE identities SET item_id=? WHERE domain=? AND ns=? AND key=?",
                (item_id, domain, ns, norm))
        return item_id
    # Authoritative-ns holders win so external references stay stable:
    # merging a name-only item into an mbid item, not the reverse.
    winner, loser = (other, item_id) if _authority(conn, other) >= _authority(conn, item_id) \
        else (item_id, other)
    merge_items(conn, loser, winner)
    return winner


def _authority(conn: sqlite3.Connection, item_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM identities WHERE item_id=? AND ns != 'name'"
        " AND ns != 'jellyfin'", (item_id,)).fetchone()
    return row["n"]


def identities_of(conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
    """First key per namespace (insertion order) — what actuation needs
    (one tvdb id, one mbid). All spellings live under ns 'name'; use a
    direct query when every alias is wanted."""
    out: dict[str, str] = {}
    for r in conn.execute(
            "SELECT ns, key FROM identities WHERE item_id=? ORDER BY rowid", (item_id,)):
        out.setdefault(r["ns"], r["key"])
    return out


def merge_items(conn: sqlite3.Connection, loser: int, winner: int) -> None:
    """Two items discovered to be the same entity: repoint everything at
    the winner. Events may collide on the uniqueness key (the same play
    recorded under both spellings) — collisions are genuine duplicates
    and are dropped."""
    if loser == winner:
        return
    conn.execute("UPDATE identities SET item_id=? WHERE item_id=?", (winner, loser))
    conn.execute("UPDATE OR IGNORE events SET item_id=? WHERE item_id=?", (winner, loser))
    conn.execute("DELETE FROM events WHERE item_id=?", (loser,))
    conn.execute("UPDATE OR IGNORE candidates SET item_id=? WHERE item_id=?", (winner, loser))
    conn.execute("DELETE FROM candidates WHERE item_id=?", (loser,))
    conn.execute("UPDATE candidates SET seed_item_id=? WHERE seed_item_id=?", (winner, loser))
    # When both sides hold an open recommendation, the user's verdict
    # outranks a standing proposal: clear the winner's proposed row so the
    # loser's approved one carries over instead of being IGNOREd away.
    conn.execute(
        "DELETE FROM recommendations WHERE item_id=? AND status='proposed' AND profile IN"
        " (SELECT profile FROM recommendations WHERE item_id=? AND status='approved')",
        (winner, loser))
    conn.execute("UPDATE OR IGNORE recommendations SET item_id=? WHERE item_id=?",
                 (winner, loser))
    conn.execute("DELETE FROM recommendations WHERE item_id=?", (loser,))
    conn.execute("UPDATE OR IGNORE library SET item_id=? WHERE item_id=?", (winner, loser))
    conn.execute("DELETE FROM library WHERE item_id=?", (loser,))
    conn.execute("UPDATE OR IGNORE embeddings SET item_id=? WHERE item_id=?", (winner, loser))
    conn.execute("DELETE FROM embeddings WHERE item_id=?", (loser,))
    # Merge metadata non-destructively (winner's keys win), then drop.
    row = conn.execute("SELECT * FROM items WHERE id=?", (loser,)).fetchone()
    if row is not None:
        w = conn.execute("SELECT year FROM items WHERE id=?", (winner,)).fetchone()
        upsert_item_fields(conn, winner, title=None,
                           year=row["year"] if w and w["year"] is None else None,
                           meta=json.loads(row["meta"]), prefer_existing=True)
        conn.execute("DELETE FROM items WHERE id=?", (loser,))


def upsert_item_fields(
    conn: sqlite3.Connection,
    item_id: int,
    title: str | None = None,
    year: int | None = None,
    meta: dict[str, Any] | None = None,
    enriched: bool = False,
    prefer_existing: bool = False,
) -> None:
    """Non-destructive field update: never blanks a field, merges meta
    key-wise. prefer_existing flips the meta merge for merge_items, where
    the winner's metadata must survive the loser's."""
    ts = now()
    row = conn.execute("SELECT meta FROM items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        return
    old = json.loads(row["meta"])
    merged = {**(meta or {}), **old} if prefer_existing else {**old, **(meta or {})}
    conn.execute(
        "UPDATE items SET title=COALESCE(?, title), year=COALESCE(?, year), meta=?,"
        " enriched_at=CASE WHEN ? THEN ? ELSE enriched_at END, updated_at=?"
        " WHERE id=?",
        (title, year, json.dumps(merged), enriched, ts, ts, item_id))


# ── events ───────────────────────────────────────────────────────────


def add_event(
    conn: sqlite3.Connection,
    ts: str,
    item_id: int,
    kind: str,
    weight: float,
    source: str,
    meta: dict[str, Any] | None = None,
    dedup: str = "",
    profile: str = "default",
) -> bool:
    """Returns False when the event already existed (idempotent re-sync)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO events (profile, ts, item_id, kind, weight, source, dedup, meta)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (profile, ts, item_id, kind, weight, source, dedup, json.dumps(meta or {})),
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


# Profile-scoped state (sync cursors, taste models): one flat namespace
# per profile so two profiles' Last.fm cursors or centroids never collide.
def pkey(profile: str, key: str) -> str:
    return f"p:{profile}:{key}"


def pget_state(conn: sqlite3.Connection, profile: str, key: str,
               default: str | None = None) -> str | None:
    return get_state(conn, pkey(profile, key), default)


def pset_state(conn: sqlite3.Connection, profile: str, key: str, value: str) -> None:
    set_state(conn, pkey(profile, key), value)


# ── embeddings ───────────────────────────────────────────────────────


def put_embedding(conn, item_id: int, model: str, vec: bytes, dim: int) -> None:
    conn.execute(
        "INSERT INTO embeddings (item_id, model, dim, vec, created_at) VALUES (?,?,?,?,?)"
        " ON CONFLICT(item_id, model) DO UPDATE SET vec=excluded.vec, dim=excluded.dim,"
        " created_at=excluded.created_at",
        (item_id, model, dim, vec, now()),
    )


def iter_embeddings(conn, model: str, item_ids: Iterable[int] | None = None):
    """Yields (item_id, dim, vec_bytes)."""
    if item_ids is None:
        yield from conn.execute(
            "SELECT item_id, dim, vec FROM embeddings WHERE model=?", (model,))
        return
    id_list = list(item_ids)
    for i in range(0, len(id_list), 500):
        chunk = id_list[i : i + 500]
        marks = ",".join("?" * len(chunk))
        yield from conn.execute(
            f"SELECT item_id, dim, vec FROM embeddings WHERE model=? AND item_id IN ({marks})",
            (model, *chunk),
        )
