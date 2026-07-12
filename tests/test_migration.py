"""Regression tests for the store migrations (gustarr.db).

v1/v2 stores are built by hand with raw SQL in their historical shapes —
the current write path can no longer produce them, which is the point —
then opened through db.connect(), the only migration entry. Everything
runs offline against tmp_path databases.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from gustarr import db

MBID = "53f7a13b-0f75-4b31-9c30-2ed40de6dc35"
OTHER_MBID = "b0b2c1ff-1111-4222-8333-444455556666"
TRACK_MBID = "9c3fe4a1-aaaa-4bbb-8ccc-ddddeeeeffff"

# A key a pre-normalization gustarr stored verbatim from a full-width
# Last.fm spelling with a stray trailing space.
RAW_KEY = "ＹＯＡＳＯＢＩ "

T0 = "2024-01-01T00:00:00Z"
T1 = "2024-02-01T00:00:00Z"
T2 = "2024-03-01T00:00:00Z"

# The v2 shape: TEXT item ids ("domain:ns:key"), external ids in an
# items.ids JSON column, alias redirects in item_aliases.
V2_SCHEMA = """
CREATE TABLE items (
  id          TEXT PRIMARY KEY,
  domain      TEXT NOT NULL,
  title       TEXT,
  year        INTEGER,
  ids         TEXT NOT NULL DEFAULT '{}',
  meta        TEXT NOT NULL DEFAULT '{}',
  enriched_at TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX idx_items_domain ON items(domain);
CREATE TABLE events (
  id      INTEGER PRIMARY KEY,
  profile TEXT NOT NULL DEFAULT 'default',
  ts      TEXT NOT NULL,
  item_id TEXT NOT NULL,
  kind    TEXT NOT NULL,
  weight  REAL NOT NULL,
  source  TEXT NOT NULL,
  dedup   TEXT NOT NULL DEFAULT '',
  meta    TEXT NOT NULL DEFAULT '{}',
  UNIQUE(profile, ts, item_id, kind, source, dedup)
);
CREATE INDEX idx_events_item ON events(profile, item_id);
CREATE TABLE candidates (
  profile        TEXT NOT NULL DEFAULT 'default',
  item_id        TEXT NOT NULL,
  source         TEXT NOT NULL,
  seed_item_id   TEXT,
  external_score REAL,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  meta           TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (profile, item_id, source)
);
CREATE TABLE recommendations (
  id       INTEGER PRIMARY KEY,
  profile  TEXT NOT NULL DEFAULT 'default',
  run_id   TEXT NOT NULL,
  ts       TEXT NOT NULL,
  domain   TEXT NOT NULL,
  item_id  TEXT NOT NULL,
  score    REAL NOT NULL,
  why      TEXT NOT NULL DEFAULT '{}',
  status   TEXT NOT NULL DEFAULT 'proposed',
  acted_at TEXT
);
CREATE INDEX idx_recs_status ON recommendations(profile, status);
CREATE UNIQUE INDEX idx_recs_open_item
  ON recommendations(profile, item_id) WHERE status IN ('proposed', 'approved');
CREATE TABLE library (
  item_id  TEXT PRIMARY KEY,
  arr      TEXT NOT NULL,
  arr_id   INTEGER,
  status   TEXT,
  added_at TEXT,
  meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE embeddings (
  item_id    TEXT NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, model)
);
CREATE TABLE item_aliases (
  alias_id     TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL
);
CREATE TABLE state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

# The v1 shape: v2 minus the profile columns, minus events.dedup, with
# unprefixed profile state keys (model:, centroid:, lastfm:, jellyfin:).
V1_SCHEMA = """
CREATE TABLE items (
  id          TEXT PRIMARY KEY,
  domain      TEXT NOT NULL,
  title       TEXT,
  year        INTEGER,
  ids         TEXT NOT NULL DEFAULT '{}',
  meta        TEXT NOT NULL DEFAULT '{}',
  enriched_at TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE TABLE events (
  id      INTEGER PRIMARY KEY,
  ts      TEXT NOT NULL,
  item_id TEXT NOT NULL,
  kind    TEXT NOT NULL,
  weight  REAL NOT NULL,
  source  TEXT NOT NULL,
  meta    TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE candidates (
  item_id        TEXT NOT NULL,
  source         TEXT NOT NULL,
  seed_item_id   TEXT,
  external_score REAL,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  meta           TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (item_id, source)
);
CREATE TABLE recommendations (
  id       INTEGER PRIMARY KEY,
  run_id   TEXT NOT NULL,
  ts       TEXT NOT NULL,
  domain   TEXT NOT NULL,
  item_id  TEXT NOT NULL,
  score    REAL NOT NULL,
  why      TEXT NOT NULL DEFAULT '{}',
  status   TEXT NOT NULL DEFAULT 'proposed',
  acted_at TEXT
);
CREATE TABLE library (
  item_id  TEXT PRIMARY KEY,
  arr      TEXT NOT NULL,
  arr_id   INTEGER,
  status   TEXT,
  added_at TEXT,
  meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE embeddings (
  item_id    TEXT NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, model)
);
CREATE TABLE state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def _open_raw(path, script):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(script)
    return conn


def v2_store(path):
    conn = _open_raw(path, V2_SCHEMA)
    conn.execute("INSERT INTO state VALUES ('schema_version', '2')")
    return conn


def v2_item(conn, item_id, domain, title=None, year=None, ids=None, meta=None, created=T0):
    conn.execute(
        "INSERT INTO items (id, domain, title, year, ids, meta, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (item_id, domain, title, year, json.dumps(ids or {}), json.dumps(meta or {}),
         created, created))


def v2_event(conn, item_id, ts, kind="scrobble", weight=1.0, source="lastfm", dedup=""):
    conn.execute(
        "INSERT INTO events (profile, ts, item_id, kind, weight, source, dedup, meta)"
        " VALUES ('default',?,?,?,?,?,?,'{}')",
        (ts, item_id, kind, weight, source, dedup))


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _dump(conn):
    return {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}"))
            for t in ("items", "identities", "events", "candidates", "recommendations",
                      "library", "embeddings", "state")}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


def test_failed_migration_rolls_back_to_v2_and_retries(db_path, monkeypatch):
    """A crash anywhere inside the v2→v3 migration must leave the v2 store
    exactly as it was (transactional DDL, no *_v2 leftovers), and the next
    open must run the migration to completion with every row intact."""
    raw = v2_store(db_path)
    v2_item(raw, f"artist:mbid:{MBID}", "artist", title="Yorushika",
            ids={"mbid": MBID, "lastfm": "Yorushika"})
    v2_event(raw, f"artist:mbid:{MBID}", T0)
    v2_event(raw, f"artist:mbid:{MBID}", T1, kind="loved", weight=4.0)
    raw.commit()
    raw.close()

    class Boom(Exception):
        pass

    def explode(conn):
        raise Boom("crash after the tables were renamed")

    with monkeypatch.context() as m:
        m.setattr(db, "_exec_schema", explode)
        with pytest.raises(Boom):
            db.connect(db_path)

    raw = sqlite3.connect(db_path)
    raw.row_factory = sqlite3.Row
    tables = _tables(raw)
    assert "items" in tables and "identities" not in tables
    assert not any(t.endswith("_v2") for t in tables)
    cols = {r["name"] for r in raw.execute("PRAGMA table_info(items)")}
    assert "ids" in cols  # still the v2 shape, not misread as healthy v3
    assert raw.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    raw.close()

    conn = db.connect(db_path)  # unpatched retry completes the migration
    item = db.lookup_item(conn, "artist", "mbid", MBID)
    assert item is not None
    rows = conn.execute("SELECT item_id, kind FROM events ORDER BY ts").fetchall()
    assert [(r["item_id"], r["kind"]) for r in rows] == [(item, "scrobble"), (item, "loved")]
    assert db.get_state(conn, "schema_version") == "3"
    assert not any(t.endswith("_v2") for t in _tables(conn))
    conn.close()


def test_leftover_items_v2_table_refuses_with_backup_hint(db_path):
    """items_v2 lying around means a half-finished migration from an engine
    without transactional DDL — connect must refuse loudly, not guess."""
    raw = v2_store(db_path)
    v2_item(raw, f"artist:mbid:{MBID}", "artist")
    raw.execute("CREATE TABLE items_v2 (id TEXT PRIMARY KEY)")
    raw.commit()
    raw.close()

    with pytest.raises(RuntimeError, match="backup"):
        db.connect(db_path)


def test_shared_mbid_unions_two_v2_items(db_path):
    """Two v2 items claiming the same mbid are one entity: one v3 item holds
    both items' other identities, both event histories (identical rows
    deduped), both embeddings, merged candidates/library, alias redirects."""
    a, b = f"artist:mbid:{MBID}", "artist:lastfm:ヨルシカ"
    raw = v2_store(db_path)
    v2_item(raw, a, "artist", title="Yorushika",
            ids={"mbid": MBID, "lastfm": "Yorushika"}, meta={"genres": ["rock"]}, created=T0)
    v2_item(raw, b, "artist", title="ヨルシカ", ids={"mbid": MBID},
            meta={"genres": ["j-pop"], "country": "JP"}, created=T1)
    v2_event(raw, a, T0)
    v2_event(raw, b, T1)
    v2_event(raw, a, T2, kind="loved", weight=4.0)
    v2_event(raw, b, T2, kind="loved", weight=4.0)  # same love seen under both spellings
    raw.execute(
        "INSERT INTO candidates (profile, item_id, source, seed_item_id, external_score,"
        " first_seen, last_seen, meta) VALUES ('default',?,'lastfm_similar',?,0.9,?,?,'{}')",
        (a, b, T0, T1))
    raw.execute(
        "INSERT INTO candidates (profile, item_id, source, seed_item_id, external_score,"
        " first_seen, last_seen, meta) VALUES ('default',?,'tmdb_similar',NULL,0.5,?,?,'{}')",
        (b, T0, T1))
    raw.execute("INSERT INTO library VALUES (?, 'lidarr', 7, 'monitored', ?, '{}')", (a, T0))
    raw.execute("INSERT INTO embeddings VALUES (?, 'm-a', 1, X'003C', ?)", (a, T0))
    raw.execute("INSERT INTO embeddings VALUES (?, 'm-b', 1, X'003C', ?)", (b, T0))
    raw.execute("INSERT INTO item_aliases VALUES ('artist:lastfm:よるしか', ?)", (a,))
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    item = db.lookup_item(conn, "artist", "mbid", MBID)
    row = conn.execute("SELECT * FROM items WHERE id=?", (item,)).fetchone()
    assert row["title"] == "Yorushika"  # the authoritative member's fields win
    assert row["created_at"] == T0  # the group is as old as its oldest member
    assert json.loads(row["meta"]) == {"genres": ["rock"], "country": "JP"}

    got = {(r["ns"], r["key"]) for r in conn.execute(
        "SELECT ns, key FROM identities WHERE item_id=?", (item,))}
    assert got == {("mbid", MBID), ("name", "yorushika"),
                   ("name", "ヨルシカ"), ("name", "よるしか")}
    assert "item_aliases" not in _tables(conn)

    events = conn.execute(
        "SELECT kind, ts FROM events WHERE item_id=? ORDER BY ts", (item,)).fetchall()
    assert [(r["kind"], r["ts"]) for r in events] == [
        ("scrobble", T0), ("scrobble", T1), ("loved", T2)]  # twin loved rows deduped

    cands = conn.execute(
        "SELECT source, item_id, seed_item_id FROM candidates ORDER BY source").fetchall()
    assert [(r["source"], r["item_id"], r["seed_item_id"]) for r in cands] == [
        ("lastfm_similar", item, item), ("tmdb_similar", item, None)]
    assert [tuple(r) for r in conn.execute("SELECT item_id, arr FROM library")] == [
        (item, "lidarr")]
    embs = {r["model"] for r in conn.execute(
        "SELECT model FROM embeddings WHERE item_id=?", (item,))}
    assert embs == {"m-a", "m-b"}
    conn.close()


def test_name_collision_between_two_mbids_keeps_items_split(db_path):
    """Two artists with their own mbids sharing a Last.fm name key are
    proven DIFFERENT: two v3 items, the first writer keeps the name, the
    second stays reachable by its mbid, and histories stay separate."""
    a, b = f"artist:mbid:{MBID}", f"artist:mbid:{OTHER_MBID}"
    raw = v2_store(db_path)
    v2_item(raw, a, "artist", title="The Ravens (UK)",
            ids={"mbid": MBID, "lastfm": "The Ravens"}, created=T0)
    v2_item(raw, b, "artist", title="The Ravens (JP)",
            ids={"mbid": OTHER_MBID, "lastfm": "The Ravens"}, created=T1)
    v2_event(raw, a, T0)
    v2_event(raw, b, T1)
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    first = db.lookup_item(conn, "artist", "mbid", MBID)
    second = db.lookup_item(conn, "artist", "mbid", OTHER_MBID)
    assert first is not None and second is not None and first != second
    assert db.lookup_item(conn, "artist", "name", "The Ravens") == first
    ev = {r["ts"]: r["item_id"] for r in conn.execute("SELECT ts, item_id FROM events")}
    assert ev == {T0: first, T1: second}
    conn.close()


def test_pre_normalization_name_twins_heal_into_one(db_path):
    """Two name-only items an old gustarr minted for width/case variants of
    one spelling fold into a single item with the united history."""
    a, b = f"artist:lastfm:{RAW_KEY}", "artist:lastfm:yoasobi"
    raw = v2_store(db_path)
    v2_item(raw, a, "artist", title=RAW_KEY.strip(), ids={"lastfm": RAW_KEY}, created=T0)
    v2_item(raw, b, "artist", title="YOASOBI", ids={"lastfm": "yoasobi"}, created=T1)
    v2_event(raw, a, T0)
    v2_event(raw, b, T1)
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    item = db.lookup_item(conn, "artist", "name", "yoasobi")
    assert item is not None
    assert db.lookup_item(conn, "artist", "name", RAW_KEY) == item
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE item_id=?", (item,)).fetchone()[0] == 2
    conn.close()


def test_event_dedup_holding_an_item_id_follows_the_item(db_path):
    """dedup values that were v2 item ids (lastfm artist events carry the
    originating track id) are rewritten to the track's new integer id;
    opaque discriminators pass through untouched."""
    artist, track = f"artist:mbid:{MBID}", f"track:mbid:{TRACK_MBID}"
    raw = v2_store(db_path)
    v2_item(raw, artist, "artist", ids={"mbid": MBID}, created=T0)
    v2_item(raw, track, "track", ids={"mbid": TRACK_MBID}, created=T1)
    v2_event(raw, artist, T0, dedup=track)
    v2_event(raw, artist, T1, dedup="pbr42")
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    a = db.lookup_item(conn, "artist", "mbid", MBID)
    t = db.lookup_item(conn, "track", "mbid", TRACK_MBID)
    deds = {r["ts"]: r["dedup"] for r in conn.execute(
        "SELECT ts, dedup FROM events WHERE item_id=?", (a,))}
    assert deds == {T0: str(t), T1: "pbr42"}
    conn.close()


def test_union_keeps_the_approved_recommendation(db_path):
    """When a union folds an approved and a proposed recommendation onto one
    item, the user's verdict survives — even when the proposal was written
    first — and only one open row remains."""
    a, b = f"artist:mbid:{MBID}", "artist:lastfm:yorushika"
    raw = v2_store(db_path)
    v2_item(raw, a, "artist", ids={"mbid": MBID}, created=T0)
    v2_item(raw, b, "artist", ids={"mbid": MBID}, created=T1)
    raw.execute(
        "INSERT INTO recommendations (id, profile, run_id, ts, domain, item_id, score,"
        " why, status) VALUES (1, 'default', 'r1', ?, 'artist', ?, 0.5, '{}', 'proposed')",
        (T0, a))
    raw.execute(
        "INSERT INTO recommendations (id, profile, run_id, ts, domain, item_id, score,"
        " why, status) VALUES (2, 'default', 'r2', ?, 'artist', ?, 0.9, '{}', 'approved')",
        (T1, b))
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    item = db.lookup_item(conn, "artist", "mbid", MBID)
    rows = conn.execute("SELECT * FROM recommendations").fetchall()
    assert len(rows) == 1
    assert rows[0]["item_id"] == item
    assert rows[0]["status"] == "approved"
    conn.close()


def test_state_rewrites_series_cursors_and_drops_stale_models(db_path):
    """series_played cursors move from TEXT item ids to Jellyfin ids (minted
    from v2 meta), jellyfin-less cursors and series_complete flags drop,
    taste models drop for retraining, everything else survives."""
    s1, s2 = "series:tvdb:1234", "series:tvdb:5678"
    raw = v2_store(db_path)
    v2_item(raw, s1, "series", title="Frieren", ids={"tvdb": "1234"},
            meta={"jellyfin_id": "abc123"}, created=T0)
    v2_item(raw, s2, "series", title="Orphaned", ids={"tvdb": "5678"}, created=T1)
    for key, value in [
        (f"p:default:jellyfin:series_played:{s1}", "17"),
        (f"p:default:jellyfin:series_played:{s2}", "4"),
        (f"p:default:jellyfin:series_complete:{s1}", "1"),
        ("p:default:model:artist", "opaque-model-blob"),
        ("p:default:centroid:artist", "[0.25]"),
        ("p:default:lastfm:cursor", "1700000000"),
        ("arr:known:radarr", "{}"),
    ]:
        raw.execute("INSERT INTO state VALUES (?,?)", (key, value))
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    item = db.lookup_item(conn, "series", "tvdb", "1234")
    assert db.lookup_item(conn, "series", "jellyfin", "abc123") == item  # minted from meta
    assert db.get_state(conn, "p:default:jellyfin:series_played:abc123") == "17"
    played = [r["key"] for r in conn.execute(
        "SELECT key FROM state WHERE key LIKE '%series_played%'")]
    assert played == ["p:default:jellyfin:series_played:abc123"]  # s2's cursor dropped
    assert conn.execute(
        "SELECT COUNT(*) FROM state WHERE key LIKE '%series_complete%'").fetchone()[0] == 0
    assert db.get_state(conn, "p:default:model:artist") is None
    assert db.get_state(conn, "p:default:centroid:artist") is None
    assert db.get_state(conn, "p:default:lastfm:cursor") == "1700000000"
    assert db.get_state(conn, "arr:known:radarr") == "{}"
    conn.close()


def test_reopening_a_migrated_store_changes_nothing(db_path):
    """The migration must fire exactly once: a second connect() sees a v3
    store (no ids column, no items_v2) and leaves every table untouched."""
    raw = v2_store(db_path)
    v2_item(raw, f"artist:mbid:{MBID}", "artist",
            ids={"mbid": MBID, "lastfm": "Yorushika"}, created=T0)
    v2_item(raw, "artist:lastfm:ヨルシカ", "artist", ids={"mbid": MBID}, created=T1)
    v2_event(raw, f"artist:mbid:{MBID}", T0)
    raw.execute("INSERT INTO state VALUES ('p:default:lastfm:cursor', '1700000000')")
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    before = _dump(conn)
    conn.close()

    conn = db.connect(db_path)
    assert _dump(conn) == before
    conn.close()


def test_v1_store_migrates_along_the_full_chain(db_path):
    """A pre-profile store (no profile columns, no events.dedup, unprefixed
    state keys) upgrades v1→v2→v3 in one connect(): events gain
    profile/dedup and integer item ids, cursors get the profile prefix and
    then the v3 rewrites, models drop."""
    artist, series = f"artist:mbid:{MBID}", "series:tvdb:99"
    raw = _open_raw(db_path, V1_SCHEMA)
    raw.execute(
        "INSERT INTO items (id, domain, title, ids, meta, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (artist, "artist", "Yorushika",
         json.dumps({"mbid": MBID, "lastfm": "Yorushika"}), "{}", T0, T0))
    raw.execute(
        "INSERT INTO items (id, domain, title, ids, meta, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (series, "series", "Frieren", json.dumps({"tvdb": "99"}),
         json.dumps({"jellyfin_id": "jf99"}), T1, T1))
    raw.execute(
        "INSERT INTO events (ts, item_id, kind, weight, source, meta)"
        " VALUES (?,?,?,?,?,'{}')", (T0, artist, "scrobble", 1.0, "lastfm"))
    for key, value in [("lastfm:cursor", "123"), ("model:artist", "m"),
                       (f"jellyfin:series_played:{series}", "7"), ("arr:known:radarr", "{}")]:
        raw.execute("INSERT INTO state VALUES (?,?)", (key, value))
    raw.commit()
    raw.close()

    conn = db.connect(db_path)
    assert db.get_state(conn, "schema_version") == "3"
    a = db.lookup_item(conn, "artist", "mbid", MBID)
    assert db.lookup_item(conn, "artist", "name", "Yorushika") == a
    ev = conn.execute("SELECT profile, item_id, dedup FROM events").fetchall()
    assert [(r["profile"], r["item_id"], r["dedup"]) for r in ev] == [("default", a, "")]
    assert db.get_state(conn, "p:default:lastfm:cursor") == "123"
    assert db.get_state(conn, "lastfm:cursor") is None
    assert conn.execute(
        "SELECT COUNT(*) FROM state WHERE key LIKE '%model%'").fetchone()[0] == 0
    assert db.get_state(conn, "p:default:jellyfin:series_played:jf99") == "7"
    assert db.get_state(conn, "arr:known:radarr") == "{}"
    conn.close()
