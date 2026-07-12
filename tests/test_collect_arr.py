"""Offline tests for the *arr inventory collector."""

from __future__ import annotations

import json

import pytest

from gustarr import config as C
from gustarr import db, http
from gustarr.collect import arr
from gustarr.signals import WEIGHTS

RADARR = "http://radarr:7878"
SONARR = "http://sonarr:8989"
LIDARR = "http://lidarr:8686"

MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"


def make_responses():
    return {
        f"{RADARR}/api/v3/movie": [
            {"id": 11, "title": "The Matrix", "year": 1999, "tmdbId": 603,
             "imdbId": "tt0133093", "genres": ["Action", "Science Fiction"],
             "monitored": True, "added": "2024-01-05T10:00:00Z", "tags": []},
            {"id": 12, "title": "Heat", "year": 1995, "tmdbId": 949,
             "imdbId": "tt0113277", "genres": ["Crime"],
             "monitored": True, "added": "2024-02-01T09:00:00Z", "tags": [5]},
        ],
        f"{RADARR}/api/v3/tag": [{"id": 1, "label": "other"}, {"id": 5, "label": "gustarr"}],
        f"{SONARR}/api/v3/series": [
            {"id": 21, "title": "The Wire", "year": 2002, "tvdbId": 79126,
             "imdbId": "tt0306414", "genres": ["Crime", "Drama"],
             "monitored": True, "added": "2024-03-01T08:00:00Z", "tags": []},
        ],
        f"{SONARR}/api/v3/tag": [{"id": 3, "label": "gustarr"}],
        f"{LIDARR}/api/v1/artist": [
            {"id": 31, "artistName": "Radiohead", "foreignArtistId": MBID,
             "genres": ["Alternative Rock"], "monitored": False,
             "added": "2024-04-01T07:00:00Z", "tags": []},
        ],
        f"{LIDARR}/api/v1/tag": [{"id": 9, "label": "gustarr"}],
    }


@pytest.fixture
def responses():
    return make_responses()


@pytest.fixture
def fake_http(monkeypatch, responses):
    calls = []

    def fake_request_json(method, url, *, headers=None, **kw):
        calls.append((method, url, headers))
        assert method == "GET"
        assert headers["X-Api-Key"] in {"rk", "sk", "lk"}
        return responses[url]

    monkeypatch.setattr(http, "request_json", fake_request_json)
    monkeypatch.setattr(http, "HOST_DELAYS", {})
    return calls


@pytest.fixture
def cfg(tmp_path):
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "radarr": {"url": RADARR, "api_key": "rk"},
        "sonarr": {"url": SONARR, "api_key": "sk"},
        "lidarr": {"url": LIDARR, "api_key": "lk"},
    })


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def matrix(conn):
    return db.lookup_item(conn, "movie", "tmdb", "603")


def heat(conn):
    return db.lookup_item(conn, "movie", "tmdb", "949")


def test_first_sync_items_library_events(conn, cfg, fake_http):
    stats = arr.sync(conn, cfg)
    assert stats["items"] == 4
    assert stats["library"] == 4
    assert stats["events"] == 3  # Heat is gustarr-tagged: no library_add
    assert stats["rejects"] == 0
    assert stats["removed"] == 0

    row = conn.execute("SELECT * FROM items WHERE id=?", (matrix(conn),)).fetchone()
    assert row["title"] == "The Matrix" and row["year"] == 1999
    # the canonical id resolves the item, secondary ids become identities
    assert db.identities_of(conn, matrix(conn)) == {"tmdb": "603", "imdb": "tt0133093"}
    assert json.loads(row["meta"])["genres"] == ["Action", "Science Fiction"]
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 4
    wire = db.lookup_item(conn, "series", "tvdb", "79126")
    assert conn.execute("SELECT id FROM items WHERE domain='series'").fetchone()[0] == wire
    assert db.lookup_item(conn, "series", "imdb", "tt0306414") == wire

    lib = conn.execute("SELECT * FROM library WHERE item_id=?", (matrix(conn),)).fetchone()
    assert lib["arr"] == "radarr" and lib["arr_id"] == 11
    assert lib["status"] == "monitored" and lib["added_at"] == "2024-01-05T10:00:00Z"
    assert json.loads(lib["meta"]) == {}

    tagged = conn.execute("SELECT * FROM library WHERE item_id=?", (heat(conn),)).fetchone()
    assert json.loads(tagged["meta"]) == {"gustarr": True}

    radiohead = db.lookup_item(conn, "artist", "mbid", MBID)
    artist_lib = conn.execute("SELECT * FROM library WHERE arr='lidarr'").fetchone()
    assert artist_lib["item_id"] == radiohead
    assert artist_lib["status"] == "unmonitored"

    ev = conn.execute("SELECT * FROM events WHERE item_id=?", (matrix(conn),)).fetchone()
    assert ev["kind"] == "library_add" and ev["source"] == "arr"
    assert ev["ts"] == "2024-01-05T10:00:00Z"
    assert ev["weight"] == WEIGHTS["library_add"]
    # legacy single-user config: the fan-out is exactly the default profile
    assert ev["profile"] == "default"
    # the gustarr-tagged add produced no event
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE item_id=?", (heat(conn),)).fetchone()[0] == 0

    # known state keys on the *arr's external ids — merge-proof, no item ids
    known = json.loads(db.get_state(conn, "arr:known:radarr"))
    assert known == {"603": 11, "949": 12}
    assert json.loads(db.get_state(conn, "arr:known:lidarr")) == {MBID: 31}


def test_resync_is_idempotent(conn, cfg, fake_http):
    arr.sync(conn, cfg)
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    stats = arr.sync(conn, cfg)
    assert stats["events"] == 0 and stats["rejects"] == 0 and stats["removed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == n_events
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 4


def test_deleted_gustarr_item_becomes_reject(conn, cfg, fake_http, responses):
    arr.sync(conn, cfg)
    heat_id = heat(conn)
    # user deletes both radarr movies: only the gustarr-tagged one is a signal
    responses[f"{RADARR}/api/v3/movie"] = []
    stats = arr.sync(conn, cfg)
    assert stats["rejects"] == 1
    assert stats["removed"] == 2

    rejects = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchall()
    assert len(rejects) == 1
    rej = rejects[0]
    assert rej["item_id"] == heat_id
    # no recommendations row owns the add: damped household evidence
    assert rej["weight"] == WEIGHTS["reject"] * 0.3
    assert rej["source"] == "arr"
    assert json.loads(rej["meta"]) == {"deleted": True, "shared": True}

    assert conn.execute("SELECT COUNT(*) FROM library WHERE arr='radarr'").fetchone()[0] == 0
    assert json.loads(db.get_state(conn, "arr:known:radarr")) == {}
    # untouched arrs keep their rows and produce nothing new
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 2


def test_v2_known_state_migrates_on_read(conn, cfg, fake_http):
    """A store upgraded from v2 still holds text item ids in arr:known; the
    first sync must read them as the same library instead of mass-removing
    (and mass-rejecting) everything once."""
    arr.sync(conn, cfg)
    db.set_state(conn, "arr:known:radarr",
                 json.dumps({"movie:tmdb:603": 11, "movie:tmdb:949": 12}))
    db.set_state(conn, "arr:known:lidarr", json.dumps({f"artist:mbid:{MBID}": 31}))

    stats = arr.sync(conn, cfg)

    assert stats["rejects"] == 0 and stats["removed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 4
    # the state is rewritten in the new external-id format
    assert json.loads(db.get_state(conn, "arr:known:radarr")) == {"603": 11, "949": 12}
    assert json.loads(db.get_state(conn, "arr:known:lidarr")) == {MBID: 31}


def test_lidarr_junk_foreign_id_skips_row_not_stage(conn, cfg, fake_http, responses):
    """A lidarr entry whose foreignArtistId is whitespace-only folds to an
    empty key: the row is counted skipped, the stage completes, and the
    entries after it still land."""
    responses[f"{LIDARR}/api/v1/artist"].insert(0, {
        "id": 30, "artistName": "Ghost Entry", "foreignArtistId": "   ",
        "genres": [], "monitored": True, "added": "2024-04-02T07:00:00Z", "tags": []})

    stats = arr.sync(conn, cfg)

    assert stats["skipped"] == 1
    assert stats["items"] == 4 and stats["library"] == 4
    assert stats["events"] == 3  # Heat is gustarr-tagged: no library_add
    # the junk row minted nothing; Radiohead (listed after it) still landed
    assert db.lookup_item(conn, "artist", "mbid", MBID) is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM items WHERE domain='artist'").fetchone()[0] == 1
    # the skipped row never entered the known state, so the next sync
    # neither removes nor rejects anything on its account
    assert json.loads(db.get_state(conn, "arr:known:lidarr")) == {MBID: 31}
    stats2 = arr.sync(conn, cfg)
    assert stats2["skipped"] == 1
    assert stats2["removed"] == 0 and stats2["rejects"] == 0


def test_unconfigured_arrs_skipped(conn, tmp_path, monkeypatch, responses):
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "radarr": {"url": RADARR, "api_key": "rk"}})
    urls = []

    def fake_request_json(method, url, **kw):
        urls.append(url)
        return responses[url]

    monkeypatch.setattr(http, "request_json", fake_request_json)
    stats = arr.sync(conn, cfg)
    assert stats["items"] == 2
    assert urls and all(u.startswith(RADARR) for u in urls)


def test_events_fan_out_to_every_profile(conn, tmp_path, fake_http, responses):
    # the *arr can't say who added or deleted — the whole household gets
    # the (modest) signal, per the profile contract
    cfg = C._build({
        "core": {"data_dir": str(tmp_path)},
        "radarr": {"url": RADARR, "api_key": "rk"},
        "profiles": {"alice": {"lastfm_user": "a"}, "bob": {}},
    })
    stats = arr.sync(conn, cfg)

    assert stats["events"] == 2  # one library_add per profile (Heat is tagged)
    adds = {r["profile"] for r in conn.execute(
        "SELECT profile FROM events WHERE kind='library_add' AND item_id=?",
        (matrix(conn),))}
    assert adds == {"alice", "bob"}
    # library stays global: one row, no profile dimension
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 2

    # deleting the gustarr-tagged movie with no rec on record falls back
    # to the household fan-out — damped and flagged shared, since it says
    # little about any one person's taste
    heat_id = heat(conn)
    responses[f"{RADARR}/api/v3/movie"] = []
    stats = arr.sync(conn, cfg)
    assert stats["rejects"] == 2
    rejects = {(r["profile"], r["item_id"]) for r in conn.execute(
        "SELECT profile, item_id FROM events WHERE kind='reject'")}
    assert rejects == {("alice", heat_id), ("bob", heat_id)}
    for r in conn.execute("SELECT weight, meta FROM events WHERE kind='reject'"):
        assert r["weight"] == WEIGHTS["reject"] * 0.3
        assert json.loads(r["meta"]) == {"deleted": True, "shared": True}


def test_deleted_gustarr_item_reject_hits_owning_profile_only(
        conn, tmp_path, fake_http, responses):
    """The rec that landed the add names whose taste the deletion judges:
    the reject must hit that profile alone, at full weight, newest owner
    winning when the item was recommended more than once."""
    cfg = C._build({
        "core": {"data_dir": str(tmp_path)},
        "radarr": {"url": RADARR, "api_key": "rk"},
        "profiles": {"alice": {}, "bob": {}},
    })
    arr.sync(conn, cfg)
    heat_id = heat(conn)

    def add_rec(profile, status, acted_at):
        conn.execute(
            "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score,"
            " status, acted_at) VALUES (?,?,?,?,?,?,?,?)",
            (profile, "test", acted_at, "movie", heat_id, 0.5, status, acted_at))

    add_rec("bob", "added", "2024-01-01T00:00:00Z")  # superseded by alice's newer add
    add_rec("alice", "auto_added", "2024-06-01T00:00:00Z")

    responses[f"{RADARR}/api/v3/movie"] = []
    stats = arr.sync(conn, cfg)

    assert stats["rejects"] == 1
    rejects = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchall()
    assert len(rejects) == 1
    rej = rejects[0]
    assert rej["item_id"] == heat_id
    assert rej["profile"] == "alice"  # the newest owner, nobody else
    assert rej["weight"] == WEIGHTS["reject"]  # her own add: full strength
    assert json.loads(rej["meta"]) == {"deleted": True}
