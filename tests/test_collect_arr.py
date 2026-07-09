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


def test_first_sync_items_library_events(conn, cfg, fake_http):
    stats = arr.sync(conn, cfg)
    assert stats["items"] == 4
    assert stats["library"] == 4
    assert stats["events"] == 3  # Heat is gustarr-tagged: no library_add
    assert stats["rejects"] == 0
    assert stats["removed"] == 0

    row = conn.execute("SELECT * FROM items WHERE id='movie:tmdb:603'").fetchone()
    assert row["title"] == "The Matrix" and row["year"] == 1999
    assert json.loads(row["ids"]) == {"tmdb": 603, "imdb": "tt0133093"}
    assert json.loads(row["meta"])["genres"] == ["Action", "Science Fiction"]
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 4
    assert conn.execute("SELECT id FROM items WHERE domain='series'").fetchone()[0] \
        == "series:tvdb:79126"

    lib = conn.execute("SELECT * FROM library WHERE item_id='movie:tmdb:603'").fetchone()
    assert lib["arr"] == "radarr" and lib["arr_id"] == 11
    assert lib["status"] == "monitored" and lib["added_at"] == "2024-01-05T10:00:00Z"
    assert json.loads(lib["meta"]) == {}

    tagged = conn.execute("SELECT * FROM library WHERE item_id='movie:tmdb:949'").fetchone()
    assert json.loads(tagged["meta"]) == {"gustarr": True}

    artist_lib = conn.execute("SELECT * FROM library WHERE arr='lidarr'").fetchone()
    assert artist_lib["item_id"] == f"artist:mbid:{MBID}"
    assert artist_lib["status"] == "unmonitored"

    ev = conn.execute("SELECT * FROM events WHERE item_id='movie:tmdb:603'").fetchone()
    assert ev["kind"] == "library_add" and ev["source"] == "arr"
    assert ev["ts"] == "2024-01-05T10:00:00Z"
    assert ev["weight"] == WEIGHTS["library_add"]
    # the gustarr-tagged add produced no event
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE item_id='movie:tmdb:949'").fetchone()[0] == 0

    known = json.loads(db.get_state(conn, "arr:known:radarr"))
    assert known == {"movie:tmdb:603": 11, "movie:tmdb:949": 12}
    assert json.loads(db.get_state(conn, "arr:known:lidarr")) == {f"artist:mbid:{MBID}": 31}


def test_resync_is_idempotent(conn, cfg, fake_http):
    arr.sync(conn, cfg)
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    stats = arr.sync(conn, cfg)
    assert stats["events"] == 0 and stats["rejects"] == 0 and stats["removed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == n_events
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 4


def test_deleted_gustarr_item_becomes_reject(conn, cfg, fake_http, responses):
    arr.sync(conn, cfg)
    # user deletes both radarr movies: only the gustarr-tagged one is a signal
    responses[f"{RADARR}/api/v3/movie"] = []
    stats = arr.sync(conn, cfg)
    assert stats["rejects"] == 1
    assert stats["removed"] == 2

    rejects = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchall()
    assert len(rejects) == 1
    rej = rejects[0]
    assert rej["item_id"] == "movie:tmdb:949"
    assert rej["weight"] == WEIGHTS["reject"]
    assert rej["source"] == "arr"
    assert json.loads(rej["meta"]) == {"deleted": True}

    assert conn.execute("SELECT COUNT(*) FROM library WHERE arr='radarr'").fetchone()[0] == 0
    assert json.loads(db.get_state(conn, "arr:known:radarr")) == {}
    # untouched arrs keep their rows and produce nothing new
    assert conn.execute("SELECT COUNT(*) FROM library").fetchone()[0] == 2


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
