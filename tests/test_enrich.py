"""Offline tests for the enrich stage: all HTTP mocked at gustarr.http.request_json."""

from __future__ import annotations

import json

import pytest

from gustarr import config as C
from gustarr import db, http, ids
from gustarr.enrich import run

MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"

MOVIE_DETAIL = {
    "id": 603,
    "title": "The Matrix",
    "release_date": "1999-03-30",
    "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
    "keywords": {"keywords": [{"id": 1, "name": "cyberpunk"}, {"id": 2, "name": "dystopia"}]},
    "overview": "A hacker learns the truth.",
    "original_language": "en",
    "popularity": 81.5,
    "vote_average": 8.2,
    "runtime": 136,
}

SERIES_DETAIL = {
    "id": 1396,
    "name": "Breaking Bad",
    "first_air_date": "2008-01-20",
    "genres": [{"name": "Drama"}],
    "keywords": {"results": [{"name": "drug cartel"}]},
    "overview": "A chemistry teacher breaks bad.",
    "original_language": "en",
    "popularity": 300.0,
    "number_of_seasons": 5,
}

MB_ARTIST = {
    "id": MBID,
    "name": "Radiohead",
    "type": "Group",
    "country": "GB",
    "life-span": {"begin": "1991"},
    "tags": [
        {"count": 5, "name": "rock"},
        {"count": 30, "name": "alternative rock"},
        {"count": 12, "name": "experimental"},
    ],
    "genres": [{"count": 10, "name": "art rock"}],
}

LASTFM_ARTIST = {
    "artist": {
        "name": "Radiohead",
        "stats": {"listeners": "5000000", "playcount": "900000000"},
        "similar": {"artist": [{"name": "Thom Yorke"}, {"name": "Blur"}]},
        "bio": {
            "summary": "Radiohead are an English rock band. "
            '<a href="https://www.last.fm/music/Radiohead">Read more on Last.fm</a>'
        },
    }
}


class FakeApi:
    """Dispatches on URL substring, in registration order (specific first)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, *, params=None, **kw):
        self.calls.append((method, url, dict(params or {})))
        for frag, payload in self.routes:
            if frag in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected request: {url} {params}")


@pytest.fixture(autouse=True)
def _no_politeness(monkeypatch):
    monkeypatch.setattr(http, "HOST_DELAYS", {})


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def make_cfg(tmp_path, **sections):
    return C._build({"core": {"data_dir": str(tmp_path)}, **sections})


def mock_api(monkeypatch, routes):
    api = FakeApi(routes)
    monkeypatch.setattr(http, "request_json", api)
    return api


def test_movie_tmdb_detail(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = ids.make("movie", "tmdb", "603")
    db.upsert_item(conn, iid, "movie", title="The Matrix", ids={"tmdb": 603})
    mock_api(monkeypatch, [("/movie/603", MOVIE_DETAIL)])

    stats = run(conn, cfg)

    assert stats == {"enriched": 1, "merged": 0, "skipped": 0, "errors": 0}
    row = conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    assert row["enriched_at"] is not None
    assert row["year"] == 1999
    meta = json.loads(row["meta"])
    assert meta["genres"] == ["Action", "Science Fiction"]
    assert meta["keywords"] == ["cyberpunk", "dystopia"]
    assert meta["runtime"] == 136
    assert meta["original_language"] == "en"


def test_movie_imdb_find_merges_and_repoints_events(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    fid = ids.make("movie", "imdb", "tt0133093")
    db.upsert_item(conn, fid, "movie", title="The Matrix")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "complete", 0.8, "jellyfin")
    api = mock_api(monkeypatch, [
        ("/find/tt0133093", {"movie_results": [{"id": 603}]}),
        ("/movie/603", MOVIE_DETAIL),
    ])

    stats = run(conn, cfg)

    cid = ids.make("movie", "tmdb", "603")
    assert stats["merged"] == 1 and stats["enriched"] == 1 and stats["errors"] == 0
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (fid,)).fetchone() is None
    events = [r["item_id"] for r in conn.execute("SELECT item_id FROM events")]
    assert events == [cid]
    row = conn.execute("SELECT enriched_at, ids FROM items WHERE id=?", (cid,)).fetchone()
    assert row["enriched_at"] is not None
    assert json.loads(row["ids"])["tmdb"] == 603
    find_call = api.calls[0]
    assert find_call[2]["external_source"] == "imdb_id"


def test_series_tvdb_find_then_detail(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = ids.make("series", "tvdb", "81189")
    db.upsert_item(conn, iid, "series")
    mock_api(monkeypatch, [
        ("/find/81189", {"tv_results": [{"id": 1396}]}),
        ("/tv/1396", SERIES_DETAIL),
    ])

    stats = run(conn, cfg)

    assert stats["enriched"] == 1 and stats["merged"] == 0
    row = conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    assert row["enriched_at"] is not None
    assert row["title"] == "Breaking Bad"
    assert row["year"] == 2008
    assert json.loads(row["ids"]) == {"tvdb": "81189", "tmdb": 1396}
    meta = json.loads(row["meta"])
    assert meta["keywords"] == ["drug cartel"]
    assert meta["number_of_seasons"] == 5


def test_artist_mbid_musicbrainz_plus_lastfm(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    iid = ids.make("artist", "mbid", MBID)
    db.upsert_item(conn, iid, "artist", ids={"mbid": MBID})
    mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("audioscrobbler", LASTFM_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats["enriched"] == 1 and stats["errors"] == 0
    row = conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    assert row["enriched_at"] is not None
    assert row["title"] == "Radiohead"
    meta = json.loads(row["meta"])
    assert meta["tags"] == ["alternative rock", "experimental", "rock"]
    assert meta["genres"] == ["art rock"]
    assert meta["type"] == "Group" and meta["country"] == "GB"
    assert meta["begin_year"] == 1991
    assert meta["bio"] == "Radiohead are an English rock band."
    assert meta["listeners"] == 5000000
    assert meta["similar"] == ["Thom Yorke", "Blur"]


def test_artist_fallback_search_merges(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    fid = ids.make("artist", "lastfm", "Radiohead")
    db.upsert_item(conn, fid, "artist", title="Radiohead")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("/ws/2/artist", {"artists": [{"id": MBID, "score": 100, "name": "Radiohead"}]}),
        ("audioscrobbler", LASTFM_ARTIST),
    ])

    stats = run(conn, cfg)

    cid = ids.make("artist", "mbid", MBID)
    assert stats["merged"] == 1 and stats["enriched"] == 1 and stats["errors"] == 0
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (fid,)).fetchone() is None
    events = [r["item_id"] for r in conn.execute("SELECT item_id FROM events")]
    assert events == [cid]
    row = conn.execute("SELECT ids, meta, enriched_at FROM items WHERE id=?", (cid,)).fetchone()
    assert row["enriched_at"] is not None
    assert json.loads(row["ids"])["mbid"] == MBID
    assert json.loads(row["meta"])["bio"] == "Radiohead are an English rock band."
    search_call = api.calls[0]
    assert search_call[2]["query"] == 'artist:"Radiohead"'


def test_artist_fallback_low_score_enriches_from_lastfm_only(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    fid = ids.make("artist", "lastfm", "Radiohead")
    db.upsert_item(conn, fid, "artist", title="Radiohead")
    mock_api(monkeypatch, [
        ("/ws/2/artist", {"artists": [{"id": "zzz", "score": 55, "name": "Radio Head Trib"}]}),
        ("audioscrobbler", LASTFM_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats == {"enriched": 1, "merged": 0, "skipped": 0, "errors": 0}
    row = conn.execute("SELECT * FROM items WHERE id=?", (fid,)).fetchone()
    assert row["enriched_at"] is not None
    meta = json.loads(row["meta"])
    assert meta["listeners"] == 5000000
    assert "mbid" not in json.loads(row["ids"])


def test_track_lastfm_fallback_just_marked(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    tid = ids.make("track", "lastfm", "Radiohead", "Paranoid Android")
    db.upsert_item(conn, tid, "track", title="Paranoid Android")
    api = mock_api(monkeypatch, [])

    stats = run(conn, cfg)

    assert stats == {"enriched": 0, "merged": 0, "skipped": 1, "errors": 0}
    assert api.calls == []
    row = conn.execute("SELECT enriched_at, meta FROM items WHERE id=?", (tid,)).fetchone()
    assert row["enriched_at"] is not None
    assert "enrich_error" not in json.loads(row["meta"])


def test_album_mbid_release_lookup(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    amid = "1b022e01-4da6-387b-8658-8678046e4cef"
    iid = ids.make("album", "mbid", amid)
    db.upsert_item(conn, iid, "album", ids={"mbid": amid})
    mock_api(monkeypatch, [
        (f"/release/{amid}", {
            "title": "OK Computer",
            "date": "1997-05-21",
            "artist-credit": [{"artist": {"name": "Radiohead"}}],
            "tags": [{"count": 3, "name": "art rock"}],
        }),
    ])

    stats = run(conn, cfg)

    assert stats["enriched"] == 1
    row = conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    assert row["title"] == "OK Computer" and row["year"] == 1997
    meta = json.loads(row["meta"])
    assert meta["artists"] == ["Radiohead"] and meta["tags"] == ["art rock"]


def test_api_error_still_sets_enriched_at(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = ids.make("movie", "tmdb", "604")
    db.upsert_item(conn, iid, "movie", title="Broken")
    url = "https://api.themoviedb.org/3/movie/604"
    mock_api(monkeypatch, [("/movie/604", http.ApiError(url, 404, "not found"))])

    stats = run(conn, cfg)

    assert stats == {"enriched": 0, "merged": 0, "skipped": 0, "errors": 1}
    row = conn.execute("SELECT enriched_at, meta FROM items WHERE id=?", (iid,)).fetchone()
    assert row["enriched_at"] is not None
    assert "404" in json.loads(row["meta"])["enrich_error"]


def test_referenced_items_first_and_limit(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    quiet = ids.make("movie", "tmdb", "1")
    hot = ids.make("movie", "tmdb", "2")
    db.upsert_item(conn, quiet, "movie")
    db.upsert_item(conn, hot, "movie")
    db.add_event(conn, "2026-01-01T00:00:00Z", hot, "play", 0.3, "jellyfin")
    mock_api(monkeypatch, [("/movie/2", {**MOVIE_DETAIL, "id": 2})])

    stats = run(conn, cfg, limit=1)

    assert stats["enriched"] == 1 and stats["errors"] == 0
    got = conn.execute("SELECT enriched_at FROM items WHERE id=?", (hot,)).fetchone()
    left = conn.execute("SELECT enriched_at FROM items WHERE id=?", (quiet,)).fetchone()
    assert got["enriched_at"] is not None
    assert left["enriched_at"] is None


def test_domain_filter(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    db.upsert_item(conn, ids.make("movie", "tmdb", "603"), "movie", ids={"tmdb": 603})
    db.upsert_item(conn, ids.make("track", "lastfm", "a", "b"), "track")
    mock_api(monkeypatch, [])

    stats = run(conn, cfg, domain="track")

    assert stats == {"enriched": 0, "merged": 0, "skipped": 1, "errors": 0}
    movie = conn.execute(
        "SELECT enriched_at FROM items WHERE domain='movie'").fetchone()
    assert movie["enriched_at"] is None
