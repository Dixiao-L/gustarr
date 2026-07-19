"""Offline tests for the enrich stage: all HTTP mocked at gustarr.http.request_json."""

from __future__ import annotations

import json

import pytest

from gustarr import config as C
from gustarr import db, http
from gustarr.enrich import run

MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"

BASE = {"enriched": 0, "merged": 0, "alias_conflicts": 0, "skipped": 0, "errors": 0}

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
    "poster_path": "/matrix.jpg",
    "videos": {"results": [
        {"site": "YouTube", "type": "Teaser", "official": True, "key": "mx-teaser"},
        {"site": "YouTube", "type": "Trailer", "official": True, "key": "mx-trailer"},
    ]},
}

SERIES_DETAIL = {
    "id": 1396,
    "name": "Breaking Bad",
    "first_air_date": "2008-01-20",
    "genres": [{"name": "Drama"}],
    "keywords": {"results": [{"name": "drug cartel"}]},
    "external_ids": {"tvdb_id": 81189, "imdb_id": "tt0903747"},
    "overview": "A chemistry teacher breaks bad.",
    "original_language": "en",
    "popularity": 300.0,
    "number_of_seasons": 5,
    "poster_path": "/bb.jpg",
    "videos": {"results": [
        {"site": "Vimeo", "type": "Trailer", "official": True, "key": "vim-1"},
        {"site": "YouTube", "type": "Trailer", "official": True, "key": "bb-trailer"},
    ]},
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

DEEZER_PIC = "https://cdn-images.dzcdn.net/images/artist/abc123/500x500-000000-80-0-0.jpg"

DEEZER_ARTIST = {"data": [{"id": 399, "name": "Radiohead", "picture_big": DEEZER_PIC}]}

RG_MBID = "b1392450-e666-3926-a536-22c65f834433"

MB_RELEASE_GROUP = {
    "id": RG_MBID,
    "title": "OK Computer",
    "first-release-date": "1997-05-21",
    "primary-type": "Album",
    "artist-credit": [{"artist": {"id": MBID, "name": "Radiohead"}}],
    "tags": [
        {"count": 3, "name": "art rock"},
        {"count": 9, "name": "alternative rock"},
    ],
    "genres": [{"count": 7, "name": "art rock"}],
}

# a RELEASE mbid Last.fm hands out where a release-group id belongs
REL_MBID = "f1d5f0b2-3f9e-4b8a-8a3d-2c1e5b6a7c89"

YORU_MBID = "0b8a3e2b-6f1d-4c58-9e2a-7f0f2b1c9d44"

OTHER_MBID = "11111111-2222-3333-4444-555555555555"

# kana-primary artist whose romanized spellings live only in MB aliases
MB_YORUSHIKA = {
    "id": YORU_MBID,
    "name": "ヨルシカ",
    "type": "Group",
    "country": "JP",
    "life-span": {"begin": "2017"},
    "tags": [{"count": 4, "name": "j-rock"}],
    "genres": [],
    "aliases": [{"name": "Yorushika"}, {"name": "yorusika"}],
}

KINOKO_MBID = "e5a1c2d3-4b6f-4a8e-9c0d-1f2a3b4c5d6e"

# kana-primary artist whose MB alias carries the space scrobblers drop
MB_KINOKO = {
    "id": KINOKO_MBID,
    "name": "きのこ帝国",
    "type": "Group",
    "country": "JP",
    "life-span": {"begin": "2007"},
    "tags": [{"count": 3, "name": "shoegaze"}],
    "genres": [],
    "aliases": [{"name": "Kinoko Teikoku"}],
}

ALPHA_MBID = "aaaa1111-2222-4333-8444-555566667777"
BETA_MBID = "bbbb1111-2222-4333-8444-555566667777"

# two authoritative artists whose alias lists fold to the SAME spaceless
# key: "Kinoko Teikoku" and "Kinoko  Tei Koku" are both "kinokoteikoku"
# once whitespace folds out, so a name-only "KinokoTeikoku" twin has no
# defensible owner between them
MB_ALPHA = {
    "id": ALPHA_MBID,
    "name": "Alpha",
    "type": "Group",
    "tags": [],
    "genres": [],
    "aliases": [{"name": "Kinoko Teikoku"}],
}

MB_BETA = {
    "id": BETA_MBID,
    "name": "Beta",
    "type": "Group",
    "tags": [],
    "genres": [],
    "aliases": [{"name": "Kinoko  Tei Koku"}],
}


def item(conn, item_id):
    return conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()


def meta_of(conn, item_id):
    return json.loads(item(conn, item_id)["meta"])


def name_target(conn, name):
    return db.lookup_item(conn, "artist", "name", name)


def mbid_keys(conn, item_id):
    return {r["key"] for r in conn.execute(
        "SELECT key FROM identities WHERE item_id=? AND ns='mbid'", (item_id,))}


def event_items(conn):
    return [r["item_id"] for r in conn.execute("SELECT item_id FROM events ORDER BY ts")]


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
    iid = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    api = mock_api(monkeypatch, [("/movie/603", MOVIE_DETAIL)])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    assert row["year"] == 1999
    meta = json.loads(row["meta"])
    assert meta["genres"] == ["Action", "Science Fiction"]
    assert meta["keywords"] == ["cyberpunk", "dystopia"]
    assert meta["runtime"] == 136
    assert meta["original_language"] == "en"
    assert meta["poster_path"] == "/matrix.jpg"
    assert meta["trailer"] == "mx-trailer"
    assert api.calls[0][2]["append_to_response"] == "keywords,videos"


def test_movie_imdb_find_attaches_tmdb(conn, tmp_path, monkeypatch):
    """An imdb-only movie learns its tmdb identity in place: same item,
    no merge, and actuation can reach it afterwards."""
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    fid = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "complete", 0.8, "jellyfin")
    api = mock_api(monkeypatch, [
        ("/find/tt0133093", {"movie_results": [{"id": 603}]}),
        ("/movie/603", MOVIE_DETAIL),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert db.identities_of(conn, fid) == {"imdb": "tt0133093", "tmdb": "603"}
    assert event_items(conn) == [fid]
    assert item(conn, fid)["enriched_at"] is not None
    find_call = api.calls[0]
    assert find_call[2]["external_source"] == "imdb_id"


def test_movie_imdb_find_merges_with_tmdb_twin_and_repoints_events(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    twin = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    db.upsert_item_fields(conn, twin, enriched=True)  # enriched last run
    fid = db.resolve_item(conn, "movie", "imdb", "tt0133093")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "complete", 0.8, "jellyfin")
    mock_api(monkeypatch, [
        ("/find/tt0133093", {"movie_results": [{"id": 603}]}),
        ("/movie/603", MOVIE_DETAIL),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    assert event_items(conn) == [twin]
    # the imdb spelling now lands on the survivor on arrival
    assert db.lookup_item(conn, "movie", "imdb", "tt0133093") == twin
    assert item(conn, twin)["enriched_at"] is not None


def test_series_tvdb_find_then_detail(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "series", "tvdb", "81189")
    api = mock_api(monkeypatch, [
        ("/find/81189", {"tv_results": [{"id": 1396}]}),
        ("/tv/1396", SERIES_DETAIL),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    assert row["title"] == "Breaking Bad"
    assert row["year"] == 2008
    assert db.identities_of(conn, iid) == {"tvdb": "81189", "tmdb": "1396"}
    meta = json.loads(row["meta"])
    assert meta["keywords"] == ["drug cartel"]
    assert meta["number_of_seasons"] == 5
    assert meta["poster_path"] == "/bb.jpg"
    assert meta["trailer"] == "bb-trailer"  # the Vimeo trailer must be skipped
    assert "no_tvdb" not in meta
    assert api.calls[1][2]["append_to_response"] == "keywords,external_ids,videos"


def test_series_tmdb_keyed_learns_tvdb_and_merges_with_twin(conn, tmp_path, monkeypatch):
    """A tmdb-keyed candidate series must learn the tvdb identity Sonarr
    adds by; when the tvdb item already exists (arr inventory) the two
    merge, dragging the candidate rows along."""
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    twin = db.resolve_item(conn, "series", "tvdb", "81189", title="Breaking Bad")
    db.upsert_item_fields(conn, twin, enriched=True)
    fid = db.resolve_item(conn, "series", "tmdb", "1396", title="Breaking Bad")
    conn.execute(
        "INSERT INTO candidates (item_id, source, first_seen, last_seen) VALUES (?,?,?,?)",
        (fid, "tmdb_similar", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))
    api = mock_api(monkeypatch, [("/tv/1396", SERIES_DETAIL)])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    cands = [r["item_id"] for r in conn.execute("SELECT item_id FROM candidates")]
    assert cands == [twin]
    assert db.identities_of(conn, twin) == {"tvdb": "81189", "tmdb": "1396"}
    assert item(conn, twin)["enriched_at"] is not None
    assert "external_ids" in api.calls[0][2]["append_to_response"]


def test_null_poster_path_never_stored(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "603")
    mock_api(monkeypatch, [("/movie/603", {**MOVIE_DETAIL, "poster_path": None})])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert "poster_path" not in meta_of(conn, iid)


def test_trailer_official_beats_fan_trailer_and_teaser(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "603")
    videos = {"results": [
        {"site": "YouTube", "type": "Teaser", "official": True, "key": "tz"},
        {"site": "YouTube", "type": "Trailer", "official": False, "key": "fan"},
        {"site": "YouTube", "type": "Trailer", "official": True, "key": "off"},
    ]}
    mock_api(monkeypatch, [("/movie/603", {**MOVIE_DETAIL, "videos": videos})])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert meta_of(conn, iid)["trailer"] == "off"


def test_trailer_skips_non_youtube_and_falls_back_to_teaser(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "603")
    videos = {"results": [
        {"site": "Vimeo", "type": "Trailer", "official": True, "key": "vim"},
        {"site": "YouTube", "type": "Teaser", "official": False, "key": "tz"},
    ]}
    mock_api(monkeypatch, [("/movie/603", {**MOVIE_DETAIL, "videos": videos})])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert meta_of(conn, iid)["trailer"] == "tz"


def test_no_usable_video_means_no_trailer_key(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    plain = db.resolve_item(conn, "movie", "tmdb", "603")
    vimeo_only = db.resolve_item(conn, "movie", "tmdb", "604")
    no_videos = {k: v for k, v in MOVIE_DETAIL.items() if k != "videos"}
    mock_api(monkeypatch, [
        ("/movie/603", no_videos),
        ("/movie/604", {**MOVIE_DETAIL, "id": 604, "videos": {"results": [
            {"site": "Vimeo", "type": "Trailer", "official": True, "key": "vim"}]}}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 2}
    assert "trailer" not in meta_of(conn, plain)
    assert "trailer" not in meta_of(conn, vimeo_only)


def test_series_without_tvdb_mapping_enriched_under_tmdb(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    fid = db.resolve_item(conn, "series", "tmdb", "1396")
    mock_api(monkeypatch, [("/tv/1396", {**SERIES_DETAIL, "external_ids": {"tvdb_id": None}})])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, fid)
    assert row["enriched_at"] is not None
    assert json.loads(row["meta"])["no_tvdb"] is True
    assert "tvdb" not in db.identities_of(conn, fid)


def test_artist_mbid_musicbrainz_plus_lastfm(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    iid = db.resolve_item(conn, "artist", "mbid", MBID)
    api = mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", DEEZER_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
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
    assert meta["image"] == DEEZER_PIC
    # stored even when MB knows none: key-present marks "fetched" for dedupe
    assert meta["aliases"] == []
    # the primary spelling now lands on this item on arrival
    assert name_target(conn, "Radiohead") == iid
    dz = [c for c in api.calls if "deezer" in c[1]]
    assert dz == [("GET", "https://api.deezer.com/search/artist",
                   {"q": "Radiohead", "limit": 1})]


def test_artist_name_only_search_attaches_mbid(conn, tmp_path, monkeypatch):
    """A confident MB search upgrades the name-keyed item in place: it
    gains the mbid identity and its events never move."""
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    fid = db.resolve_item(conn, "artist", "name", "Radiohead", title="Radiohead")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("/ws/2/artist", {"artists": [{"id": MBID, "score": 100, "name": "Radiohead"}]}),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", DEEZER_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert db.identities_of(conn, fid)["mbid"] == MBID
    assert event_items(conn) == [fid]
    row = item(conn, fid)
    assert row["enriched_at"] is not None
    assert json.loads(row["meta"])["bio"] == "Radiohead are an English rock band."
    assert json.loads(row["meta"])["image"] == DEEZER_PIC
    search_call = api.calls[0]
    assert search_call[2]["query"] == 'artist:"Radiohead"'


def test_artist_name_only_merges_into_existing_mbid_item(conn, tmp_path, monkeypatch):
    """When the mbid item already exists, learning the mbid reveals the
    twin: the authoritative holder survives and the scrobbles follow."""
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    cid = db.resolve_item(conn, "artist", "mbid", MBID)
    fid = db.resolve_item(conn, "artist", "name", "Radiohead", title="Radiohead")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("/ws/2/artist", {"artists": [{"id": MBID, "score": 100, "name": "Radiohead"}]}),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", DEEZER_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    assert event_items(conn) == [cid]
    row = item(conn, cid)
    assert row["enriched_at"] is not None
    assert json.loads(row["meta"])["bio"] == "Radiohead are an English rock band."
    assert name_target(conn, "Radiohead") == cid


def test_artist_fallback_low_score_enriches_from_lastfm_only(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    fid = db.resolve_item(conn, "artist", "name", "Radiohead", title="Radiohead")
    mock_api(monkeypatch, [
        # the low score triggers an alias check on the top hit; none match
        ("/ws/2/artist/zzz", {"id": "zzz", "name": "Radio Head Trib",
                              "aliases": [{"name": "The Tribute"}]}),
        ("/ws/2/artist", {"artists": [{"id": "zzz", "score": 55, "name": "Radio Head Trib"}]}),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", DEEZER_ARTIST),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, fid)
    assert row["enriched_at"] is not None
    meta = json.loads(row["meta"])
    assert meta["listeners"] == 5000000
    assert meta["image"] == DEEZER_PIC  # portraits work even without an MB match
    assert "mbid" not in db.identities_of(conn, fid)


def test_artist_deezer_failure_never_fails_item(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    iid = db.resolve_item(conn, "artist", "mbid", MBID)
    url = "https://api.deezer.com/search/artist"
    mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", http.ApiError(url, 503, "down")),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    meta = json.loads(row["meta"])
    assert "image" not in meta
    assert meta["bio"] == "Radiohead are an English rock band."  # Last.fm meta survived


@pytest.mark.parametrize("payload", [
    {"data": []},  # no hit
    {"data": [{"name": "Radiohead", "picture_big": ""}]},  # empty url
    # Deezer's generic silhouette lives under an empty /artist// path
    {"data": [{"name": "Radiohead",
               "picture_big": "https://cdn-images.dzcdn.net/images/artist//500x500.jpg"}]},
])
def test_artist_deezer_placeholder_or_miss_skipped(conn, tmp_path, monkeypatch, payload):
    cfg = make_cfg(tmp_path, lastfm={"api_key": "lk"})
    iid = db.resolve_item(conn, "artist", "mbid", MBID)
    mock_api(monkeypatch, [
        (f"/artist/{MBID}", MB_ARTIST),
        ("audioscrobbler", LASTFM_ARTIST),
        ("deezer", payload),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert "image" not in meta_of(conn, iid)


def test_artist_alias_bridging_merges_romaji_twin_with_events(conn, tmp_path, monkeypatch):
    """The live-store failure mode: months of scrobbles sit on a romaji
    name-keyed item while the mbid item is kana-primary. Enriching the
    mbid item must pull the twin (and its events) in via MB's alias list."""
    cfg = make_cfg(tmp_path)
    fid = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    db.upsert_item_fields(conn, fid, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    db.add_event(conn, "2026-02-01T00:00:00Z", fid, "loved", 1.0, "lastfm")
    cid = db.resolve_item(conn, "artist", "mbid", YORU_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{YORU_MBID}", MB_YORUSHIKA),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    assert event_items(conn) == [cid, cid]
    row = item(conn, cid)
    assert row["title"] == "ヨルシカ"
    assert json.loads(row["meta"])["aliases"] == ["Yorushika", "yorusika"]  # raw, as MB wrote them
    # every spelling — primary and aliases — now lands here on arrival
    assert name_target(conn, "Yorushika") == cid
    assert name_target(conn, "ヨルシカ") == cid
    assert name_target(conn, "yorusika") == cid


def test_artist_low_score_search_rescued_by_exact_alias(conn, tmp_path, monkeypatch):
    """A romaji query against a kana-primary artist scores below
    MB_MERGE_SCORE while being exactly right; the alias list proves it."""
    cfg = make_cfg(tmp_path)
    fid = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [
        (f"/artist/{YORU_MBID}", MB_YORUSHIKA),
        ("/ws/2/artist", {"artists": [{"id": YORU_MBID, "score": 62, "name": "ヨルシカ",
                                       "aliases": [{"name": "Yorushika"}]}]}),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert db.identities_of(conn, fid)["mbid"] == YORU_MBID
    assert event_items(conn) == [fid]
    assert item(conn, fid)["title"] == "ヨルシカ"
    # the search payload carried aliases, so no extra detail fetch happened
    detail_calls = [c for c in api.calls if f"/artist/{YORU_MBID}" in c[1]]
    assert len(detail_calls) == 1


def test_artist_low_score_search_fetches_aliases_when_payload_lacks_them(
        conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    fid = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    api = mock_api(monkeypatch, [
        (f"/artist/{YORU_MBID}", MB_YORUSHIKA),
        # search hit has no alias list at all → top hit fetched with inc=aliases
        ("/ws/2/artist", {"artists": [{"id": YORU_MBID, "score": 70, "name": "ヨルシカ"}]}),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    assert db.identities_of(conn, fid)["mbid"] == YORU_MBID
    detail_calls = [c for c in api.calls if f"/artist/{YORU_MBID}" in c[1]]
    assert detail_calls[0][2] == {"inc": "aliases", "fmt": "json"}


def test_artist_shared_alias_with_other_mbid_artist_is_refused(conn, tmp_path, monkeypatch):
    """A spelling already claimed by another artist holding its OWN mbid
    is a conflict, not an identity assertion: MB alias lists carry other
    entities' names (former band names, personas), so the attach is
    refused, counted, and both artists survive with their histories."""
    cfg = make_cfg(tmp_path)
    other = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="Yorushika")
    db.attach_identity(conn, other, "name", "Yorushika")
    db.upsert_item_fields(conn, other, enriched=True)
    cid = db.resolve_item(conn, "artist", "mbid", YORU_MBID)
    db.add_event(conn, "2026-01-01T00:00:00Z", cid, "scrobble", 0.15, "lastfm")
    mock_api(monkeypatch, [
        (f"/artist/{YORU_MBID}", MB_YORUSHIKA),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "alias_conflicts": 1}
    assert item(conn, cid) is not None
    assert event_items(conn) == [cid]
    # each keeps its own mbid and its own spellings; the contested one
    # stays with its current owner
    assert mbid_keys(conn, cid) == {YORU_MBID}
    assert mbid_keys(conn, other) == {OTHER_MBID}
    assert name_target(conn, "Yorushika") == other
    assert name_target(conn, "ヨルシカ") == cid
    assert name_target(conn, "yorusika") == cid


def test_artist_alias_spaceless_twin_merges_with_events(conn, tmp_path, monkeypatch):
    """The 0.5.0 whitespace fold: MB's alias "Kinoko Teikoku" arrives
    while months of scrobbles sit on a spelling twin that dropped the
    space. The twin is the same exact string after a deterministic fold —
    never fuzzy — so its history unites under the mbid item."""
    cfg = make_cfg(tmp_path)
    fid = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    db.upsert_item_fields(conn, fid, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    db.add_event(conn, "2026-02-01T00:00:00Z", fid, "loved", 1.0, "lastfm")
    cid = db.resolve_item(conn, "artist", "mbid", KINOKO_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{KINOKO_MBID}", MB_KINOKO),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    assert event_items(conn) == [cid, cid]
    # both spellings land here on arrival, each under its own stored
    # normalize_key'd row: spaceless is a lookup fold, never a storage key
    assert name_target(conn, "Kinoko Teikoku") == cid
    assert name_target(conn, "KinokoTeikoku") == cid
    assert name_target(conn, "きのこ帝国") == cid
    keys = {r["key"] for r in conn.execute(
        "SELECT key FROM identities WHERE item_id=? AND ns='name'", (cid,))}
    assert keys == {"きのこ帝国", "kinoko teikoku", "kinokoteikoku"}

    # idempotent: re-enriching the survivor re-registers every spelling
    # silently — nothing merges twice, no event moves
    conn.execute("UPDATE items SET enriched_at=NULL WHERE id=?", (cid,))
    assert run(conn, cfg) == {**BASE, "enriched": 1}
    assert event_items(conn) == [cid, cid]


def test_artist_spaceless_twin_with_own_mbid_is_refused(conn, tmp_path, monkeypatch):
    """The cross-entity refusal rule is untouched by the whitespace fold:
    a spaceless twin holding its OWN mbid is a different entity, so the
    hunt counts a conflict and merges nothing."""
    cfg = make_cfg(tmp_path)
    other = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="KinokoTeikoku")
    db.attach_identity(conn, other, "name", "KinokoTeikoku")
    db.upsert_item_fields(conn, other, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", other, "scrobble", 0.15, "lastfm")
    cid = db.resolve_item(conn, "artist", "mbid", KINOKO_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{KINOKO_MBID}", MB_KINOKO),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "alias_conflicts": 1}
    # both artists survive; the twin keeps its spelling and its history
    assert item(conn, cid) is not None and item(conn, other) is not None
    assert event_items(conn) == [other]
    assert mbid_keys(conn, cid) == {KINOKO_MBID}
    assert mbid_keys(conn, other) == {OTHER_MBID}
    assert name_target(conn, "KinokoTeikoku") == other
    # the spaced alias itself still lands on the mbid item
    assert name_target(conn, "Kinoko Teikoku") == cid
    assert name_target(conn, "きのこ帝国") == cid


def test_artist_two_claimant_fold_refuses_name_only_twin(conn, tmp_path, monkeypatch):
    """Two authoritative artists' spellings fold to the same spaceless
    key: the twin hunt then has no defensible owner for the name-only
    scrobble twin, so the identity call is refused — the twin survives
    with its history instead of being guessed into either artist."""
    cfg = make_cfg(tmp_path)
    twin = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    db.upsert_item_fields(conn, twin, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", twin, "scrobble", 0.15, "lastfm")
    db.add_event(conn, "2026-02-01T00:00:00Z", twin, "loved", 1.0, "lastfm")
    beta = db.resolve_item(conn, "artist", "mbid", BETA_MBID, title="Beta",
                           meta={"aliases": ["Kinoko  Tei Koku"]})
    db.upsert_item_fields(conn, beta, enriched=True)  # settled a previous run
    alpha = db.resolve_item(conn, "artist", "mbid", ALPHA_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{ALPHA_MBID}", MB_ALPHA),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "alias_conflicts": 1}
    # the twin is nobody's: it survives whole, events included
    assert item(conn, twin) is not None
    assert event_items(conn) == [twin, twin]
    assert name_target(conn, "KinokoTeikoku") == twin
    # the spaced alias itself still lands on the hunting artist
    assert name_target(conn, "Kinoko Teikoku") == alpha


def test_artist_two_claimant_fold_refuses_twin_either_order(conn, tmp_path, monkeypatch):
    """THE pin is order-independence: Alpha settled first and Beta doing
    the hunting must refuse just the same — whichever artist enriches
    first, the twin is never absorbed."""
    cfg = make_cfg(tmp_path)
    twin = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    db.upsert_item_fields(conn, twin, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", twin, "scrobble", 0.15, "lastfm")
    db.add_event(conn, "2026-02-01T00:00:00Z", twin, "loved", 1.0, "lastfm")
    alpha = db.resolve_item(conn, "artist", "mbid", ALPHA_MBID, title="Alpha",
                            meta={"aliases": ["Kinoko Teikoku"]})
    db.upsert_item_fields(conn, alpha, enriched=True)  # settled a previous run
    beta = db.resolve_item(conn, "artist", "mbid", BETA_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{BETA_MBID}", MB_BETA),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "alias_conflicts": 1}
    assert item(conn, twin) is not None
    assert event_items(conn) == [twin, twin]
    assert name_target(conn, "KinokoTeikoku") == twin
    assert name_target(conn, "Kinoko  Tei Koku") == beta


def test_artist_attached_identity_claimant_still_refuses_twin(conn, tmp_path, monkeypatch):
    """A claimant needs no fetched alias list: Beta holding the folded
    spelling as an attached name identity alone (no meta.aliases) still
    bars the hunt — the claimant index reads identities as well as the
    alias lists enrich fetched."""
    cfg = make_cfg(tmp_path)
    twin = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    db.upsert_item_fields(conn, twin, enriched=True)
    db.add_event(conn, "2026-01-01T00:00:00Z", twin, "scrobble", 0.15, "lastfm")
    beta = db.resolve_item(conn, "artist", "mbid", BETA_MBID, title="Beta")
    db.attach_identity(conn, beta, "name", "Kinoko Tei Koku")
    db.upsert_item_fields(conn, beta, enriched=True)  # aliases never fetched
    alpha = db.resolve_item(conn, "artist", "mbid", ALPHA_MBID)
    mock_api(monkeypatch, [
        (f"/artist/{ALPHA_MBID}", MB_ALPHA),
        ("deezer", {"data": []}),
    ])

    stats = run(conn, cfg)

    # both stored keys in the contested bucket (the twin's and beta's
    # own identity) are refused to the hunter
    assert stats == {**BASE, "enriched": 1, "alias_conflicts": 2}
    assert item(conn, twin) is not None and item(conn, beta) is not None
    assert event_items(conn) == [twin]
    assert name_target(conn, "KinokoTeikoku") == twin
    assert name_target(conn, "Kinoko Tei Koku") == beta
    assert name_target(conn, "Kinoko Teikoku") == alpha


def test_track_name_keyed_just_marked(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    tid = db.resolve_item(conn, "track", "name", "Radiohead Paranoid Android",
                          title="Paranoid Android")
    api = mock_api(monkeypatch, [])

    stats = run(conn, cfg)

    assert stats == {**BASE, "skipped": 1}
    assert api.calls == []
    row = item(conn, tid)
    assert row["enriched_at"] is not None
    assert "enrich_error" not in json.loads(row["meta"])


def test_album_mbid_release_group_lookup(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "album", "mbid", RG_MBID)
    api = mock_api(monkeypatch, [(f"/release-group/{RG_MBID}", MB_RELEASE_GROUP)])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    assert row["title"] == "OK Computer" and row["year"] == 1997
    meta = json.loads(row["meta"])
    assert meta["artists"] == ["Radiohead"] and meta["artist"] == "Radiohead"
    assert meta["tags"] == ["alternative rock", "art rock"]
    assert meta["genres"] == ["art rock"]
    assert meta["type"] == "Album"
    assert meta["image"] == f"https://coverartarchive.org/release-group/{RG_MBID}/front-250"
    assert meta["release_group_mbid"] == RG_MBID  # what apply's _music_mbid reads
    assert meta["artist_mbid"] == MBID  # a relation, so meta — never identities
    assert mbid_keys(conn, iid) == {RG_MBID}
    # the cover URL is stored optimistically: the only request is MB's
    assert [c[1] for c in api.calls] == \
        [f"https://musicbrainz.org/ws/2/release-group/{RG_MBID}"]
    assert api.calls[0][2] == {"inc": "artist-credits+tags+genres", "fmt": "json"}


def test_album_titled_but_untagged_still_enriched(conn, tmp_path, monkeypatch):
    """Collector-titled albums lack tags/cover, which ranking and the UI
    now consume, so a title alone must not fast-stamp an mbid album."""
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "album", "mbid", RG_MBID, title="OK Computer")
    mock_api(monkeypatch, [(f"/release-group/{RG_MBID}", MB_RELEASE_GROUP)])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    meta = meta_of(conn, iid)
    assert meta["tags"] == ["alternative rock", "art rock"]
    assert meta["image"].endswith(f"/release-group/{RG_MBID}/front-250")


def test_album_titled_with_tags_fast_stamps(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "album", "mbid", RG_MBID, title="OK Computer",
                          meta={"tags": ["art rock"]})
    api = mock_api(monkeypatch, [])

    stats = run(conn, cfg)

    assert stats == {**BASE, "skipped": 1}
    assert api.calls == []  # already enriched: no MB round-trip on requeue
    assert item(conn, iid)["enriched_at"] is not None


def test_album_without_mbid_fast_stamps(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    db.resolve_item(conn, "album", "name", "Radiohead OK Computer", title="OK Computer")
    api = mock_api(monkeypatch, [])

    stats = run(conn, cfg)

    assert stats == {**BASE, "skipped": 1}
    assert api.calls == []


def test_album_release_mbid_404_attaches_release_group(conn, tmp_path, monkeypatch):
    """Last.fm sometimes hands over a RELEASE mbid; the release-group
    lookup 404s, but one /release call recovers the group id instead of
    stamping a permanent error. The release id identity stays (it keeps
    landing arriving scrobbles here) and the group id actuation needs
    travels in meta.release_group_mbid."""
    cfg = make_cfg(tmp_path)
    fid = db.resolve_item(conn, "album", "mbid", REL_MBID)
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    rg_url = f"https://musicbrainz.org/ws/2/release-group/{REL_MBID}"
    api = mock_api(monkeypatch, [
        (f"/release-group/{REL_MBID}", http.ApiError(rg_url, 404, "not found")),
        (f"/release-group/{RG_MBID}", MB_RELEASE_GROUP),
        (f"/release/{REL_MBID}", {"id": REL_MBID, "release-group": {"id": RG_MBID}}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1}
    row = item(conn, fid)
    assert row["enriched_at"] is not None
    assert row["title"] == "OK Computer"
    assert event_items(conn) == [fid]
    assert mbid_keys(conn, fid) == {REL_MBID, RG_MBID}
    # identities_of keeps the original release id first; apply reads the
    # meta key, so actuation still gets the group id
    assert db.identities_of(conn, fid)["mbid"] == REL_MBID
    meta = json.loads(row["meta"])
    assert meta["release_group_mbid"] == RG_MBID
    assert meta["image"].endswith(f"/release-group/{RG_MBID}/front-250")
    release_call = [c for c in api.calls if f"/release/{REL_MBID}" in c[1]]
    assert release_call[0][2] == {"inc": "release-groups", "fmt": "json"}


def test_album_release_mbid_404_merges_with_release_group_twin(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    twin = db.resolve_item(conn, "album", "mbid", RG_MBID)
    fid = db.resolve_item(conn, "album", "mbid", REL_MBID)
    db.add_event(conn, "2026-01-01T00:00:00Z", fid, "scrobble", 0.15, "lastfm")
    rg_url = f"https://musicbrainz.org/ws/2/release-group/{REL_MBID}"
    mock_api(monkeypatch, [
        (f"/release-group/{REL_MBID}", http.ApiError(rg_url, 404, "not found")),
        (f"/release-group/{RG_MBID}", MB_RELEASE_GROUP),
        (f"/release/{REL_MBID}", {"id": REL_MBID, "release-group": {"id": RG_MBID}}),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "enriched": 1, "merged": 1}
    assert item(conn, fid) is None
    assert event_items(conn) == [twin]
    assert mbid_keys(conn, twin) == {REL_MBID, RG_MBID}
    assert db.identities_of(conn, twin)["mbid"] == RG_MBID  # its own id stays first
    assert meta_of(conn, twin)["release_group_mbid"] == RG_MBID
    assert item(conn, twin)["enriched_at"] is not None


def test_album_double_404_is_permanent(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "album", "mbid", REL_MBID)
    rg_url = f"https://musicbrainz.org/ws/2/release-group/{REL_MBID}"
    rel_url = f"https://musicbrainz.org/ws/2/release/{REL_MBID}"
    mock_api(monkeypatch, [
        (f"/release-group/{REL_MBID}", http.ApiError(rg_url, 404, "not found")),
        (f"/release/{REL_MBID}", http.ApiError(rel_url, 404, "not found")),
    ])

    stats = run(conn, cfg)

    assert stats == {**BASE, "errors": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None  # permanently stamped, never requeued
    assert "404" in json.loads(row["meta"])["enrich_error"]


def test_album_release_lookup_5xx_stays_queued(conn, tmp_path, monkeypatch):
    """A transient failure on the recovery lookup must not become
    permanent just because the first 404 was."""
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "album", "mbid", REL_MBID)
    rg_url = f"https://musicbrainz.org/ws/2/release-group/{REL_MBID}"
    rel_url = f"https://musicbrainz.org/ws/2/release/{REL_MBID}"
    mock_api(monkeypatch, [
        (f"/release-group/{REL_MBID}", http.ApiError(rg_url, 404, "not found")),
        (f"/release/{REL_MBID}", http.ApiError(rel_url, 503, "down")),
    ])

    stats = run(conn, cfg)

    assert stats["errors"] == 1
    assert item(conn, iid)["enriched_at"] is None


def test_track_titled_with_mbid_keeps_fast_path(conn, tmp_path, monkeypatch):
    """Only albums earn the release-group lookup: a titled track with an
    mbid must still stamp without any HTTP."""
    cfg = make_cfg(tmp_path)
    tid = db.resolve_item(conn, "track", "mbid", "0f13fa17-40ee-4d3d-b16f-6a832d2d1a29",
                          title="Paranoid Android")
    api = mock_api(monkeypatch, [])

    stats = run(conn, cfg)

    assert stats == {**BASE, "skipped": 1}
    assert api.calls == []
    assert item(conn, tid)["enriched_at"] is not None


def test_api_error_still_sets_enriched_at(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "604", title="Broken")
    url = "https://api.themoviedb.org/3/movie/604"
    mock_api(monkeypatch, [("/movie/604", http.ApiError(url, 404, "not found"))])

    stats = run(conn, cfg)

    assert stats == {**BASE, "errors": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    assert "404" in json.loads(row["meta"])["enrich_error"]


def test_lookup_miss_is_permanent(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    fid = db.resolve_item(conn, "movie", "imdb", "tt0000000")
    mock_api(monkeypatch, [("/find/tt0000000", {"movie_results": []})])

    stats = run(conn, cfg)

    assert stats["errors"] == 1
    row = item(conn, fid)
    assert row["enriched_at"] is not None
    assert "no movie for imdb" in json.loads(row["meta"])["enrich_error"]


@pytest.mark.parametrize("status", [None, 500, 503, 429, 401, 403])
def test_transient_failure_leaves_item_queued_for_retry(conn, tmp_path, monkeypatch, status):
    """Outages, rate limits and bad credentials must not poison the
    backlog: enriched_at stays NULL and the next run picks the item up."""
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "605", title="Flaky")
    url = "https://api.themoviedb.org/3/movie/605"
    mock_api(monkeypatch, [("/movie/605", http.ApiError(url, status, "boom"))])

    stats = run(conn, cfg)

    assert stats == {**BASE, "errors": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is None
    assert "boom" in json.loads(row["meta"])["enrich_error"]

    # service recovered: the same item is retried and enriched, and the
    # stale failure note goes with it (meta merges key-wise, so the
    # success write alone would leave it behind)
    mock_api(monkeypatch, [("/movie/605", {**MOVIE_DETAIL, "id": 605, "title": "Flaky"})])
    stats = run(conn, cfg)
    assert stats == {**BASE, "enriched": 1}
    row = item(conn, iid)
    assert row["enriched_at"] is not None
    assert "enrich_error" not in json.loads(row["meta"])


def test_unexpected_exception_is_transient(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    iid = db.resolve_item(conn, "movie", "tmdb", "606", title="Buggy")
    mock_api(monkeypatch, [("/movie/606", ValueError("surprise"))])

    stats = run(conn, cfg)

    assert stats["errors"] == 1
    row = item(conn, iid)
    assert row["enriched_at"] is None
    assert "surprise" in json.loads(row["meta"])["enrich_error"]


def test_referenced_items_first_and_limit(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    quiet = db.resolve_item(conn, "movie", "tmdb", "1")
    hot = db.resolve_item(conn, "movie", "tmdb", "2")
    db.add_event(conn, "2026-01-01T00:00:00Z", hot, "play", 0.3, "jellyfin")
    mock_api(monkeypatch, [("/movie/2", {**MOVIE_DETAIL, "id": 2})])

    stats = run(conn, cfg, limit=1)

    assert stats == {**BASE, "enriched": 1}
    assert item(conn, hot)["enriched_at"] is not None
    assert item(conn, quiet)["enriched_at"] is None


def test_domain_filter(conn, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, tmdb={"api_key": "k"})
    db.resolve_item(conn, "movie", "tmdb", "603")
    db.resolve_item(conn, "track", "name", "a b", title="b")
    mock_api(monkeypatch, [])

    stats = run(conn, cfg, domain="track")

    assert stats == {**BASE, "skipped": 1}
    movie = conn.execute("SELECT enriched_at FROM items WHERE domain='movie'").fetchone()
    assert movie["enriched_at"] is None
