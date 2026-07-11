"""Offline tests for the ListenBrainz collector (httpx.MockTransport, no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from gustarr import config as C
from gustarr import db
from gustarr import http as http_mod
from gustarr.collect import listenbrainz

MBID_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
MBID_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
MBID_C = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
MBID_X = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"  # CF hit with no resolvable metadata
MBID_W = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"  # weekly-only track
ARTIST_1 = "11111111-1111-4111-8111-111111111111"
RELEASE_1 = "99999999-9999-4999-8999-999999999999"  # shared by tracks A and B
RELEASE_2 = "88888888-8888-4888-8888-888888888888"  # track C's, artist has no mbid
PL_OLD = "00000000-0000-4000-8000-000000000001"
PL_NEW = "00000000-0000-4000-8000-000000000002"

CF_RESPONSE = {
    "payload": {
        "mbids": [
            {"recording_mbid": MBID_A, "score": 0.9},
            {"recording_mbid": MBID_B, "score": 0.7},
            {"recording_mbid": MBID_C, "score": 0.5},
            {"recording_mbid": MBID_X, "score": 0.4},
        ]
    }
}

METADATA = {
    MBID_A: {
        "recording": {"name": "Song A"},
        "artist": {"name": "Artist One", "artist_credit_id": 1,
                   "artists": [{"artist_mbid": ARTIST_1, "name": "Artist One"}]},
        "release": {"name": "Album A", "mbid": RELEASE_1},
    },
    MBID_B: {
        "recording": {"name": "Song B"},
        "artist": {"name": "Artist One", "artist_credit_id": 1,
                   "artists": [{"artist_mbid": ARTIST_1, "name": "Artist One"}]},
        "release": {"name": "Album A", "mbid": RELEASE_1},
    },
    MBID_C: {
        "recording": {"name": "Song C"},
        "artist": {"name": "The Unknowns", "artists": []},
        "release": {"name": "Album C", "mbid": RELEASE_2},
    },
}

CREATEDFOR = {
    "playlists": [
        {"playlist": {"title": "Daily Jams for alice, 2026-07-08",
                      "identifier": "https://listenbrainz.org/playlist/ffffffff-0000-4000-8000-0000000000ff",
                      "date": "2026-07-08T00:00:00Z"}},
        {"playlist": {"title": "Weekly Exploration for alice, week of 2026-06-29",
                      "identifier": f"https://listenbrainz.org/playlist/{PL_OLD}",
                      "date": "2026-06-29T00:00:00Z"}},
        {"playlist": {"title": "Weekly Exploration for alice, week of 2026-07-06",
                      "identifier": f"https://listenbrainz.org/playlist/{PL_NEW}",
                      "date": "2026-07-06T00:00:00Z"}},
    ]
}

WEEKLY_JSPF = {
    "playlist": {
        "title": "Weekly Exploration for alice, week of 2026-07-06",
        "track": [
            {"identifier": [f"https://musicbrainz.org/recording/{MBID_W}"],
             "title": "Song W", "creator": "Artist Two"},
            {"identifier": f"https://musicbrainz.org/recording/{MBID_A}",
             "title": "Song A", "creator": "Artist One"},
        ],
    }
}


def make_handler(seen: list, cf_status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen.append((request.method, path, dict(request.headers),
                     request.content.decode() if request.content else ""))
        if path == "/1/cf/recommendation/user/alice/recording":
            if cf_status == 204:
                return httpx.Response(204)
            return httpx.Response(200, json=CF_RESPONSE)
        if path == "/1/metadata/recording/":
            body = json.loads(request.content)
            hits = {m: METADATA[m] for m in body["recording_mbids"] if m in METADATA}
            return httpx.Response(200, json=hits)
        if path == "/1/user/alice/playlists/createdfor":
            return httpx.Response(200, json=CREATEDFOR if cf_status == 200 else {"playlists": []})
        if path == f"/1/playlists/{PL_NEW}":
            return httpx.Response(200, json=WEEKLY_JSPF)
        return httpx.Response(404)

    return handler


def install(monkeypatch, handler) -> None:
    real = http_mod.request_json
    transport = httpx.MockTransport(handler)

    def patched(method, url, **kw):
        kw["transport"] = transport
        return real(method, url, **kw)

    monkeypatch.setattr(http_mod, "HOST_DELAYS", {})
    monkeypatch.setattr(http_mod, "request_json", patched)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    return C._build({"core": {"data_dir": str(tmp_path)},
                     "listenbrainz": {"user": "alice", "token": "sekrit"}})


def candidate(conn, item_id, source):
    return conn.execute(
        "SELECT * FROM candidates WHERE item_id=? AND source=?", (item_id, source)).fetchone()


def test_cf_candidates_scores_and_artist_aggregation(conn, cfg, monkeypatch):
    seen: list = []
    install(monkeypatch, make_handler(seen))

    stats = listenbrainz.sync(conn, cfg)

    assert stats["cf_tracks"] == 4
    assert stats["cf_artists"] == 2

    row = candidate(conn, f"track:mbid:{MBID_A}", "listenbrainz_cf")
    assert row["external_score"] == pytest.approx(0.9)
    item = conn.execute("SELECT * FROM items WHERE id=?", (f"track:mbid:{MBID_A}",)).fetchone()
    assert item["title"] == "Song A"
    assert json.loads(item["meta"])["artist"] == "Artist One"
    assert json.loads(item["ids"])["mbid"] == MBID_A

    # artist score is the max over that artist's CF tracks this run (0.9 > 0.7)
    arow = candidate(conn, f"artist:mbid:{ARTIST_1}", "listenbrainz_cf_artist")
    assert arow["external_score"] == pytest.approx(0.9)
    aitem = conn.execute("SELECT * FROM items WHERE id=?", (f"artist:mbid:{ARTIST_1}",)).fetchone()
    assert aitem["title"] == "Artist One"

    # artist with no MBID falls back to the name-keyed lastfm namespace
    fb = candidate(conn, "artist:lastfm:the unknowns", "listenbrainz_cf_artist")
    assert fb["external_score"] == pytest.approx(0.5)

    # unresolvable mbid still becomes a track candidate, just without metadata
    assert candidate(conn, f"track:mbid:{MBID_X}", "listenbrainz_cf") is not None

    # metadata resolved in one batch with the right inc
    meta_calls = [c for c in seen if c[1] == "/1/metadata/recording/"]
    assert len(meta_calls) == 1
    body = json.loads(meta_calls[0][3])
    assert sorted(body["recording_mbids"]) == sorted([MBID_A, MBID_B, MBID_C, MBID_X])
    assert body["inc"] == "artist release"

    # token flows as an Authorization header on every call
    assert all(c[2].get("authorization") == "Token sekrit" for c in seen)


def test_cf_groups_tracks_into_album_candidates(conn, cfg, monkeypatch):
    install(monkeypatch, make_handler([]))

    stats = listenbrainz.sync(conn, cfg)

    assert stats["cf_albums"] == 2
    # tracks A (0.9) and B (0.7) share RELEASE_1: the album takes the max
    row = candidate(conn, f"album:mbid:{RELEASE_1}", "listenbrainz_cf_album")
    assert row["external_score"] == pytest.approx(0.9)
    item = conn.execute(
        "SELECT * FROM items WHERE id=?", (f"album:mbid:{RELEASE_1}",)).fetchone()
    assert item["domain"] == "album" and item["title"] == "Album A"
    assert json.loads(item["ids"]) == {"mbid": RELEASE_1, "artist_mbid": ARTIST_1}
    meta = json.loads(item["meta"])
    assert meta["artist"] == "Artist One" and meta["artist_mbid"] == ARTIST_1

    # release whose artist has no mbid still becomes an album, sans artist_mbid
    row2 = candidate(conn, f"album:mbid:{RELEASE_2}", "listenbrainz_cf_album")
    assert row2["external_score"] == pytest.approx(0.5)
    item2 = conn.execute(
        "SELECT * FROM items WHERE id=?", (f"album:mbid:{RELEASE_2}",)).fetchone()
    assert item2["title"] == "Album C"
    assert json.loads(item2["ids"]) == {"mbid": RELEASE_2}
    assert json.loads(item2["meta"]) == {"artist": "The Unknowns"}

    # metadata-less track X grew no album row: exactly the two above exist
    n = conn.execute(
        "SELECT COUNT(*) c FROM candidates WHERE source='listenbrainz_cf_album'"
    ).fetchone()["c"]
    assert n == 2


def test_weekly_exploration_fetches_newest_playlist(conn, cfg, monkeypatch):
    seen: list = []
    install(monkeypatch, make_handler(seen))

    stats = listenbrainz.sync(conn, cfg)

    assert stats["weekly_tracks"] == 2
    playlist_calls = [c[1] for c in seen if c[1].startswith("/1/playlists/")]
    assert playlist_calls == [f"/1/playlists/{PL_NEW}"]

    row = candidate(conn, f"track:mbid:{MBID_W}", "listenbrainz_weekly")
    assert row is not None
    item = conn.execute("SELECT * FROM items WHERE id=?", (f"track:mbid:{MBID_W}",)).fetchone()
    assert item["title"] == "Song W"
    fb_artist = conn.execute(
        "SELECT * FROM items WHERE id=?", ("artist:lastfm:artist two",)).fetchone()
    assert fb_artist["title"] == "Artist Two"

    # a track found by both CF and the playlist keeps both provenance rows
    sources = {r["source"] for r in conn.execute(
        "SELECT source FROM candidates WHERE item_id=?", (f"track:mbid:{MBID_A}",))}
    assert sources == {"listenbrainz_cf", "listenbrainz_weekly"}


def test_cf_204_not_ready(conn, tmp_path, monkeypatch):
    seen: list = []
    install(monkeypatch, make_handler(seen, cf_status=204))
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "listenbrainz": {"user": "alice"}})

    stats = listenbrainz.sync(conn, cfg)

    assert stats["cf"] == "not_ready"
    assert stats["cf_tracks"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"] == 0
    # no token configured → no Authorization header sent
    assert all("authorization" not in c[2] for c in seen)


def test_rerun_is_idempotent_and_refreshes_last_seen(conn, cfg, monkeypatch):
    install(monkeypatch, make_handler([]))

    listenbrainz.sync(conn, cfg)
    n_candidates = conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"]
    n_items = conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
    assert n_candidates == 10  # 4 cf tracks + 2 cf artists + 2 cf albums + 2 weekly tracks
    conn.execute("UPDATE candidates SET last_seen='2020-01-01T00:00:00Z',"
                 " first_seen='2020-01-01T00:00:00Z'")

    stats = listenbrainz.sync(conn, cfg)

    assert stats["cf_tracks"] == 4
    assert conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"] == n_candidates
    assert conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"] == n_items
    stale = conn.execute(
        "SELECT COUNT(*) c FROM candidates WHERE last_seen='2020-01-01T00:00:00Z'").fetchone()["c"]
    assert stale == 0
    kept = conn.execute(
        "SELECT COUNT(*) c FROM candidates WHERE first_seen='2020-01-01T00:00:00Z'").fetchone()["c"]
    assert kept == n_candidates  # conflict update must not touch first_seen


def test_missing_user_skips(conn, tmp_path):
    cfg = C._build({"core": {"data_dir": str(tmp_path)}})
    assert listenbrainz.sync(conn, cfg) == {"skipped": "listenbrainz not configured"}
