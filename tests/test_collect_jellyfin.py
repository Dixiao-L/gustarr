"""Offline Jellyfin collector tests against an httpx.MockTransport fake server."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from gustarr import config as C
from gustarr import db
from gustarr import http as ghttp
from gustarr.collect import jellyfin
from gustarr.signals import WEIGHTS

BASE = "http://jelly.test"
MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
ARTIST_MBID = f"artist:mbid:{MBID}"
ARTIST_LASTFM = "artist:lastfm:boards of canada"


@pytest.fixture(autouse=True)
def _no_politeness(monkeypatch):
    monkeypatch.setattr(ghttp, "HOST_DELAYS", {})


def make_cfg(tmp_path, user="tab"):
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "jellyfin": {"url": BASE, "api_key": "sekrit", "user": user},
    })


def server_state():
    return {
        "users": [{"Name": "Tab", "Id": "u1"}, {"Name": "guest", "Id": "u2"}],
        "library": [
            {"Id": "m1", "Type": "Movie", "Name": "The Matrix", "ProductionYear": 1999,
             "ProviderIds": {"Tmdb": "603", "Imdb": "tt0133093"},
             "UserData": {"Played": True, "IsFavorite": True, "PlayCount": 1,
                          "LastPlayedDate": "2024-05-01T20:00:00.0000000Z"}},
            {"Id": "m2", "Type": "Movie", "Name": "Obscure Film",
             "ProviderIds": {"Imdb": "tt0000001"},
             "UserData": {"Played": False, "IsFavorite": False}},
            {"Id": "s1", "Type": "Series", "Name": "Severance", "ProductionYear": 2022,
             "ProviderIds": {"Tvdb": "371980"},
             "UserData": {"Played": True, "IsFavorite": False}},
            {"Id": "a1", "Type": "MusicArtist", "Name": "Radiohead",
             "ProviderIds": {"MusicBrainzArtist": MBID}, "UserData": {}},
            {"Id": "a2", "Type": "MusicArtist", "Name": "Boards of Canada",
             "ProviderIds": {}, "UserData": {}},
            {"Id": "x1", "Type": "Movie", "Name": "", "ProviderIds": {}, "UserData": {}},
        ],
        "episodes": [
            {"Id": "e1", "Type": "Episode", "SeriesId": "s1"},
            {"Id": "e2", "Type": "Episode", "SeriesId": "s1"},
            {"Id": "e3", "Type": "Episode", "SeriesId": "s1"},
        ],
        "by_id": {
            "s1": {"Id": "s1", "Type": "Series", "Name": "Severance", "ProductionYear": 2022,
                   "ProviderIds": {"Tvdb": "371980"}, "RecursiveItemCount": 5},
            "a1": {"Id": "a1", "Type": "MusicArtist", "Name": "Radiohead",
                   "ProviderIds": {"MusicBrainzArtist": MBID}},
            "a2": {"Id": "a2", "Type": "MusicArtist", "Name": "Boards of Canada",
                   "ProviderIds": {}},
        },
        "audio": [
            {"Id": "t1", "Type": "Audio", "Name": "Paranoid Android", "Album": "OK Computer",
             "ArtistItems": [{"Id": "a1", "Name": "Radiohead"}],
             "UserData": {"PlayCount": 7, "LastPlayedDate": "2024-06-01T10:00:00.0000000Z"}},
            {"Id": "t2", "Type": "Audio", "Name": "Roygbiv", "Album": "MHTRTC",
             "ArtistItems": [{"Id": "a2", "Name": "Boards of Canada"}],
             "UserData": {"PlayCount": 1, "LastPlayedDate": "2024-06-02T10:00:00.0000000Z"}},
        ],
        "pbr_rows": [[1, "2024-06-01 10:00:00", "u1", "t1", "Audio", 240],
                     [2, "2024-06-02 20:00:00", "u1", "m1", "Movie", 8160]],
        "pbr_available": True,
    }


def make_transport(state):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Emby-Token") == "sekrit"
        path = request.url.path
        q = dict(request.url.params)
        if path == "/Users":
            return httpx.Response(200, json=state["users"])
        if path == "/user_usage_stats/submit_custom_query":
            if not state["pbr_available"]:
                return httpx.Response(404, text="plugin not installed")
            body = json.loads(request.content)
            cur = int(re.search(r"rowid > (\d+)", body["CustomQueryString"]).group(1))
            rows = [r for r in state["pbr_rows"] if r[0] > cur]
            cols = ["rowid", "DateCreated", "UserId", "ItemId", "ItemType", "PlayDuration"]
            return httpx.Response(200, json={"colums": cols, "results": rows})
        if path == "/Users/u1/Items":
            if "Ids" in q:
                found = [state["by_id"][i] for i in q["Ids"].split(",") if i in state["by_id"]]
                return httpx.Response(200, json={"Items": found,
                                                 "TotalRecordCount": len(found)})
            inc = q.get("IncludeItemTypes", "")
            if "Episode" in inc:
                data = state["episodes"]
            elif "Audio" in inc:
                data = state["audio"]
            else:
                data = state["library"]
            start = int(q.get("StartIndex", 0))
            page = data[start:start + 2]  # tiny pages force the pagination loop
            return httpx.Response(200, json={"Items": page, "TotalRecordCount": len(data)})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_sync_maps_canonical_ids_and_events(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))

    row = conn.execute("SELECT * FROM items WHERE id='movie:tmdb:603'").fetchone()
    assert row["title"] == "The Matrix" and row["year"] == 1999
    assert json.loads(row["ids"]) == {"tmdb": 603, "imdb": "tt0133093"}
    assert json.loads(row["meta"])["jellyfin_id"] == "m1"
    have = {r["id"] for r in conn.execute("SELECT id FROM items")}
    assert {"movie:imdb:tt0000001", "series:tvdb:371980", ARTIST_MBID, ARTIST_LASTFM} <= have

    ev = {(r["item_id"], r["kind"]): r for r in conn.execute("SELECT * FROM events")}
    assert ev[("movie:tmdb:603", "favorite")]["weight"] == WEIGHTS["favorite"]
    assert ev[("movie:tmdb:603", "complete")]["weight"] == WEIGHTS["complete"]
    assert ev[("movie:tmdb:603", "complete")]["ts"] == "2024-05-01T20:00:00Z"
    assert all(r["source"] == "jellyfin" for r in ev.values())
    # a Played series must not yield complete from the library pass (3/5 < 80%)
    assert ("series:tvdb:371980", "complete") not in ev
    assert ev[("series:tvdb:371980", "play")]["weight"] == WEIGHTS["play"]
    assert ev[(ARTIST_MBID, "scrobble")]["weight"] == pytest.approx(5 * WEIGHTS["scrobble"])
    assert json.loads(ev[(ARTIST_MBID, "scrobble")]["meta"])["delta"] == 7
    assert ev[(ARTIST_LASTFM, "scrobble")]["weight"] == pytest.approx(WEIGHTS["scrobble"])

    assert stats["items"] == 5 and stats["skipped"] == 1
    assert stats["favorites"] == 1 and stats["completes"] == 1
    assert stats["series_plays"] == 1 and stats["series_completes"] == 0
    assert stats["scrobbles"] == 2
    assert stats["playback_reporting"] == 2
    assert db.get_state(conn, "jellyfin:pbr_rowid") == "2"


def test_second_sync_is_a_noop(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    transport = make_transport(server_state())
    jellyfin.sync(conn, cfg, transport=transport)
    before = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]

    stats = jellyfin.sync(conn, cfg, transport=transport)
    after = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
    assert after == before
    assert stats["favorites"] == stats["completes"] == 0
    assert stats["series_plays"] == stats["series_completes"] == stats["scrobbles"] == 0
    assert stats["playback_reporting"] == 0


def test_progress_deltas_emit_incrementally(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # incremental count-delta emission is the FALLBACK path (plugin
    # absent); with PBR active these deltas become cursor maintenance
    state["pbr_available"] = False
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)

    # one more episode (4/5 >= 80%) and two more listens of t1
    state["episodes"].append({"Id": "e4", "Type": "Episode", "SeriesId": "s1"})
    state["audio"][0]["UserData"] = {"PlayCount": 9,
                                     "LastPlayedDate": "2024-06-15T10:00:00.0000000Z"}
    # both syncs run within the same wall-clock second; distinct 'now' keeps
    # the (ts,item,kind,source) uniqueness key from swallowing the new play
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 1 and stats["series_completes"] == 1
    assert stats["scrobbles"] == 1
    row = conn.execute(
        "SELECT weight, meta FROM events WHERE kind='scrobble' AND ts='2024-06-15T10:00:00Z'"
    ).fetchone()
    assert row["weight"] == pytest.approx(2 * WEIGHTS["scrobble"])
    assert json.loads(row["meta"])["delta"] == 2
    complete = conn.execute(
        "SELECT meta FROM events WHERE kind='complete' AND item_id='series:tvdb:371980'"
    ).fetchone()
    assert json.loads(complete["meta"]) == {"episodes_played": 4, "episodes_total": 5}

    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-02T00:00:00Z")
    stats3 = jellyfin.sync(conn, cfg, transport=transport)
    assert stats3["series_plays"] == stats3["series_completes"] == stats3["scrobbles"] == 0


def test_same_second_tracks_by_one_artist_both_scrobble(tmp_path):
    # marking an album played stamps every track with the same LastPlayedDate;
    # the track-id dedup keeps the artist rollup from swallowing all but one
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    same_ts = "2024-06-03T09:00:00.0000000Z"
    state["audio"] = [
        {"Id": "t1", "Type": "Audio", "Name": "Paranoid Android", "Album": "OK Computer",
         "ArtistItems": [{"Id": "a1", "Name": "Radiohead"}],
         "UserData": {"PlayCount": 1, "LastPlayedDate": same_ts}},
        {"Id": "t3", "Type": "Audio", "Name": "Karma Police", "Album": "OK Computer",
         "ArtistItems": [{"Id": "a1", "Name": "Radiohead"}],
         "UserData": {"PlayCount": 1, "LastPlayedDate": same_ts}},
    ]
    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))
    assert stats["scrobbles"] == 2
    rows = conn.execute(
        "SELECT dedup, weight FROM events WHERE item_id=? AND kind='scrobble'",
        (ARTIST_MBID,)).fetchall()
    assert {r["dedup"] for r in rows} == {"t1", "t3"}
    assert sum(r["weight"] for r in rows) == pytest.approx(2 * WEIGHTS["scrobble"])
    # both cursors advanced, so the re-sync stays a noop
    stats2 = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))
    assert stats2["scrobbles"] == 0


def test_cursor_holds_when_duplicate_key_blocks_scrobble(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # incremental count-delta emission is the FALLBACK path (plugin
    # absent); with PBR active these deltas become cursor maintenance
    state["pbr_available"] = False
    jellyfin.sync(conn, cfg, transport=make_transport(state))
    # PlayCount rises but the server never updates LastPlayedDate: the insert
    # collides with the stored row, so the cursor must not swallow the plays
    state["audio"][0]["UserData"]["PlayCount"] = 9
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats["scrobbles"] == 0
    assert db.get_state(conn, "jellyfin:track_plays:t1") == "7"
    # once the ts finally moves, the pending delta lands intact
    state["audio"][0]["UserData"]["LastPlayedDate"] = "2024-06-20T10:00:00.0000000Z"
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats["scrobbles"] == 1
    row = conn.execute(
        "SELECT weight, meta FROM events WHERE ts='2024-06-20T10:00:00Z'").fetchone()
    assert row["weight"] == pytest.approx(2 * WEIGHTS["scrobble"])
    assert json.loads(row["meta"])["delta"] == 2
    assert db.get_state(conn, "jellyfin:track_plays:t1") == "9"


def test_merged_fallback_artist_is_not_resurrected(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # favorited, never played: the flag ts falls back to db.now(), which
    # drifts between syncs — exactly the case that used to duplicate forever
    state["library"][4]["UserData"] = {"IsFavorite": True}
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)
    canon = "artist:mbid:9578c268-fe14-4ec7-a390-85aa71d38afa"
    db.merge_item(conn, ARTIST_LASTFM, canon)

    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["favorites"] == 0 and stats["scrobbles"] == 0
    dead = conn.execute("SELECT COUNT(*) c FROM items WHERE id=?", (ARTIST_LASTFM,)).fetchone()
    assert dead["c"] == 0
    orphans = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE item_id=?", (ARTIST_LASTFM,)).fetchone()
    assert orphans["c"] == 0
    kinds = {r["kind"]: r["c"] for r in conn.execute(
        "SELECT kind, COUNT(*) c FROM events WHERE item_id=? GROUP BY kind", (canon,))}
    assert kinds == {"favorite": 1, "scrobble": 1}
    # the re-sync still refreshed the canonical row's metadata
    row = conn.execute("SELECT title FROM items WHERE id=?", (canon,)).fetchone()
    assert row["title"] == "Boards of Canada"


def test_series_cursors_follow_merged_id(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # incremental count-delta emission is the FALLBACK path (plugin
    # absent); with PBR active these deltas become cursor maintenance
    state["pbr_available"] = False
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)
    db.merge_item(conn, "series:tvdb:371980", "series:tmdb:999")

    # unchanged progress after the merge must not re-emit play/complete
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 0 and stats["series_completes"] == 0
    orphans = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE item_id='series:tvdb:371980'").fetchone()
    assert orphans["c"] == 0

    # new progress lands on the canonical id under the migrated cursor keys
    state["episodes"] += [{"Id": "e4", "Type": "Episode", "SeriesId": "s1"},
                          {"Id": "e5", "Type": "Episode", "SeriesId": "s1"}]
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-02T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 1 and stats["series_completes"] == 1
    kinds = sorted(r["kind"] for r in conn.execute(
        "SELECT kind FROM events WHERE item_id='series:tmdb:999' AND ts LIKE '2099-01-02%'"))
    assert kinds == ["complete", "play"]
    assert db.get_state(conn, "jellyfin:series_played:series:tmdb:999") == "5"


def test_missing_user_errors_clearly(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, user="nobody")
    with pytest.raises(ValueError, match="nobody"):
        jellyfin.sync(conn, cfg, transport=make_transport(server_state()))


def test_playback_reporting_absent_is_skipped(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    state["pbr_available"] = False
    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))
    assert stats["playback_reporting"] == "unavailable"
    assert stats["items"] == 5  # the rest of the sync still ran


def test_pbr_rows_become_precise_events(tmp_path):
    """Playback Reporting rows are the precise history: per-play timestamps,
    durations, movie completion detection, other-user filtering — while the
    count-delta paths bootstrap pre-plugin backlog and then go quiet."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    state["by_id"]["t1"] = state["audio"][0]
    state["by_id"]["m3"] = {"Id": "m3", "Type": "Movie", "Name": "Heat",
                            "ProviderIds": {"Tmdb": "949"},
                            "RunTimeTicks": 9000 * 10_000_000}
    state["by_id"]["e2"] = {"Id": "e2", "Type": "Episode", "Name": "Ep 2", "SeriesId": "s1"}
    state["pbr_rows"] = [
        [1, "2024-06-01 10:00:00", "u1", "t1", "Audio", 240],
        [2, "2024-06-02 20:00:00", "u1", "m3", "Movie", 8160],   # 8160/9000 >= 0.85
        [3, "2024-06-03 21:00:00", "u1", "e2", "Episode", 1500],
        [4, "2024-06-03 22:00:00", "u2", "t1", "Audio", 240],    # other user: ignored
        [5, "2024-06-04 10:00:00", "u1", "gone", "Movie", 100],  # deleted item: skipped
    ]
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))

    assert stats["pbr_scrobbles"] == 1
    assert stats["pbr_plays"] == 2       # movie m3 + episode->series
    assert stats["pbr_completes"] == 1   # m3 by duration
    rows = {(r["item_id"], r["kind"], r["dedup"]): r
            for r in conn.execute("SELECT * FROM events WHERE source='jellyfin'")}
    art = f"artist:mbid:{MBID}"
    assert rows[(art, "scrobble", "pbr1")]["ts"] == "2024-06-01T10:00:00Z"
    assert rows[("movie:tmdb:949", "play", "pbr2")]["meta"]
    assert ("movie:tmdb:949", "complete", "") in rows
    assert rows[("series:tvdb:371980", "play", "pbr3")]["ts"] == "2024-06-03T21:00:00Z"
    assert not any(d == "pbr4" for _, _, d in rows)  # guest listen filtered
    assert not any(d == "pbr5" for _, _, d in rows)  # deleted item skipped
    # bootstrap: pre-plugin playcounts still emitted once via the count path
    assert stats["scrobbles"] == 2  # t1 + t2 backlogs

    # steady state: nothing new, and count-deltas stay cursor-only
    state["audio"][0]["UserData"]["PlayCount"] = 9
    state["audio"][0]["UserData"]["LastPlayedDate"] = "2024-06-20T10:00:00.0000000Z"
    stats2 = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats2["pbr_scrobbles"] == stats2["scrobbles"] == 0
    assert db.get_state(conn, "jellyfin:track_plays:t1") == "9"
