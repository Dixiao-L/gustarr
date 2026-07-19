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

BASE = "http://jelly.test"
MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"


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
        m = re.fullmatch(r"/Users/([^/]+)/Items", path)
        if m:
            if "Ids" in q:
                found = [state["by_id"][i] for i in q["Ids"].split(",") if i in state["by_id"]]
                return httpx.Response(200, json={"Items": found,
                                                 "TotalRecordCount": len(found)})
            # legacy single-user state keeps u1's view at the top level;
            # multi-profile tests park per-uid views under 'per_user'
            view = state.get("per_user", {}).get(
                m.group(1), state if m.group(1) == "u1" else None)
            if view is None:
                return httpx.Response(404)
            inc = q.get("IncludeItemTypes", "")
            if "Episode" in inc:
                data = view["episodes"]
            elif "Audio" in inc:
                data = view["audio"]
            else:
                data = view["library"]
            start = int(q.get("StartIndex", 0))
            page = data[start:start + 2]  # tiny pages force the pagination loop
            return httpx.Response(200, json={"Items": page, "TotalRecordCount": len(data)})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


# item ids are surrogates: every test recovers them through the identities
# the collector is contractually required to write
def movie(conn):
    return db.lookup_item(conn, "movie", "tmdb", "603")


def series(conn):
    return db.lookup_item(conn, "series", "tvdb", "371980")


def radiohead(conn):
    return db.lookup_item(conn, "artist", "mbid", MBID)


def boc(conn):
    return db.lookup_item(conn, "artist", "name", "Boards of Canada")


def test_sync_maps_items_identities_and_events(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))

    matrix = movie(conn)
    row = conn.execute("SELECT * FROM items WHERE id=?", (matrix,)).fetchone()
    assert row["title"] == "The Matrix" and row["year"] == 1999
    # every name learned in one pass: all provider ids + the jellyfin id
    assert db.identities_of(conn, matrix) == {
        "tmdb": "603", "imdb": "tt0133093", "jellyfin": "m1"}
    assert db.lookup_item(conn, "movie", "imdb", "tt0000001") is not None
    assert None not in {series(conn), radiohead(conn), boc(conn)}
    # an mbid artist also teaches its spelling and its jellyfin id
    assert db.identities_of(conn, radiohead(conn)) == {
        "mbid": MBID, "name": "radiohead", "jellyfin": "a1"}
    # future passes can resolve purely by jellyfin id
    assert db.lookup_item(conn, "series", "jellyfin", "s1") == series(conn)
    assert db.lookup_item(conn, "artist", "jellyfin", "a2") == boc(conn)

    # events carry a scale multiplier, never a frozen weight: flags and
    # single plays store 1, the capped batched listen count is the scale
    ev = {(r["item_id"], r["kind"]): r for r in conn.execute("SELECT * FROM events")}
    assert ev[(matrix, "favorite")]["scale"] == 1.0
    assert ev[(matrix, "complete")]["scale"] == 1.0
    assert ev[(matrix, "complete")]["ts"] == "2024-05-01T20:00:00Z"
    assert all(r["source"] == "jellyfin" for r in ev.values())
    # a Played series must not yield complete from the library pass (3/5 < 80%)
    assert (series(conn), "complete") not in ev
    assert ev[(series(conn), "play")]["scale"] == 1.0
    assert ev[(radiohead(conn), "scrobble")]["scale"] == pytest.approx(5)  # capped at 5 of 7
    assert json.loads(ev[(radiohead(conn), "scrobble")]["meta"])["delta"] == 7
    assert ev[(boc(conn), "scrobble")]["scale"] == pytest.approx(1)

    assert stats["items"] == 5 and stats["skipped"] == 1
    assert stats["favorites"] == 1 and stats["completes"] == 1
    assert stats["series_plays"] == 1 and stats["series_completes"] == 0
    assert stats["scrobbles"] == 2
    assert stats["playback_reporting"] == 2
    assert stats["profiles"] == 1 and stats["profiles_skipped"] == 0
    # legacy single-user config: everything lands on the synthesized default
    assert {r["profile"] for r in conn.execute("SELECT profile FROM events")} == {"default"}
    assert db.pget_state(conn, "default", "jellyfin:pbr_rowid") == "2"
    # progress cursors key on jellyfin ids, not item ids
    assert db.pget_state(conn, "default", "jellyfin:series_played:s1") == "3"


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
        "SELECT scale, meta FROM events WHERE kind='scrobble' AND ts='2024-06-15T10:00:00Z'"
    ).fetchone()
    assert row["scale"] == pytest.approx(2)
    assert json.loads(row["meta"])["delta"] == 2
    complete = conn.execute(
        "SELECT meta FROM events WHERE kind='complete' AND item_id=?",
        (series(conn),)).fetchone()
    assert json.loads(complete["meta"]) == {"episodes_played": 4, "episodes_total": 5}
    assert db.pget_state(conn, "default", "jellyfin:series_played:s1") == "4"

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
        "SELECT dedup, scale FROM events WHERE item_id=? AND kind='scrobble'",
        (radiohead(conn),)).fetchall()
    assert {r["dedup"] for r in rows} == {"t1", "t3"}
    assert sum(r["scale"] for r in rows) == pytest.approx(2)
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
    assert db.pget_state(conn, "default", "jellyfin:track_plays:t1") == "7"
    # once the ts finally moves, the pending delta lands intact
    state["audio"][0]["UserData"]["LastPlayedDate"] = "2024-06-20T10:00:00.0000000Z"
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats["scrobbles"] == 1
    row = conn.execute(
        "SELECT scale, meta FROM events WHERE ts='2024-06-20T10:00:00Z'").fetchone()
    assert row["scale"] == pytest.approx(2)
    assert json.loads(row["meta"])["delta"] == 2
    assert db.pget_state(conn, "default", "jellyfin:track_plays:t1") == "9"


def test_merged_name_artist_is_not_resurrected(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # favorited, never played: the flag ts falls back to db.now(), which
    # drifts between syncs — exactly the case that used to duplicate forever
    state["library"][4]["UserData"] = {"IsFavorite": True}
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)
    loser = boc(conn)
    # enrich learns the artist's mbid: attaching the spelling merges the
    # name-minted item into the mbid holder (the authoritative side wins)
    canon = db.resolve_item(conn, "artist", "mbid", "9578c268-fe14-4ec7-a390-85aa71d38afa")
    assert db.attach_identity(conn, canon, "name", "Boards of Canada") == canon

    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["favorites"] == 0 and stats["scrobbles"] == 0
    dead = conn.execute("SELECT COUNT(*) c FROM items WHERE id=?", (loser,)).fetchone()
    assert dead["c"] == 0
    orphans = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE item_id=?", (loser,)).fetchone()
    assert orphans["c"] == 0
    kinds = {r["kind"]: r["c"] for r in conn.execute(
        "SELECT kind, COUNT(*) c FROM events WHERE item_id=? GROUP BY kind", (canon,))}
    assert kinds == {"favorite": 1, "scrobble": 1}
    # the re-sync resolved through the taught name straight onto the winner
    assert boc(conn) == canon
    row = conn.execute("SELECT title FROM items WHERE id=?", (canon,)).fetchone()
    assert row["title"] == "Boards of Canada"


def test_series_cursor_survives_identity_merge(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    # incremental count-delta emission is the FALLBACK path (plugin
    # absent); with PBR active these deltas become cursor maintenance
    state["pbr_available"] = False
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)
    old = series(conn)
    # enrich discovers the tvdb item was also minted under a tmdb id and
    # merges them: the item id changes, the jellyfin id (the cursor) doesn't
    twin = db.resolve_item(conn, "series", "tmdb", "999")
    db.merge_items(conn, old, twin)

    # unchanged progress after the merge must not re-emit play/complete
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 0 and stats["series_completes"] == 0
    orphans = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE item_id=?", (old,)).fetchone()
    assert orphans["c"] == 0

    # new progress lands on the surviving id under the same jf-id cursor
    state["episodes"] += [{"Id": "e4", "Type": "Episode", "SeriesId": "s1"},
                          {"Id": "e5", "Type": "Episode", "SeriesId": "s1"}]
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-02T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 1 and stats["series_completes"] == 1
    kinds = sorted(r["kind"] for r in conn.execute(
        "SELECT kind FROM events WHERE item_id=? AND ts LIKE '2099-01-02%'", (twin,)))
    assert kinds == ["complete", "play"]
    assert db.pget_state(conn, "default", "jellyfin:series_played:s1") == "5"


def test_upgrade_bootstrap_seeds_cursor_without_reemitting(tmp_path, monkeypatch):
    """A store upgraded from v2 holds play history but no jf-id-keyed series
    cursor: the first walk must seed the cursor silently instead of
    re-emitting the whole backlog as one fresh play."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    state = server_state()
    state["pbr_available"] = False
    transport = make_transport(state)
    jellyfin.sync(conn, cfg, transport=transport)
    before = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
    # simulate the upgrade: history exists, the new-style cursor does not
    conn.execute("DELETE FROM state WHERE key=?",
                 (db.pkey("default", "jellyfin:series_played:s1"),))

    # distinct 'now' so a re-emit would NOT be masked by the uniqueness key
    monkeypatch.setattr(jellyfin.db, "now", lambda: "2099-01-01T00:00:00Z")
    stats = jellyfin.sync(conn, cfg, transport=transport)
    assert stats["series_plays"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == before
    assert db.pget_state(conn, "default", "jellyfin:series_played:s1") == "3"


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
    heat = db.lookup_item(conn, "movie", "tmdb", "949")
    assert rows[(radiohead(conn), "scrobble", "pbr1")]["ts"] == "2024-06-01T10:00:00Z"
    assert rows[(heat, "play", "pbr2")]["meta"]
    assert (heat, "complete", "") in rows
    assert rows[(series(conn), "play", "pbr3")]["ts"] == "2024-06-03T21:00:00Z"
    assert not any(d == "pbr4" for _, _, d in rows)  # guest listen filtered
    assert not any(d == "pbr5" for _, _, d in rows)  # deleted item skipped
    # bootstrap: pre-plugin playcounts still emitted once via the count path
    assert stats["scrobbles"] == 2  # t1 + t2 backlogs
    # ...but the series play PBR just recorded is not double-counted: the
    # count path sees the history and only seeds its cursor
    assert stats["series_plays"] == 0
    assert db.pget_state(conn, "default", "jellyfin:series_played:s1") == "3"

    # steady state: nothing new, and count-deltas stay cursor-only
    state["audio"][0]["UserData"]["PlayCount"] = 9
    state["audio"][0]["UserData"]["LastPlayedDate"] = "2024-06-20T10:00:00.0000000Z"
    stats2 = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats2["pbr_scrobbles"] == stats2["scrobbles"] == 0
    assert db.pget_state(conn, "default", "jellyfin:track_plays:t1") == "9"


def test_stale_artist_credit_chases_merged_survivor(tmp_path):
    """The audio pass caches jf-artist-id → item ints up front; resolving a
    later artist (mbid + the same spelling) merges an earlier name-only one
    away mid-loop. A track credited to the first artist must not crash on
    the stale int — the scrobble chases the jellyfin identity, which the
    merge repointed at the survivor."""
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    state["pbr_available"] = False
    mbid2 = "f22942a1-6f70-4f48-866e-238cb2308fbd"
    state["library"] = []
    state["episodes"] = []
    # a3 carries no mbid; a4 holds one AND the same spelling, so resolving
    # a4 merges a3's freshly-minted name-keyed item into the mbid holder
    state["by_id"] = {
        "a3": {"Id": "a3", "Type": "MusicArtist", "Name": "Aphex Twin",
               "ProviderIds": {}},
        "a4": {"Id": "a4", "Type": "MusicArtist", "Name": "Aphex Twin",
               "ProviderIds": {"MusicBrainzArtist": mbid2}},
    }
    state["audio"] = [
        {"Id": "t8", "Type": "Audio", "Name": "Xtal", "Album": "SAW 85-92",
         "ArtistItems": [{"Id": "a3", "Name": "Aphex Twin"}],
         "UserData": {"PlayCount": 1, "LastPlayedDate": "2024-06-05T10:00:00.0000000Z"}},
        {"Id": "t9", "Type": "Audio", "Name": "Ageispolis", "Album": "SAW 85-92",
         "ArtistItems": [{"Id": "a4", "Name": "Aphex Twin"}],
         "UserData": {"PlayCount": 1, "LastPlayedDate": "2024-06-06T10:00:00.0000000Z"}},
    ]

    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))

    survivor = db.lookup_item(conn, "artist", "mbid", mbid2)
    assert survivor is not None
    # one artist item left; the spelling and BOTH jellyfin ids point at it
    assert conn.execute(
        "SELECT COUNT(*) c FROM items WHERE domain='artist'").fetchone()["c"] == 1
    assert db.lookup_item(conn, "artist", "name", "Aphex Twin") == survivor
    assert db.lookup_item(conn, "artist", "jellyfin", "a3") == survivor
    assert db.lookup_item(conn, "artist", "jellyfin", "a4") == survivor
    # the stale-credited track's scrobble landed on the survivor, none lost
    rows = conn.execute(
        "SELECT item_id, dedup FROM events WHERE kind='scrobble'").fetchall()
    assert {r["dedup"] for r in rows} == {"t8", "t9"}
    assert all(r["item_id"] == survivor for r in rows)
    assert stats["scrobbles"] == 2
    # cursors advanced under the survivor: the re-sync stays a noop
    stats2 = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))
    assert stats2["scrobbles"] == 0


def test_junk_identity_rows_skip_without_failing_stage(tmp_path):
    """Rows the server hands back with unusable identity keys — a Movie
    whose Jellyfin Id is whitespace-only, a MusicArtist named only an
    ideographic space — must never crash the pass: the row resolves or is
    skipped, and every well-formed row around them still lands."""
    conn = db.connect(tmp_path / "t.db")
    state = server_state()
    state["pbr_available"] = False
    state["library"] += [
        {"Id": "   ", "Type": "Movie", "Name": "Junk Id Movie", "ProductionYear": 2001,
         "ProviderIds": {"Tmdb": "6100"}, "UserData": {"Played": True}},
        {"Id": "a9", "Type": "MusicArtist", "Name": "　", "ProviderIds": {},
         "UserData": {}},
    ]
    stats = jellyfin.sync(conn, make_cfg(tmp_path), transport=make_transport(state))

    # both junk rows skipped alongside the provider-tagless x1
    assert stats["skipped"] == 3
    # every well-formed row still processed, all its signals intact
    assert stats["items"] == 5
    assert stats["favorites"] == 1 and stats["completes"] == 1
    assert stats["series_plays"] == 1 and stats["scrobbles"] == 2
    # the ideographic-space artist minted nothing: radiohead + boc only
    assert conn.execute(
        "SELECT COUNT(*) c FROM items WHERE domain='artist'").fetchone()["c"] == 2
    # no identity ever landed on an empty key (a shared '' would fuse items)
    assert conn.execute(
        "SELECT COUNT(*) c FROM identities WHERE key=''").fetchone()["c"] == 0


# ── multi-profile ────────────────────────────────────────────────────


def two_profile_cfg(tmp_path):
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "jellyfin": {"url": BASE, "api_key": "sekrit"},
        "profiles": {"tab": {"jellyfin_user": "tab"},
                     "guest": {"jellyfin_user": "guest"}},
    })


def test_two_profiles_plays_land_in_own_profiles(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = two_profile_cfg(tmp_path)
    state = server_state()
    state["pbr_available"] = False  # count-delta path: per-user UserData drives events
    matrix_raw = {"Id": "m1", "Type": "Movie", "Name": "The Matrix", "ProductionYear": 1999,
                  "ProviderIds": {"Tmdb": "603"}}
    state["per_user"] = {
        "u1": {
            "library": [{**matrix_raw, "UserData": {
                "Played": True, "IsFavorite": True,
                "LastPlayedDate": "2024-05-01T20:00:00.0000000Z"}}],
            "episodes": [],
            "audio": [{"Id": "t1", "Type": "Audio", "Name": "Paranoid Android",
                       "Album": "OK Computer",
                       "ArtistItems": [{"Id": "a1", "Name": "Radiohead"}],
                       "UserData": {"PlayCount": 2,
                                    "LastPlayedDate": "2024-06-01T10:00:00.0000000Z"}}],
        },
        "u2": {
            "library": [{**matrix_raw, "UserData": {
                "Played": True, "IsFavorite": False,
                "LastPlayedDate": "2024-05-02T20:00:00.0000000Z"}}],
            "episodes": [],
            "audio": [{"Id": "t2", "Type": "Audio", "Name": "Roygbiv", "Album": "MHTRTC",
                       "ArtistItems": [{"Id": "a2", "Name": "Boards of Canada"}],
                       "UserData": {"PlayCount": 3,
                                    "LastPlayedDate": "2024-06-02T10:00:00.0000000Z"}}],
        },
    }
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))

    assert stats["profiles"] == 2 and stats["profiles_skipped"] == 0
    assert stats["favorites"] == 1 and stats["completes"] == 2
    assert stats["scrobbles"] == 2

    ev = {(r["profile"], r["item_id"], r["kind"]): r
          for r in conn.execute("SELECT * FROM events")}
    matrix = movie(conn)
    # the shared movie: both watched it, only tab favorited it
    assert ("tab", matrix, "complete") in ev
    assert ("guest", matrix, "complete") in ev
    assert ("tab", matrix, "favorite") in ev
    assert ("guest", matrix, "favorite") not in ev
    # each profile's listens stay its own
    assert ev[("tab", radiohead(conn), "scrobble")]["scale"] == pytest.approx(2)
    assert ev[("guest", boc(conn), "scrobble")]["scale"] == pytest.approx(3)
    assert ("guest", radiohead(conn), "scrobble") not in ev
    assert ("tab", boc(conn), "scrobble") not in ev

    # cursors are per profile, never shared
    assert db.pget_state(conn, "tab", "jellyfin:track_plays:t1") == "2"
    assert db.pget_state(conn, "guest", "jellyfin:track_plays:t2") == "3"
    assert db.pget_state(conn, "guest", "jellyfin:track_plays:t1") is None

    # re-sync stays a noop for both profiles
    stats2 = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats2["favorites"] == stats2["completes"] == stats2["scrobbles"] == 0


def test_pbr_rows_map_to_owning_profile(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = two_profile_cfg(tmp_path)
    state = server_state()
    empty = {"library": [], "episodes": [], "audio": []}
    state["per_user"] = {"u1": dict(empty), "u2": dict(empty)}
    state["by_id"]["t1"] = server_state()["audio"][0]
    state["by_id"]["t2"] = server_state()["audio"][1]
    state["pbr_rows"] = [
        [1, "2024-06-01 10:00:00", "u1", "t1", "Audio", 240],
        [2, "2024-06-02 20:00:00", "u2", "t2", "Audio", 200],
        [3, "2024-06-03 21:00:00", "u9", "t1", "Audio", 99],  # nobody's user: dropped
    ]
    stats = jellyfin.sync(conn, cfg, transport=make_transport(state))

    assert stats["pbr_scrobbles"] == 2
    assert stats["playback_reporting"] == 6  # each profile walks all 3 rows
    scrobbles = {(r["profile"], r["item_id"], r["dedup"]) for r in conn.execute(
        "SELECT profile, item_id, dedup FROM events WHERE kind='scrobble'")}
    assert scrobbles == {("tab", radiohead(conn), "pbr1"), ("guest", boc(conn), "pbr2")}
    # each profile's walk cursor covers the whole table independently
    assert db.pget_state(conn, "tab", "jellyfin:pbr_rowid") == "3"
    assert db.pget_state(conn, "guest", "jellyfin:pbr_rowid") == "3"

    stats2 = jellyfin.sync(conn, cfg, transport=make_transport(state))
    assert stats2["pbr_scrobbles"] == 0 and stats2["playback_reporting"] == 0
