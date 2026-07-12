"""Offline tests for the Last.fm collector (httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from gustarr import config as C
from gustarr import db
from gustarr import http as ghttp
from gustarr.collect import lastfm

RH_MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
PA_MBID = "0d5b91d1-aaaa-bbbb-cccc-1234567890ab"
UTS_NEW, UTS_MID, UTS_OLD, UTS_LOVE = 1700000100, 1700000000, 1699999000, 1690000000


def rt(artist, amb, name, tmb, album, uts):
    return {
        "artist": {"name": artist, "mbid": amb, "url": ""},
        "name": name,
        "mbid": tmb,
        "album": {"#text": album, "mbid": ""},
        "date": {"uts": str(uts)},
    }


NOWPLAYING = {
    "artist": {"name": "Radiohead", "mbid": RH_MBID},
    "name": "Let Down",
    "mbid": "",
    "album": {"#text": "OK Computer", "mbid": ""},
    "@attr": {"nowplaying": "true"},
}

RT_PAGES = {
    1: {"recenttracks": {
        "track": [
            NOWPLAYING,
            rt("Radiohead", RH_MBID, "Paranoid Android", PA_MBID, "OK Computer", UTS_NEW),
            rt("Weird Local Band", "", "Some Song", "", "Demo", UTS_MID),
        ],
        "@attr": {"page": "1", "totalPages": "2", "total": "3"},
    }},
    2: {"recenttracks": {
        "track": [rt("Radiohead", RH_MBID, "Karma Police", "", "OK Computer", UTS_OLD)],
        "@attr": {"page": "2", "totalPages": "2", "total": "3"},
    }},
}

RT_EMPTY = {"recenttracks": {"track": [], "@attr": {"page": "1", "totalPages": "0", "total": "0"}}}

LOVED = {"lovedtracks": {
    "track": [{
        "artist": {"name": "Radiohead", "mbid": RH_MBID},
        "name": "Paranoid Android",
        "mbid": PA_MBID,
        "date": {"uts": str(UTS_LOVE)},
    }],
    "@attr": {"page": "1", "totalPages": "1", "total": "1"},
}}

LOVED_EMPTY = {"lovedtracks": {"track": [], "@attr": {"page": "1", "totalPages": "0"}}}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ghttp, "HOST_DELAYS", {})


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    return C._build({"core": {"data_dir": str(tmp_path)},
                     "lastfm": {"api_key": "k", "user": "u"}})


def make_transport(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        p = dict(request.url.params)
        calls.append(p)
        assert p["api_key"] == "k" and p["user"] == "u" and p["format"] == "json"
        if p["method"] == "user.getrecenttracks":
            if int(p.get("from", 0)) > UTS_NEW:
                return httpx.Response(200, json=RT_EMPTY)
            return httpx.Response(200, json=RT_PAGES[int(p.get("page", 1))])
        if p["method"] == "user.getlovedtracks":
            return httpx.Response(200, json=LOVED)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def all_events(conn):
    return conn.execute(
        "SELECT ts, item_id, kind, source FROM events ORDER BY ts, item_id, kind").fetchall()


def test_first_sync_items_events_cursor(conn, cfg):
    calls = []
    stats = lastfm.sync(conn, cfg, transport=make_transport(calls))

    assert stats == {"scrobbles": 3, "loved": 1, "items": 5, "pages": 3,
                     "profiles": 1, "profiles_skipped": 0}

    items = {r["id"]: r for r in conn.execute("SELECT * FROM items")}
    assert set(items) == {
        f"artist:mbid:{RH_MBID}",
        "artist:lastfm:weird local band",
        f"track:mbid:{PA_MBID}",
        "track:lastfm:weird local band\x1fsome song",
        "track:lastfm:radiohead\x1fkarma police",
    }
    pa = items[f"track:mbid:{PA_MBID}"]
    assert pa["title"] == "Paranoid Android"
    assert json.loads(pa["ids"]) == {"mbid": PA_MBID}
    assert json.loads(pa["meta"]) == {
        "artist": "Radiohead",
        "artist_id": f"artist:mbid:{RH_MBID}",
        "album": "OK Computer",
    }
    fallback = items["track:lastfm:weird local band\x1fsome song"]
    assert json.loads(fallback["meta"])["artist_id"] == "artist:lastfm:weird local band"
    assert json.loads(fallback["ids"]) == {}

    events = all_events(conn)
    # 3 scrobbles + 1 loved, each mirrored onto the artist item
    assert len(events) == 8
    assert all(e["source"] == "lastfm" for e in events)
    # legacy single-user config: everything lands on the synthesized default
    assert {r["profile"] for r in conn.execute("SELECT profile FROM events")} == {"default"}
    kinds = {(e["item_id"], e["kind"]) for e in events}
    assert (f"track:mbid:{PA_MBID}", "scrobble") in kinds
    assert (f"artist:mbid:{RH_MBID}", "scrobble") in kinds
    assert (f"track:mbid:{PA_MBID}", "loved") in kinds
    assert (f"artist:mbid:{RH_MBID}", "loved") in kinds
    assert ("artist:lastfm:weird local band", "scrobble") in kinds
    # nowplaying row produced no event
    assert not any("let down" in e["item_id"] for e in events)
    scrobble_ts = {e["ts"] for e in events if e["kind"] == "scrobble"}
    assert "2023-11-14T22:15:00Z" in scrobble_ts  # UTS_NEW as ISO

    assert db.pget_state(conn, "default", lastfm.CURSOR_KEY) == str(UTS_NEW)


def test_second_sync_uses_cursor_and_adds_nothing(conn, cfg):
    lastfm.sync(conn, cfg, transport=make_transport([]))
    before = all_events(conn)

    calls = []
    stats = lastfm.sync(conn, cfg, transport=make_transport(calls))

    recent_calls = [c for c in calls if c["method"] == "user.getrecenttracks"]
    assert recent_calls[0]["from"] == str(UTS_NEW + 1)
    assert stats["scrobbles"] == 0 and stats["loved"] == 0
    assert all_events(conn) == before
    assert db.pget_state(conn, "default", lastfm.CURSOR_KEY) == str(UTS_NEW)


def test_full_rewalks_but_stays_idempotent(conn, cfg):
    lastfm.sync(conn, cfg, transport=make_transport([]))
    before = all_events(conn)

    calls = []
    stats = lastfm.sync(conn, cfg, full=True, transport=make_transport(calls))

    recent_calls = [c for c in calls if c["method"] == "user.getrecenttracks"]
    assert len(recent_calls) == 2  # both pages re-walked, no from cursor
    assert all("from" not in c for c in recent_calls)
    assert stats["scrobbles"] == 0 and stats["loved"] == 0
    assert all_events(conn) == before
    assert db.pget_state(conn, "default", lastfm.CURSOR_KEY) == str(UTS_NEW)


def test_empty_history(conn, cfg):
    def handler(request):
        p = dict(request.url.params)
        body = RT_EMPTY if p["method"] == "user.getrecenttracks" else LOVED_EMPTY
        return httpx.Response(200, json=body)

    stats = lastfm.sync(conn, cfg, transport=httpx.MockTransport(handler))
    assert stats == {"scrobbles": 0, "loved": 0, "items": 0, "pages": 2,
                     "profiles": 1, "profiles_skipped": 0}
    assert db.pget_state(conn, "default", lastfm.CURSOR_KEY) is None
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == 0


def test_sync_skipped_unless_api_key_and_user(conn, tmp_path):
    def handler(request):
        raise AssertionError("unconfigured sync must not touch the network")

    transport = httpx.MockTransport(handler)
    for section in ({"api_key": "k"}, {"user": "u"}, {}):
        partial = C._build({"core": {"data_dir": str(tmp_path)}, "lastfm": section})
        stats = lastfm.sync(conn, partial, transport=transport)
        assert stats == {"skipped": "lastfm not fully configured"}
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == 0


def test_same_second_scrobbles_of_different_tracks_both_count(conn, cfg):
    def two_at(uts, box):
        return {box: {
            "track": [
                rt("Radiohead", RH_MBID, "Airbag", "", "OK Computer", uts),
                rt("Radiohead", RH_MBID, "Paranoid Android", PA_MBID, "OK Computer", uts),
            ],
            "@attr": {"page": "1", "totalPages": "1", "total": "2"},
        }}

    def handler(request):
        p = dict(request.url.params)
        if p["method"] == "user.getrecenttracks":
            if int(p.get("from", 0)) > UTS_NEW:
                return httpx.Response(200, json=RT_EMPTY)
            return httpx.Response(200, json=two_at(UTS_NEW, "recenttracks"))
        return httpx.Response(200, json=two_at(UTS_LOVE, "lovedtracks"))

    transport = httpx.MockTransport(handler)
    stats = lastfm.sync(conn, cfg, transport=transport)
    assert stats["scrobbles"] == 2 and stats["loved"] == 2

    # both tracks' plays survive on the artist item, discriminated by dedup
    by_kind: dict[str, set[str]] = {}
    for e in conn.execute("SELECT kind, dedup FROM events WHERE item_id=?",
                          (f"artist:mbid:{RH_MBID}",)):
        by_kind.setdefault(e["kind"], set()).add(e["dedup"])
    expected = {"track:lastfm:radiohead\x1fairbag", f"track:mbid:{PA_MBID}"}
    assert by_kind["scrobble"] == expected
    assert by_kind["loved"] == expected

    # a full re-walk of the identical rows stays idempotent
    before = all_events(conn)
    stats2 = lastfm.sync(conn, cfg, full=True, transport=transport)
    assert stats2["scrobbles"] == 0 and stats2["loved"] == 0
    assert all_events(conn) == before


def test_mbid_scrobble_merges_existing_name_keyed_artist(conn, cfg):
    def one_track_page(*rows):
        return {"recenttracks": {
            "track": list(rows),
            "@attr": {"page": "1", "totalPages": "1", "total": str(len(rows))},
        }}

    old_row = rt("Radiohead", "", "Karma Police", "", "OK Computer", UTS_OLD)
    new_row = rt("Radiohead", RH_MBID, "Paranoid Android", PA_MBID, "OK Computer", UTS_NEW)
    phase = {"page": one_track_page(old_row)}

    def handler(request):
        p = dict(request.url.params)
        if p["method"] == "user.getrecenttracks":
            if int(p.get("from", 0)) > UTS_NEW:
                return httpx.Response(200, json=RT_EMPTY)
            return httpx.Response(200, json=phase["page"])
        return httpx.Response(200, json=LOVED_EMPTY)

    transport = httpx.MockTransport(handler)
    fallback, canonical = "artist:lastfm:radiohead", f"artist:mbid:{RH_MBID}"

    lastfm.sync(conn, cfg, transport=transport)  # mbid-less row mints the fallback
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (fallback,)).fetchone()

    # a later row carrying the mbid folds the fallback into the mbid item
    phase["page"] = one_track_page(new_row)
    lastfm.sync(conn, cfg, transport=transport)
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (fallback,)).fetchone() is None
    assert db.canonical_id(conn, fallback) == canonical
    moved = conn.execute(
        "SELECT ts FROM events WHERE item_id=? AND kind='scrobble'", (canonical,)).fetchall()
    assert len(moved) == 2  # the fallback's event moved onto the mbid item

    # full re-walk re-mints the name-keyed id; the alias redirects it, so
    # nothing is resurrected and no event duplicates
    phase["page"] = one_track_page(new_row, old_row)
    before = all_events(conn)
    stats = lastfm.sync(conn, cfg, full=True, transport=transport)
    assert stats["scrobbles"] == 0
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (fallback,)).fetchone() is None
    assert all_events(conn) == before


def test_page_cap_warns(conn, cfg, monkeypatch):
    monkeypatch.setattr(lastfm, "MAX_PAGES", 2)
    huge = {"recenttracks": {
        "track": [rt("Radiohead", RH_MBID, "Karma Police", "", "OK Computer", UTS_OLD)],
        "@attr": {"page": "1", "totalPages": "9999"},
    }}
    calls = []

    def handler(request):
        p = dict(request.url.params)
        calls.append(p)
        if p["method"] == "user.getrecenttracks":
            return httpx.Response(200, json=huge)
        return httpx.Response(200, json=LOVED_EMPTY)

    stats = lastfm.sync(conn, cfg, transport=httpx.MockTransport(handler))
    recent_calls = [c for c in calls if c["method"] == "user.getrecenttracks"]
    assert len(recent_calls) == 2
    assert "capped" in stats["warning"]
