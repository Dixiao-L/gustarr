"""Offline tests for the actuation slice: arr clients, apply, Jellyfin
collections. All HTTP goes through an httpx.MockTransport injected by
monkeypatching gustarr.http.request_json."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gustarr import config as C
from gustarr import db, http, settings, signals
from gustarr.actuate import apply as apply_mod
from gustarr.actuate import arr_client, jellyfin_collections
from gustarr.config import ArrConfig

# ── fakes ────────────────────────────────────────────────────────────


class FakeArr:
    def __init__(self, api_key, profiles, roots, metadata_profiles=None, tags=None):
        self.api_key = api_key
        self.profiles = profiles
        self.roots = roots
        self.metadata_profiles = metadata_profiles or []
        self.tags = list(tags or [])
        self.next_tag_id = 1
        self.posted = []
        self.fail_add = None  # (status_code, body_text)
        # Retry-After: 0 keeps http.py's 5xx retry loop sleepless in tests
        self.fail_headers = {"Retry-After": "0"}
        # lidarr album model: album rows exist only under an added artist
        self.artists = []  # artist rows in the library (with id)
        self.albums = []  # album rows, materialized by artist adds
        self.album_lookup = {}  # album mbid -> lookup metadata
        self.next_album_id = 0
        self.monitor_puts = []  # recorded PUT album/monitor bodies
        self.commands = []  # recorded POST command bodies

    def register_artist(self, foreign_id, name):
        """Pre-existing library artist (added outside gustarr): its album
        rows materialize immediately, as lidarr's own add would do."""
        row = {"id": 900 + len(self.artists), "artistName": name,
               "foreignArtistId": foreign_id}
        self.artists.append(row)
        self._spawn_albums(row)
        return row

    def _spawn_albums(self, artist_row):
        # mirror lidarr: adding an artist creates its album rows; there
        # is no direct "add album" endpoint
        for mbid, meta in self.album_lookup.items():
            if meta["artist_mbid"] == artist_row["foreignArtistId"]:
                self.next_album_id += 1
                self.albums.append({
                    "id": self.next_album_id, "foreignAlbumId": mbid,
                    "title": meta["title"], "artistId": artist_row["id"],
                    "monitored": meta.get("monitored", False)})

    def handle(self, request: httpx.Request, path: str) -> httpx.Response:
        if request.headers.get("X-Api-Key") != self.api_key:
            return httpx.Response(401, text="bad api key")
        if path == "tag":
            if request.method == "GET":
                return httpx.Response(200, json=self.tags)
            tag = {"id": self.next_tag_id, "label": json.loads(request.content)["label"]}
            self.next_tag_id += 1
            self.tags.append(tag)
            return httpx.Response(201, json=tag)
        if path == "qualityprofile":
            return httpx.Response(200, json=self.profiles)
        if path == "metadataprofile":
            return httpx.Response(200, json=self.metadata_profiles)
        if path == "rootfolder":
            return httpx.Response(200, json=self.roots)
        if path == "movie/lookup/tmdb":
            tmdb = request.url.params["tmdbId"]
            return httpx.Response(
                200, json={"title": f"Movie {tmdb}", "tmdbId": int(tmdb), "year": 1999})
        if path == "series/lookup":
            tvdb = request.url.params["term"].split(":", 1)[1]
            return httpx.Response(
                200, json=[{"title": f"Series {tvdb}", "tvdbId": int(tvdb), "year": 2005}])
        if path == "artist/lookup":
            mbid = request.url.params["term"].split(":", 1)[1]
            return httpx.Response(
                200, json=[{"artistName": f"Artist {mbid}", "foreignArtistId": mbid}])
        if path == "artist" and request.method == "GET":
            return httpx.Response(200, json=self.artists)
        if path == "album/lookup":
            mbid = request.url.params["term"].split(":", 1)[1]
            meta = self.album_lookup.get(mbid)
            if meta is None:
                return httpx.Response(200, json=[])
            # the embedded artist resource carries an id only when the
            # artist is already in the library — that id presence is what
            # the client keys the shell-add decision on
            artist = next(
                (a for a in self.artists if a["foreignArtistId"] == meta["artist_mbid"]),
                None) or {"artistName": meta["artist_name"],
                          "foreignArtistId": meta["artist_mbid"]}
            return httpx.Response(200, json=[
                {"title": meta["title"], "foreignAlbumId": mbid, "artist": artist}])
        if path == "album" and request.method == "GET":
            q = request.url.params
            if "artistId" not in q or q.get("includeAllArtistAlbums") != "true":
                return httpx.Response(
                    400, text="artistId and includeAllArtistAlbums=true required")
            wanted = int(q["artistId"])
            return httpx.Response(200, json=[a for a in self.albums if a["artistId"] == wanted])
        if path == "album/monitor" and request.method == "PUT":
            body = json.loads(request.content)
            # strict like lidarr: a monitor toggle without ids is a bug
            if not body.get("albumIds") or not isinstance(body.get("monitored"), bool):
                return httpx.Response(400, text="albumIds and monitored are required")
            self.monitor_puts.append(body)
            for album in self.albums:
                if album["id"] in body["albumIds"]:
                    album["monitored"] = body["monitored"]
            return httpx.Response(202, json=body)
        if path == "command" and request.method == "POST":
            body = json.loads(request.content)
            if not body.get("name"):
                return httpx.Response(400, text="command name required")
            self.commands.append(body)
            return httpx.Response(201, json={"id": len(self.commands), "name": body["name"]})
        if path in ("movie", "series", "artist") and request.method == "POST":
            if self.fail_add:
                status, text = self.fail_add
                return httpx.Response(status, text=text, headers=self.fail_headers)
            body = json.loads(request.content)
            # Mirror Lidarr's server-side validation: a bare id without the
            # lookup resource 400s in production ("'Artist Name' must not
            # be empty") — the earlier lenient mock hid exactly that bug.
            if path == "artist" and not body.get("artistName"):
                return httpx.Response(400, json=[{
                    "propertyName": "ArtistName",
                    "errorMessage": "'Artist Name' must not be empty.",
                    "errorCode": "NotEmptyValidator"}])
            if path == "artist" and any(
                    a["foreignArtistId"] == body.get("foreignArtistId") for a in self.artists):
                return httpx.Response(400, json=[{
                    "errorMessage": "This artist has already been added",
                    "errorCode": "ArtistExistsValidator"}])
            self.posted.append(body)
            row = {"id": 100 + len(self.posted), **body}
            if path == "artist":
                self.artists.append(row)
                self._spawn_albums(row)
            return httpx.Response(201, json=row)
        return httpx.Response(404, text=f"unhandled arr {request.method} {path}")


class FakeJellyfin:
    def __init__(self, api_key="jk"):
        self.api_key = api_key
        self.items = {}  # "tmdb.603" -> jellyfin item id
        self.collections = {}  # id -> {"name": ..., "members": [...]}
        self.next_id = 1

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Emby-Token") != self.api_key:
            return httpx.Response(401, text="bad token")
        path, q = request.url.path, request.url.params
        if path == "/Items":
            if "AnyProviderIdEquals" in q:
                jf_id = self.items.get(q["AnyProviderIdEquals"])
                found = [{"Id": jf_id}] if jf_id else []
                return httpx.Response(200, json={"Items": found})
            if q.get("IncludeItemTypes") == "BoxSet":
                term = q.get("SearchTerm", "").lower()
                found = [{"Id": cid, "Name": c["name"]}
                         for cid, c in self.collections.items() if term in c["name"].lower()]
                return httpx.Response(200, json={"Items": found})
            if "ParentId" in q:
                coll = self.collections.get(q["ParentId"])
                members = coll["members"] if coll else []
                return httpx.Response(200, json={"Items": [{"Id": m} for m in members]})
        if path == "/Collections" and request.method == "POST":
            cid = f"coll{self.next_id}"
            self.next_id += 1
            members = q["Ids"].split(",") if q.get("Ids") else []
            self.collections[cid] = {"name": q["Name"], "members": members}
            return httpx.Response(200, json={"Id": cid})
        if path.startswith("/Collections/") and path.endswith("/Items") \
                and request.method == "POST":
            cid = path.split("/")[2]
            self.collections[cid]["members"].extend(q["Ids"].split(","))
            return httpx.Response(204)
        return httpx.Response(404, text=f"unhandled jellyfin {request.method} {path}")


class FakeNet:
    def __init__(self):
        self.radarr = FakeArr(
            "rk", [{"id": 10, "name": "HD-1080p"}, {"id": 11, "name": "SD"}],
            [{"path": "/movies"}])
        self.sonarr = FakeArr("sk", [{"id": 12, "name": "HD-1080p"}], [{"path": "/tv"}])
        self.lidarr = FakeArr(
            "lk", [{"id": 20, "name": "Standard"}], [{"path": "/music"}],
            metadata_profiles=[{"id": 30, "name": "Standard"}])
        self.jellyfin = FakeJellyfin()
        self.down = set()  # hosts raising transport errors
        self.log = []  # (method, host, path)

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.log.append((request.method, host, request.url.path))
        if host in self.down:
            raise httpx.ConnectError("connection refused")
        if host == "radarr.test":
            return self.radarr.handle(request, request.url.path.removeprefix("/api/v3/"))
        if host == "sonarr.test":
            return self.sonarr.handle(request, request.url.path.removeprefix("/api/v3/"))
        if host == "lidarr.test":
            return self.lidarr.handle(request, request.url.path.removeprefix("/api/v1/"))
        if host == "jellyfin.test":
            return self.jellyfin.handle(request)
        return httpx.Response(404, text=f"no such host {host}")


# ── fixtures / helpers ───────────────────────────────────────────────


@pytest.fixture
def net(monkeypatch):
    monkeypatch.setattr(http, "HOST_DELAYS", {})
    # no-op the retry backoff so transport-error tests stay fast
    monkeypatch.setattr(http.time, "sleep", lambda _s: None)
    fake = FakeNet()
    transport = httpx.MockTransport(fake.handler)
    orig = http.request_json

    def patched(method, url, **kw):
        kw["transport"] = transport
        return orig(method, url, **kw)

    monkeypatch.setattr(http, "request_json", patched)
    return fake


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def make_cfg(tmp_path, profiles=None, **autonomy):
    auto = {"music_mode": "auto", "video_mode": "queue"}
    auto.update(autonomy)
    raw = {
        "core": {"data_dir": str(tmp_path)},
        "jellyfin": {"url": "http://jellyfin.test", "api_key": "jk", "user": "u"},
        "radarr": {"url": "http://radarr.test", "api_key": "rk",
                   "quality_profile": "HD-1080p", "root_folder": "/movies"},
        "sonarr": {"url": "http://sonarr.test", "api_key": "sk",
                   "quality_profile": "HD-1080p", "root_folder": "/tv"},
        "lidarr": {"url": "http://lidarr.test", "api_key": "lk",
                   "quality_profile": "Standard", "root_folder": "/music"},
        "autonomy": auto,
    }
    if profiles:
        raw["profiles"] = {name: {} for name in profiles}
    return C._build(raw)


def add_rec(conn, item_id, domain, title, ids_json, status="proposed",
            score=1.0, acted_at=None, ts=None, profile="default"):
    db.upsert_item(conn, item_id, domain, title=title, ids=ids_json)
    cur = conn.execute(
        "INSERT INTO recommendations"
        " (profile, run_id, ts, domain, item_id, score, why, status, acted_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (profile, "test", ts or db.now(), domain, item_id, score, "{}", status, acted_at))
    return cur.lastrowid


def rec_row(conn, rec_id):
    return conn.execute("SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()


def seed_album(net, mbid, artist_mbid, title, artist_name="Some Artist"):
    net.lidarr.album_lookup[mbid] = {
        "title": title, "artist_mbid": artist_mbid, "artist_name": artist_name}


# ── runtime settings overrides ───────────────────────────────────────


def test_paused_short_circuits_music_and_video(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto")
    prop = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    appr = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                   status="approved")
    stale = add_rec(conn, "movie:tmdb:604", "movie", "Old Prop", {"tmdb": 604},
                    ts="2020-01-01T00:00:00Z")
    settings.set(conn, "paused", True)

    stats = apply_mod.run(conn, cfg)

    assert stats == {"paused": True}
    assert net.log == []  # no HTTP at all, arrs and jellyfin included
    assert rec_row(conn, prop)["status"] == "proposed"
    assert rec_row(conn, appr)["status"] == "approved"
    assert rec_row(conn, stale)["status"] == "proposed"  # not even TTL expiry
    # dry_run short-circuits the same way
    assert apply_mod.run(conn, cfg, dry_run=True) == {"paused": True}
    assert net.log == []

    settings.clear(conn, "paused")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 1
    assert stats["video_added"] == 1
    assert rec_row(conn, stale)["status"] == "expired"


def test_music_weekly_cap_override_honored(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=3)
    r1 = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "artist:mbid:mb2", "artist", "Artist 2", {"mbid": "mb2"}, score=0.8)
    settings.set(conn, "music_max_artists_per_week", 1)
    assert db.get_state(conn, "setting:music_max_artists_per_week") is not None

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 1
    assert stats["music_added"] == 1
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["mb1"]
    assert rec_row(conn, r1)["status"] == "auto_added"
    assert rec_row(conn, r2)["status"] == "proposed"

    settings.clear(conn, "music_max_artists_per_week")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_budget"] == 2  # cfg cap 3 minus the add above
    assert rec_row(conn, r2)["status"] == "auto_added"


def test_music_mode_override_to_queue(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)  # cfg says auto
    rid = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"})
    settings.set(conn, "music_mode", "queue")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert net.lidarr.posted == []
    assert rec_row(conn, rid)["status"] == "proposed"


def test_video_cap_override_honored(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=5)
    top = add_rec(conn, "movie:tmdb:1", "movie", "M1", {"tmdb": 1}, score=0.9)
    low = add_rec(conn, "movie:tmdb:2", "movie", "M2", {"tmdb": 2}, score=0.8)
    settings.set(conn, "video_queue_max_pending", 1)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, top)["status"] == "auto_added"
    assert rec_row(conn, low)["status"] == "proposed"


# ── music autonomy ───────────────────────────────────────────────────


def test_music_weekly_cap_respected(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=3)
    now = db.now()
    last_week = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # two artists already acted this ISO week, one before it
    add_rec(conn, "artist:mbid:aa1", "artist", "Acted One", {"mbid": "aa1"},
            status="auto_added", acted_at=now)
    add_rec(conn, "artist:mbid:aa2", "artist", "Acted Two", {"mbid": "aa2"},
            status="added", acted_at=now)
    add_rec(conn, "artist:mbid:aa3", "artist", "Old Acted", {"mbid": "aa3"},
            status="auto_added", acted_at=last_week)
    r1 = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "artist:mbid:mb2", "artist", "Artist 2", {"mbid": "mb2"}, score=0.8)
    r3 = add_rec(conn, "artist:mbid:mb3", "artist", "Artist 3", {"mbid": "mb3"}, score=0.7)

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 1
    assert stats["music_added"] == 1
    assert len(net.lidarr.posted) == 1
    body = net.lidarr.posted[0]
    assert body["foreignArtistId"] == "mb1"
    assert body["qualityProfileId"] == 20
    assert body["metadataProfileId"] == 30
    assert body["rootFolderPath"] == "/music"
    assert body["addOptions"] == {"monitor": "all", "searchForMissingAlbums": True}
    assert rec_row(conn, r1)["status"] == "auto_added"
    assert rec_row(conn, r1)["acted_at"]
    assert rec_row(conn, r2)["status"] == "proposed"
    assert rec_row(conn, r3)["status"] == "proposed"
    events = conn.execute(
        "SELECT ts, kind, weight, source FROM events WHERE item_id='artist:mbid:mb1'").fetchall()
    # audit trail only: gustarr's own add must not read as user praise
    assert [(e["kind"], e["weight"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]
    assert "auto_add" not in signals.WEIGHTS
    assert signals.aggregate_label([(e["ts"], e["kind"], e["weight"]) for e in events]) == 0.0


def test_music_budget_shared_across_profiles(conn, tmp_path, net):
    """One disk, one Lidarr: what alice's queue already spent this week
    is gone for bob — the acted count ignores profile by design."""
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"], music_max_artists_per_week=2)
    add_rec(conn, "artist:mbid:aa1", "artist", "Alice Spent", {"mbid": "aa1"},
            status="auto_added", acted_at=db.now(), profile="alice")
    b1 = add_rec(conn, "artist:mbid:b1", "artist", "Bob 1", {"mbid": "b1"}, score=0.9,
                 profile="bob")
    b2 = add_rec(conn, "artist:mbid:b2", "artist", "Bob 2", {"mbid": "b2"}, score=0.8,
                 profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 1  # cap 2 minus alice's add, not bob's own 2
    assert stats["music_added"] == 1
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["b1"]
    assert rec_row(conn, b1)["status"] == "auto_added"
    assert rec_row(conn, b2)["status"] == "proposed"
    # the audit event names whose queue the add came from
    ev = conn.execute(
        "SELECT profile, kind FROM events WHERE item_id='artist:mbid:b1'").fetchone()
    assert (ev["profile"], ev["kind"]) == ("bob", "auto_add")


def test_music_queue_mode_leaves_proposed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    r1 = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"})
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert net.lidarr.posted == []
    assert rec_row(conn, r1)["status"] == "proposed"


def test_approved_artist_added_in_queue_mode(conn, tmp_path, net):
    # explicit approval is consent: actuated even when music_mode='queue'
    cfg = make_cfg(tmp_path, music_mode="queue")
    rid = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), "artist:mbid:mb1", "approve", 1.0, "user", {"rec_id": rid})

    stats = apply_mod.run(conn, cfg)

    assert stats["music_added"] == 1
    assert net.lidarr.posted[0]["foreignArtistId"] == "mb1"
    row = rec_row(conn, rid)
    assert row["status"] == "added"
    assert row["acted_at"]
    # no extra taste event: the approve written at approval time is enough
    kinds = [e["kind"] for e in conn.execute(
        "SELECT kind FROM events WHERE item_id='artist:mbid:mb1'")]
    assert kinds == ["approve"]


def test_approved_artist_exempt_from_weekly_budget(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=0)
    rid = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"},
                  status="approved")
    prop = add_rec(conn, "artist:mbid:mb2", "artist", "Artist 2", {"mbid": "mb2"}, score=0.9)
    stats = apply_mod.run(conn, cfg)
    assert stats["music_budget"] == 0
    assert stats["music_added"] == 1  # the approval, not the proposal
    assert rec_row(conn, rid)["status"] == "added"
    assert rec_row(conn, prop)["status"] == "proposed"


def test_approved_artist_without_mbid_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    rid = add_rec(conn, "artist:lastfm:someband", "artist", "Some Band", {},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert net.lidarr.posted == []
    assert any("mbid" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"


def test_approved_artist_survives_lidarr_outage(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    net.lidarr.fail_add = (503, "service unavailable")
    rid = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert any("Artist 1" in e for e in stats["errors"])
    row = rec_row(conn, rid)
    assert row["status"] == "approved"  # retryable, not terminally failed
    assert json.loads(row["why"])["attempts"] == 1
    # arr back up: the same approval lands on the next apply
    net.lidarr.fail_add = None
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_artist_4xx_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    net.lidarr.fail_add = (400, "Invalid foreignArtistId")
    rid = add_rec(conn, "artist:mbid:bogus", "artist", "Bogus", {"mbid": "bogus"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert rec_row(conn, rid)["status"] == "failed"


def test_music_skips_artists_without_mbid(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=1)
    add_rec(conn, "artist:lastfm:someband", "artist", "Some Band", {}, score=0.99)
    r2 = add_rec(conn, "artist:mbid:mb2", "artist", "Artist 2", {"mbid": "mb2"}, score=0.5)
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 1
    assert net.lidarr.posted[0]["foreignArtistId"] == "mb2"
    assert rec_row(conn, r2)["status"] == "auto_added"


def test_lidarr_failure_keeps_proposed_and_counts_attempt(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.lidarr.fail_add = (400, "Invalid foreignArtistId")
    r1 = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"})
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert any("Artist 1" in e for e in stats["errors"])
    row = rec_row(conn, r1)
    assert row["status"] == "proposed"
    assert row["acted_at"] is None
    assert json.loads(row["why"])["attempts"] == 1
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    # second failing run keeps counting
    apply_mod.run(conn, cfg)
    assert json.loads(rec_row(conn, r1)["why"])["attempts"] == 2


# ── album autonomy ───────────────────────────────────────────────────


def test_album_budget_setting_key(conn, tmp_path):
    cfg = make_cfg(tmp_path, music_max_albums_per_week=7)
    assert settings.get(conn, cfg, "music_max_albums_per_week") == 7  # cfg is the default
    assert settings.set(conn, "music_max_albums_per_week", "2") == 2  # string coerced
    assert settings.get(conn, cfg, "music_max_albums_per_week") == 2
    with pytest.raises(ValueError):
        settings.set(conn, "music_max_albums_per_week", -1)
    settings.clear(conn, "music_max_albums_per_week")
    assert settings.get(conn, cfg, "music_max_albums_per_week") == 7


def test_album_weekly_cap_respected(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_albums_per_week=3)
    now = db.now()
    last_week = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # two albums already acted this ISO week, one before it
    add_rec(conn, "album:mbid:done1", "album", "Done One", {"mbid": "done1"},
            status="auto_added", acted_at=now)
    add_rec(conn, "album:mbid:done2", "album", "Done Two", {"mbid": "done2"},
            status="added", acted_at=now)
    add_rec(conn, "album:mbid:done3", "album", "Old Done", {"mbid": "done3"},
            status="auto_added", acted_at=last_week)
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    seed_album(net, "al2", "ar2", "Album 2", "Artist Two")
    r1 = add_rec(conn, "album:mbid:al1", "album", "Album 1",
                 {"mbid": "al1", "artist_mbid": "ar1"}, score=0.9)
    r2 = add_rec(conn, "album:mbid:al2", "album", "Album 2",
                 {"mbid": "al2", "artist_mbid": "ar2"}, score=0.8)

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_budget"] == 1
    assert stats["albums_added"] == 1
    # only the top-scored album landed: exactly one monitored + searched
    album = next(a for a in net.lidarr.albums if a["foreignAlbumId"] == "al1")
    assert album["monitored"] is True
    assert net.lidarr.monitor_puts == [{"albumIds": [album["id"]], "monitored": True}]
    assert net.lidarr.commands == [{"name": "AlbumSearch", "albumIds": [album["id"]]}]
    assert rec_row(conn, r1)["status"] == "auto_added"
    assert rec_row(conn, r1)["acted_at"]
    assert rec_row(conn, r2)["status"] == "proposed"
    # audit trail only: gustarr's own add must not read as user praise
    events = conn.execute(
        "SELECT kind, weight, source FROM events WHERE item_id='album:mbid:al1'").fetchall()
    assert [(e["kind"], e["weight"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]


def test_album_budget_override_honored(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_albums_per_week=5)
    seed_album(net, "al1", "ar1", "Album 1")
    seed_album(net, "al2", "ar2", "Album 2")
    r1 = add_rec(conn, "album:mbid:al1", "album", "Album 1", {"mbid": "al1"}, score=0.9)
    r2 = add_rec(conn, "album:mbid:al2", "album", "Album 2", {"mbid": "al2"}, score=0.8)
    settings.set(conn, "music_max_albums_per_week", 1)

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_budget"] == 1
    assert stats["albums_added"] == 1
    assert rec_row(conn, r1)["status"] == "auto_added"
    assert rec_row(conn, r2)["status"] == "proposed"


def test_album_artist_shell_created_unmonitored_when_absent(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    seed_album(net, "al1", "ar1", "OK Album", "New Artist")
    rid = add_rec(conn, "album:mbid:al1", "album", "OK Album", {"mbid": "al1"})

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_added"] == 1
    # the artist arrives as an unmonitored shell: we want THIS album,
    # never the whole discography
    shell = net.lidarr.posted[0]
    assert shell["foreignArtistId"] == "ar1"
    assert shell["artistName"] == "New Artist"
    assert shell["monitored"] is False
    assert shell["addOptions"] == {"monitor": "none", "searchForMissingAlbums": False}
    assert shell["qualityProfileId"] == 20
    assert shell["metadataProfileId"] == 30
    assert shell["rootFolderPath"] == "/music"
    assert shell["tags"] == [1]
    assert [a["monitored"] for a in net.lidarr.albums] == [True]
    assert rec_row(conn, rid)["status"] == "auto_added"


def test_approved_album_added_in_queue_mode(conn, tmp_path, net):
    # explicit approval is consent: actuated even when music_mode='queue'
    cfg = make_cfg(tmp_path, music_mode="queue")
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    net.lidarr.register_artist("ar1", "Artist One")
    rid = add_rec(conn, "album:mbid:al1", "album", "Album 1", {"mbid": "al1"},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), "album:mbid:al1", "approve", 1.0, "user", {"rec_id": rid})
    prop = add_rec(conn, "album:mbid:al2", "album", "Album 2", {"mbid": "al2"}, score=0.9)

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_added"] == 1
    assert net.lidarr.posted == []  # artist already in the library: no shell add
    assert len(net.lidarr.monitor_puts) == 1
    assert [c["name"] for c in net.lidarr.commands] == ["AlbumSearch"]
    row = rec_row(conn, rid)
    assert row["status"] == "added"
    assert row["acted_at"]
    assert rec_row(conn, prop)["status"] == "proposed"  # queue mode: no auto pick
    # no extra taste event: the approve written at approval time is enough
    kinds = [e["kind"] for e in conn.execute(
        "SELECT kind FROM events WHERE item_id='album:mbid:al1'")]
    assert kinds == ["approve"]


def test_album_already_monitored_is_noop_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    net.lidarr.album_lookup["al1"]["monitored"] = True
    net.lidarr.register_artist("ar1", "Artist One")
    rid = add_rec(conn, "album:mbid:al1", "album", "Album 1", {"mbid": "al1"},
                  status="approved")

    stats = apply_mod.run(conn, cfg)

    # already monitored IS the desired end state: success, nothing re-done
    assert stats["albums_added"] == 1
    assert stats["errors"] == []
    assert net.lidarr.posted == []
    assert net.lidarr.monitor_puts == []
    assert net.lidarr.commands == []
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_album_survives_lidarr_outage(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    net.lidarr.fail_add = (503, "service unavailable")  # the artist-shell add 503s
    rid = add_rec(conn, "album:mbid:al1", "album", "Album 1", {"mbid": "al1"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["albums_added"] == 0
    assert any("Album 1" in e for e in stats["errors"])
    row = rec_row(conn, rid)
    assert row["status"] == "approved"  # retryable, not terminally failed
    assert json.loads(row["why"])["attempts"] == 1
    # lidarr back up: the same approval lands on the next apply
    net.lidarr.fail_add = None
    stats = apply_mod.run(conn, cfg)
    assert stats["albums_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_album_unknown_mbid_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    rid = add_rec(conn, "album:mbid:nope", "album", "Ghost Album", {"mbid": "nope"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["albums_added"] == 0
    assert any("found nothing" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"  # deterministic 4xx-style failure


# ── video approvals ──────────────────────────────────────────────────


def test_approved_movie_added_without_event_duplication(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), "movie:tmdb:603", "approve", 1.0, "user", {"rec_id": rid})

    stats = apply_mod.run(conn, cfg)

    assert stats["video_added"] == 1
    assert len(net.radarr.posted) == 1
    body = net.radarr.posted[0]
    assert body["tmdbId"] == 603
    assert body["qualityProfileId"] == 10
    assert body["rootFolderPath"] == "/movies"
    assert body["monitored"] is True
    assert body["addOptions"] == {"searchForMovie": True}
    row = rec_row(conn, rid)
    assert row["status"] == "added"
    assert row["acted_at"]
    kinds = [e["kind"] for e in conn.execute(
        "SELECT kind FROM events WHERE item_id='movie:tmdb:603'")]
    assert kinds == ["approve"]


def test_approved_series_added_via_sonarr(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "series:tvdb:81189", "series", "Breaking Bad", {"tvdb": 81189},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    body = net.sonarr.posted[0]
    assert body["tvdbId"] == 81189
    assert body["seasonFolder"] is True
    assert body["addOptions"] == {"searchForMissingEpisodes": True}
    assert rec_row(conn, rid)["status"] == "added"


def test_series_with_only_tmdb_id_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "series:tmdb:1396", "series", "Some Show", {"tmdb": 1396},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 1
    assert any("tvdb" in e for e in stats["errors"])
    assert net.sonarr.posted == []
    assert rec_row(conn, rid)["status"] == "failed"


def test_approved_video_survives_arr_outage(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (503, "service unavailable")
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert stats["video_failed"] == 0
    assert any("The Matrix" in e for e in stats["errors"])
    row = rec_row(conn, rid)
    assert row["status"] == "approved"  # retryable, not terminally failed
    assert json.loads(row["why"])["attempts"] == 1
    # arr back up: the approval lands on the next apply
    net.radarr.fail_add = None
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_video_survives_connect_error(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.down.add("radarr.test")
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 0
    assert rec_row(conn, rid)["status"] == "approved"
    net.down.clear()
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_video_4xx_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (400, '[{"errorMessage": "TMDb id required"}]')
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 1
    assert rec_row(conn, rid)["status"] == "failed"


def test_radarr_already_exists_is_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (400, '[{"errorMessage": "This movie has already been added"}]')
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert stats["errors"] == []
    assert rec_row(conn, rid)["status"] == "added"


def test_exists_validator_error_code_is_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.sonarr.fail_add = (
        400, '[{"errorCode": "SeriesExistsValidator", "errorMessage": "whatever"}]')
    rid = add_rec(conn, "series:tvdb:81189", "series", "Breaking Bad", {"tvdb": 81189},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_path_conflict_400_is_not_a_duplicate(conn, tmp_path, net):
    # "already" alone must not read as success: this 400 is a real failure
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (
        400, '[{"propertyName": "Path", "errorCode": "MoviePathValidator",'
        ' "errorMessage": "Path is already configured for an existing movie"}]')
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert stats["video_failed"] == 1
    assert any("Path is already configured" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"


# ── video auto mode ──────────────────────────────────────────────────


def test_video_auto_mode_adds_proposed_within_cap(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=2)
    top = add_rec(conn, "movie:tmdb:1", "movie", "M1", {"tmdb": 1}, score=0.9)
    mid = add_rec(conn, "series:tvdb:2", "series", "S2", {"tvdb": 2}, score=0.8)
    low = add_rec(conn, "movie:tmdb:3", "movie", "M3", {"tmdb": 3}, score=0.2)

    stats = apply_mod.run(conn, cfg)

    assert stats["video_added"] == 2
    assert [b["tmdbId"] for b in net.radarr.posted] == [1]
    assert [b["tvdbId"] for b in net.sonarr.posted] == [2]
    assert rec_row(conn, top)["status"] == "auto_added"
    assert rec_row(conn, mid)["status"] == "auto_added"
    assert rec_row(conn, low)["status"] == "proposed"  # over the per-run cap
    # audit event only, invisible to training
    events = conn.execute(
        "SELECT kind, weight, source FROM events WHERE item_id='movie:tmdb:1'").fetchall()
    assert [(e["kind"], e["weight"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]


def test_video_auto_mode_prefers_approved_and_skips_idless(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=1)
    approved = add_rec(conn, "movie:tmdb:10", "movie", "Approved", {"tmdb": 10},
                       status="approved", score=0.1)
    # tvdb-less series must not burn the single auto slot
    idless = add_rec(conn, "series:tmdb:11", "series", "No tvdb", {"tmdb": 11}, score=0.9)
    auto = add_rec(conn, "movie:tmdb:12", "movie", "Auto", {"tmdb": 12}, score=0.5)

    stats = apply_mod.run(conn, cfg)

    assert stats["video_added"] == 2
    assert rec_row(conn, approved)["status"] == "added"
    assert rec_row(conn, auto)["status"] == "auto_added"
    assert rec_row(conn, idless)["status"] == "proposed"


def test_video_queue_mode_ignores_proposed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)  # video_mode defaults to queue
    rid = add_rec(conn, "movie:tmdb:1", "movie", "M1", {"tmdb": 1}, score=0.9)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert net.radarr.posted == []
    assert rec_row(conn, rid)["status"] == "proposed"


# ── crash safety ─────────────────────────────────────────────────────


def test_successful_adds_committed_immediately(conn, tmp_path, net):
    # an arr add is irreversible: its record must survive the caller
    # never reaching conn.commit() (crash later in the run)
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    apply_mod.run(conn, cfg)
    other = db.connect(tmp_path / "t.db")
    try:
        row = other.execute(
            "SELECT status FROM recommendations WHERE id=?", (rid,)).fetchone()
        assert row["status"] == "added"
    finally:
        other.close()


def test_video_queue_overflow_expires_lowest_scores(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_queue_max_pending=2)
    keep1 = add_rec(conn, "movie:tmdb:1", "movie", "M1", {"tmdb": 1}, score=0.9)
    keep2 = add_rec(conn, "movie:tmdb:2", "movie", "M2", {"tmdb": 2}, score=0.8)
    drop1 = add_rec(conn, "movie:tmdb:3", "movie", "M3", {"tmdb": 3}, score=0.2)
    drop2 = add_rec(conn, "series:tvdb:4", "series", "S4", {"tvdb": 4}, score=0.1)
    stats = apply_mod.run(conn, cfg)
    assert stats["overflow_expired"] == 2
    assert rec_row(conn, keep1)["status"] == "proposed"
    assert rec_row(conn, keep2)["status"] == "proposed"
    assert rec_row(conn, drop1)["status"] == "expired"
    assert rec_row(conn, drop2)["status"] == "expired"


def test_stale_proposals_expire(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)  # proposal_ttl_days default 30
    rid = add_rec(conn, "movie:tmdb:603", "movie", "Old Prop", {"tmdb": 603},
                  ts="2020-01-01T00:00:00Z")
    stats = apply_mod.run(conn, cfg)
    assert stats["expired"] == 1
    assert rec_row(conn, rid)["status"] == "expired"


# ── dry run ──────────────────────────────────────────────────────────


def test_dry_run_mutates_nothing(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    r1 = add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603},
                 status="approved")

    stats = apply_mod.run(conn, cfg, dry_run=True)

    assert net.log == []  # no HTTP at all
    assert sorted(stats["would_add"]) == ["Artist 1", "The Matrix"]
    assert rec_row(conn, r1)["status"] == "proposed"
    assert rec_row(conn, r2)["status"] == "approved"
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert stats["jellyfin"] == {"would_sync": 0}


# ── arr client details ───────────────────────────────────────────────


def test_tag_ensured_once_across_adds(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie:tmdb:1", "movie", "M1", {"tmdb": 1}, status="approved", score=0.9)
    add_rec(conn, "movie:tmdb:2", "movie", "M2", {"tmdb": 2}, status="approved", score=0.8)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 2
    tag_calls = [m for m, h, p in net.log if h == "radarr.test" and p == "/api/v3/tag"]
    assert tag_calls == ["GET", "POST"]  # miss once, create once, then cached
    assert [b["tags"] for b in net.radarr.posted] == [[1], [1]]


def test_quality_profile_error_lists_available_names(net):
    client = arr_client.RadarrClient(
        ArrConfig(url="http://radarr.test", api_key="rk", quality_profile="Ultra-4K"))
    with pytest.raises(arr_client.ArrError) as exc:
        client.quality_profile_id()
    msg = str(exc.value)
    assert "Ultra-4K" in msg
    assert "HD-1080p" in msg
    assert "SD" in msg


def test_root_folder_prefers_configured_else_first(net):
    net.radarr.roots = [{"path": "/other"}, {"path": "/movies"}]
    client = arr_client.RadarrClient(
        ArrConfig(url="http://radarr.test", api_key="rk", root_folder="/movies/"))
    assert client.root_folder_path() == "/movies"
    # empty setting: first folder is the sensible default
    client = arr_client.RadarrClient(
        ArrConfig(url="http://radarr.test", api_key="rk", root_folder=""))
    assert client.root_folder_path() == "/other"


def test_root_folder_explicit_mismatch_raises(net):
    # an explicit setting must never silently fall back to another disk
    net.radarr.roots = [{"path": "/other"}]
    client = arr_client.RadarrClient(
        ArrConfig(url="http://radarr.test", api_key="rk", root_folder="/missing"))
    with pytest.raises(arr_client.ArrError) as exc:
        client.root_folder_path()
    msg = str(exc.value)
    assert "/missing" in msg
    assert "/other" in msg


# ── jellyfin collections ─────────────────────────────────────────────


def test_jellyfin_collections_created_then_idempotent(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603}, status="added",
            acted_at=db.now())
    add_rec(conn, "artist:mbid:mb1", "artist", "Artist 1", {"mbid": "mb1"},
            status="auto_added", acted_at=db.now())
    net.jellyfin.items["tmdb.603"] = "jf-603"
    net.jellyfin.items["musicbrainzartist.mb1"] = "jf-mb1"

    stats = jellyfin_collections.sync_collections(conn, cfg)
    assert stats == {"checked": 2, "matched": 2, "collections_created": 2,
                     "collection_adds": 2}
    by_name = {c["name"]: c["members"] for c in net.jellyfin.collections.values()}
    assert by_name["Gustarr Discover: Movies"] == ["jf-603"]
    assert by_name["Gustarr Discover: Music"] == ["jf-mb1"]

    again = jellyfin_collections.sync_collections(conn, cfg)
    assert again["collections_created"] == 0
    assert again["collection_adds"] == 0
    by_name = {c["name"]: c["members"] for c in net.jellyfin.collections.values()}
    assert by_name["Gustarr Discover: Movies"] == ["jf-603"]


def test_jellyfin_collection_extended_not_recreated(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.jellyfin.collections["coll-pre"] = {
        "name": "Gustarr Discover: Movies", "members": ["jf-1"]}
    add_rec(conn, "movie:tmdb:2", "movie", "M2", {"tmdb": 2}, status="added")
    net.jellyfin.items["tmdb.2"] = "jf-2"
    stats = jellyfin_collections.sync_collections(conn, cfg)
    assert stats["collections_created"] == 0
    assert stats["collection_adds"] == 1
    assert net.jellyfin.collections["coll-pre"]["members"] == ["jf-1", "jf-2"]


def test_jellyfin_skipped_without_config(conn, tmp_path, net):
    cfg = C._build({"core": {"data_dir": str(tmp_path)}})
    assert jellyfin_collections.sync_collections(conn, cfg) == {"skipped": True}


def test_jellyfin_failure_is_best_effort(conn, tmp_path, net, monkeypatch):
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie:tmdb:603", "movie", "The Matrix", {"tmdb": 603}, status="approved")

    def boom(*a, **kw):
        raise RuntimeError("jellyfin down")

    monkeypatch.setattr(jellyfin_collections, "sync_collections", boom)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1  # arr work still landed
    assert "jellyfin down" in stats["jellyfin_error"]
