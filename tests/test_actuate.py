"""Offline tests for the actuation slice: arr clients, apply, Jellyfin
collections and weekly playlists. All HTTP goes through an
httpx.MockTransport injected by monkeypatching gustarr.http.request_json."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gustarr import config as C
from gustarr import db, http, queue, settings, signals
from gustarr.actuate import apply as apply_mod
from gustarr.actuate import arr_client, jellyfin_collections, jellyfin_playlist
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
    """Jellyfin 10.x as the collections sync sees it: a typed library
    searchable by title (Fields=ProviderIds exposes provider ids for the
    client-side verify) plus the /Collections surface. AnyProviderIdEquals
    is deliberately NOT a filter: the real server ignores the unknown
    parameter and returns everything of the requested type, so a
    regression back to it matches garbage here as in production."""

    def __init__(self, api_key="jk"):
        self.api_key = api_key
        self.library = []  # {"Id", "Name", "Type", "ProviderIds": {...}}
        self.collections = {}  # id -> {"name": ..., "members": [...]}
        self.next_id = 1
        self.item_searches = []  # SearchTerm values of non-BoxSet lookups

    def add_library_item(self, jf_id, name, item_type, providers=None):
        self.library.append({"Id": jf_id, "Name": name, "Type": item_type,
                             "ProviderIds": dict(providers or {})})

    def _list_items(self, q) -> httpx.Response:
        if "ParentId" in q:
            coll = self.collections.get(q["ParentId"])
            members = coll["members"] if coll else []
            return httpx.Response(200, json={"Items": [{"Id": m} for m in members]})
        # collections surface in /Items as BoxSet rows
        items = self.library + [
            {"Id": cid, "Name": c["name"], "Type": "BoxSet", "ProviderIds": {}}
            for cid, c in self.collections.items()]
        types = q["IncludeItemTypes"].split(",") if q.get("IncludeItemTypes") else None
        if types:
            items = [i for i in items if i["Type"] in types]
        if "SearchTerm" in q:
            if types != ["BoxSet"]:
                self.item_searches.append(q["SearchTerm"])
            term = q["SearchTerm"].casefold()
            items = [i for i in items if term in i["Name"].casefold()]
        if "Limit" in q:
            items = items[: int(q["Limit"])]
        with_pids = "ProviderIds" in (q.get("Fields") or "").split(",")
        return httpx.Response(200, json={"Items": [
            {"Id": i["Id"], "Name": i["Name"], "Type": i["Type"],
             **({"ProviderIds": dict(i["ProviderIds"])} if with_pids else {})}
            for i in items]})

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Emby-Token") != self.api_key:
            return httpx.Response(401, text="bad token")
        path, q = request.url.path, request.url.params
        if path == "/Items" and request.method == "GET":
            return self._list_items(q)
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


def add_item(conn, domain, idents, title=None, meta=None):
    """Item with every given identity: the first (ns, key) resolves it,
    the rest are attached — the same two-step every collector performs."""
    (ns, key), *rest = idents.items()
    item_id = db.resolve_item(conn, domain, ns, str(key), title=title, meta=meta)
    for extra_ns, extra_key in rest:
        item_id = db.attach_identity(conn, item_id, extra_ns, str(extra_key))
    return item_id


def add_rec(conn, domain, title, idents, status="proposed",
            score=1.0, acted_at=None, ts=None, profile="default", meta=None):
    item_id = add_item(conn, domain, idents, title=title, meta=meta)
    cur = conn.execute(
        "INSERT INTO recommendations"
        " (profile, run_id, ts, domain, item_id, score, why, status, acted_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (profile, "test", ts or db.now(), domain, item_id, score, "{}", status, acted_at))
    return cur.lastrowid


def rec_row(conn, rec_id):
    return conn.execute("SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()


def item_of(conn, rec_id):
    return rec_row(conn, rec_id)["item_id"]


def events_of(conn, item_id):
    return conn.execute(
        "SELECT ts, kind, scale, source, profile FROM events WHERE item_id=?",
        (item_id,)).fetchall()


def seed_album(net, mbid, artist_mbid, title, artist_name="Some Artist"):
    net.lidarr.album_lookup[mbid] = {
        "title": title, "artist_mbid": artist_mbid, "artist_name": artist_name}


# ── runtime settings overrides ───────────────────────────────────────


def test_paused_short_circuits_music_and_video(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto")
    prop = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    appr = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                   status="approved")
    stale = add_rec(conn, "movie", "Old Prop", {"tmdb": 604},
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
    r1 = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "artist", "Artist 2", {"mbid": "mb2"}, score=0.8)
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
    rid = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"})
    settings.set(conn, "music_mode", "queue")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert net.lidarr.posted == []
    assert rec_row(conn, rid)["status"] == "proposed"


def test_video_cap_override_honored(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=5)
    top = add_rec(conn, "movie", "M1", {"tmdb": 1}, score=0.9)
    low = add_rec(conn, "movie", "M2", {"tmdb": 2}, score=0.8)
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
    add_rec(conn, "artist", "Acted One", {"mbid": "aa1"},
            status="auto_added", acted_at=now)
    add_rec(conn, "artist", "Acted Two", {"mbid": "aa2"},
            status="added", acted_at=now)
    add_rec(conn, "artist", "Old Acted", {"mbid": "aa3"},
            status="auto_added", acted_at=last_week)
    r1 = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "artist", "Artist 2", {"mbid": "mb2"}, score=0.8)
    r3 = add_rec(conn, "artist", "Artist 3", {"mbid": "mb3"}, score=0.7)

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
    events = events_of(conn, item_of(conn, r1))
    # audit trail only: gustarr's own add must not read as user praise —
    # the kind has no weight AND the scale is zero, belt and braces
    assert [(e["kind"], e["scale"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]
    assert "auto_add" not in signals.WEIGHTS
    assert signals.aggregate_label([(e["ts"], e["kind"], e["scale"]) for e in events]) == 0.0


def test_music_budget_shared_across_profiles(conn, tmp_path, net):
    """One disk, one Lidarr: what alice's queue already spent this week
    is gone for bob — the acted count ignores profile by design."""
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"], music_max_artists_per_week=2)
    add_rec(conn, "artist", "Alice Spent", {"mbid": "aa1"},
            status="auto_added", acted_at=db.now(), profile="alice")
    b1 = add_rec(conn, "artist", "Bob 1", {"mbid": "b1"}, score=0.9,
                 profile="bob")
    b2 = add_rec(conn, "artist", "Bob 2", {"mbid": "b2"}, score=0.8,
                 profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 1  # cap 2 minus alice's add, not bob's own 2
    assert stats["music_added"] == 1
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["b1"]
    assert rec_row(conn, b1)["status"] == "auto_added"
    assert rec_row(conn, b2)["status"] == "proposed"
    # the audit event names whose queue the add came from
    ev = conn.execute(
        "SELECT profile, kind FROM events WHERE item_id=?",
        (item_of(conn, b1),)).fetchone()
    assert (ev["profile"], ev["kind"]) == ("bob", "auto_add")


def test_music_budget_round_robin_across_profiles(conn, tmp_path, net):
    """A profile with a confident (higher-scoring) model must not outbid
    a cold-start one: the shared budget is spent one add per profile per
    turn, in config order — never in one score-ordered pass."""
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"], music_max_artists_per_week=2)
    a1 = add_rec(conn, "artist", "A1", {"mbid": "a1"}, score=0.9, profile="alice")
    a2 = add_rec(conn, "artist", "A2", {"mbid": "a2"}, score=0.8, profile="alice")
    b1 = add_rec(conn, "artist", "B1", {"mbid": "b1"}, score=0.2, profile="bob")
    b2 = add_rec(conn, "artist", "B2", {"mbid": "b2"}, score=0.1, profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 2
    assert stats["music_added"] == 2
    # one add EACH: alice's a2 outscores bob's entire queue and still
    # doesn't get the second slot
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["a1", "b1"]
    assert rec_row(conn, a1)["status"] == "auto_added"
    assert rec_row(conn, b1)["status"] == "auto_added"
    assert rec_row(conn, a2)["status"] == "proposed"
    assert rec_row(conn, b2)["status"] == "proposed"


def test_music_round_robin_skips_exhausted_queues(conn, tmp_path, net):
    # carol has nothing this week: her turn is skipped, the other two
    # keep alternating instead of one of them inheriting her slots
    cfg = make_cfg(tmp_path, profiles=["alice", "carol", "bob"],
                   music_max_artists_per_week=4)
    add_rec(conn, "artist", "A1", {"mbid": "a1"}, score=0.9, profile="alice")
    add_rec(conn, "artist", "A2", {"mbid": "a2"}, score=0.8, profile="alice")
    a3 = add_rec(conn, "artist", "A3", {"mbid": "a3"}, score=0.7, profile="alice")
    add_rec(conn, "artist", "B1", {"mbid": "b1"}, score=0.6, profile="bob")
    add_rec(conn, "artist", "B2", {"mbid": "b2"}, score=0.5, profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["music_added"] == 4
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["a1", "b1", "a2", "b2"]
    assert rec_row(conn, a3)["status"] == "proposed"  # budget gone before turn 3


def test_music_single_profile_spend_order_unchanged(conn, tmp_path, net):
    # one profile: round-robin degenerates to the old single ordered
    # pass — approved first (budget-exempt), then proposed by score
    cfg = make_cfg(tmp_path, music_max_artists_per_week=2)
    appr = add_rec(conn, "artist", "Approved Low", {"mbid": "ap1"}, score=0.1,
                   status="approved")
    add_rec(conn, "artist", "P1", {"mbid": "p1"}, score=0.9)
    add_rec(conn, "artist", "P2", {"mbid": "p2"}, score=0.8)
    p3 = add_rec(conn, "artist", "P3", {"mbid": "p3"}, score=0.7)

    stats = apply_mod.run(conn, cfg)

    assert stats["music_added"] == 3  # the approval plus two budget picks
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["ap1", "p1", "p2"]
    assert rec_row(conn, appr)["status"] == "added"
    assert rec_row(conn, p3)["status"] == "proposed"


def test_music_approved_rows_do_not_burn_round_robin_turns(conn, tmp_path, net):
    """Approved rows are consented and budget-exempt: they all actuate in
    phase 1 and must not spend their profile's round-robin turns —
    alice's two approvals must not hand bob both budget slots."""
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"], music_max_artists_per_week=2)
    ap1 = add_rec(conn, "artist", "AP1", {"mbid": "ap1"}, score=0.5,
                  status="approved", profile="alice")
    ap2 = add_rec(conn, "artist", "AP2", {"mbid": "ap2"}, score=0.4,
                  status="approved", profile="alice")
    a1 = add_rec(conn, "artist", "A1", {"mbid": "a1"}, score=0.9, profile="alice")
    a2 = add_rec(conn, "artist", "A2", {"mbid": "a2"}, score=0.8, profile="alice")
    b1 = add_rec(conn, "artist", "B1", {"mbid": "b1"}, score=0.2, profile="bob")
    b2 = add_rec(conn, "artist", "B2", {"mbid": "b2"}, score=0.1, profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["music_budget"] == 2
    assert stats["music_added"] == 4  # both approvals plus the two budget picks
    # approvals first (profile then score), then the budget alternates:
    # one slot each — never both to bob because alice's turns were burnt
    assert [b["foreignArtistId"] for b in net.lidarr.posted] == ["ap1", "ap2", "a1", "b1"]
    assert rec_row(conn, ap1)["status"] == "added"
    assert rec_row(conn, ap2)["status"] == "added"
    assert rec_row(conn, a1)["status"] == "auto_added"
    assert rec_row(conn, b1)["status"] == "auto_added"
    assert rec_row(conn, a2)["status"] == "proposed"
    assert rec_row(conn, b2)["status"] == "proposed"


def test_album_budget_round_robin_across_profiles(conn, tmp_path, net):
    # the album budget is a separate pot but spends by the same turns
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"], music_max_albums_per_week=2)
    seed_album(net, "al-a1", "ar-a1", "Alice Album 1", "Artist AA1")
    seed_album(net, "al-a2", "ar-a2", "Alice Album 2", "Artist AA2")
    seed_album(net, "al-b1", "ar-b1", "Bob Album 1", "Artist AB1")
    a1 = add_rec(conn, "album", "Alice Album 1", {"mbid": "al-a1"}, score=0.9,
                 profile="alice")
    a2 = add_rec(conn, "album", "Alice Album 2", {"mbid": "al-a2"}, score=0.8,
                 profile="alice")
    b1 = add_rec(conn, "album", "Bob Album 1", {"mbid": "al-b1"}, score=0.1,
                 profile="bob")

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_budget"] == 2
    assert stats["albums_added"] == 2
    monitored = {a["foreignAlbumId"] for a in net.lidarr.albums if a["monitored"]}
    assert monitored == {"al-a1", "al-b1"}  # one each, not alice's two
    assert rec_row(conn, a1)["status"] == "auto_added"
    assert rec_row(conn, b1)["status"] == "auto_added"
    assert rec_row(conn, a2)["status"] == "proposed"


def test_music_queue_mode_leaves_proposed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    r1 = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"})
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert net.lidarr.posted == []
    assert rec_row(conn, r1)["status"] == "proposed"


def test_approved_artist_added_in_queue_mode(conn, tmp_path, net):
    # explicit approval is consent: actuated even when music_mode='queue'
    cfg = make_cfg(tmp_path, music_mode="queue")
    rid = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), item_of(conn, rid), "approve", 1.0, "user", {"rec_id": rid})

    stats = apply_mod.run(conn, cfg)

    assert stats["music_added"] == 1
    assert net.lidarr.posted[0]["foreignArtistId"] == "mb1"
    row = rec_row(conn, rid)
    assert row["status"] == "added"
    assert row["acted_at"]
    # no extra taste event: the approve written at approval time is enough
    assert [e["kind"] for e in events_of(conn, item_of(conn, rid))] == ["approve"]


def test_approved_artist_exempt_from_weekly_budget(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=0)
    rid = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"},
                  status="approved")
    prop = add_rec(conn, "artist", "Artist 2", {"mbid": "mb2"}, score=0.9)
    stats = apply_mod.run(conn, cfg)
    assert stats["music_budget"] == 0
    assert stats["music_added"] == 1  # the approval, not the proposal
    assert rec_row(conn, rid)["status"] == "added"
    assert rec_row(conn, prop)["status"] == "proposed"


def test_approved_artist_without_mbid_marked_failed(conn, tmp_path, net):
    # name-keyed item = no authoritative identity: unaddressable in lidarr
    cfg = make_cfg(tmp_path, music_mode="queue")
    rid = add_rec(conn, "artist", "Some Band", {"name": "some band"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert net.lidarr.posted == []
    assert any("mbid" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"


def test_approved_artist_survives_lidarr_outage(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    net.lidarr.fail_add = (503, "service unavailable")
    rid = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"},
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
    rid = add_rec(conn, "artist", "Bogus", {"mbid": "bogus"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 0
    assert rec_row(conn, rid)["status"] == "failed"


def test_music_skips_artists_without_mbid(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_artists_per_week=1)
    add_rec(conn, "artist", "Some Band", {"name": "some band"}, score=0.99)
    r2 = add_rec(conn, "artist", "Artist 2", {"mbid": "mb2"}, score=0.5)
    stats = apply_mod.run(conn, cfg)
    assert stats["music_added"] == 1
    assert net.lidarr.posted[0]["foreignArtistId"] == "mb2"
    assert rec_row(conn, r2)["status"] == "auto_added"


def test_lidarr_failure_keeps_proposed_and_counts_attempt(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.lidarr.fail_add = (400, "Invalid foreignArtistId")
    r1 = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"})
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
    add_rec(conn, "album", "Done One", {"mbid": "done1"},
            status="auto_added", acted_at=now)
    add_rec(conn, "album", "Done Two", {"mbid": "done2"},
            status="added", acted_at=now)
    add_rec(conn, "album", "Old Done", {"mbid": "done3"},
            status="auto_added", acted_at=last_week)
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    seed_album(net, "al2", "ar2", "Album 2", "Artist Two")
    r1 = add_rec(conn, "album", "Album 1", {"mbid": "al1"}, score=0.9)
    r2 = add_rec(conn, "album", "Album 2", {"mbid": "al2"}, score=0.8)

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
    events = events_of(conn, item_of(conn, r1))
    assert [(e["kind"], e["scale"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]


def test_album_budget_override_honored(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_max_albums_per_week=5)
    seed_album(net, "al1", "ar1", "Album 1")
    seed_album(net, "al2", "ar2", "Album 2")
    r1 = add_rec(conn, "album", "Album 1", {"mbid": "al1"}, score=0.9)
    r2 = add_rec(conn, "album", "Album 2", {"mbid": "al2"}, score=0.8)
    settings.set(conn, "music_max_albums_per_week", 1)

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_budget"] == 1
    assert stats["albums_added"] == 1
    assert rec_row(conn, r1)["status"] == "auto_added"
    assert rec_row(conn, r2)["status"] == "proposed"


def test_album_artist_shell_created_unmonitored_when_absent(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    seed_album(net, "al1", "ar1", "OK Album", "New Artist")
    rid = add_rec(conn, "album", "OK Album", {"mbid": "al1"})

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


def test_album_prefers_release_group_mbid_from_meta(conn, tmp_path, net):
    # the identities mbid may be whatever MB id a collector first saw;
    # enrich stashes the release-group mbid (what lidarr looks up by) in
    # meta — when present it must win over the identity key
    cfg = make_cfg(tmp_path)
    seed_album(net, "rg1", "ar1", "RG Album", "Artist One")
    rid = add_rec(conn, "album", "RG Album", {"mbid": "release-xyz"},
                  meta={"release_group_mbid": "rg1"})

    stats = apply_mod.run(conn, cfg)

    assert stats["albums_added"] == 1
    album = next(a for a in net.lidarr.albums if a["foreignAlbumId"] == "rg1")
    assert album["monitored"] is True
    assert rec_row(conn, rid)["status"] == "auto_added"


def test_approved_album_added_in_queue_mode(conn, tmp_path, net):
    # explicit approval is consent: actuated even when music_mode='queue'
    cfg = make_cfg(tmp_path, music_mode="queue")
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    net.lidarr.register_artist("ar1", "Artist One")
    rid = add_rec(conn, "album", "Album 1", {"mbid": "al1"},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), item_of(conn, rid), "approve", 1.0, "user", {"rec_id": rid})
    prop = add_rec(conn, "album", "Album 2", {"mbid": "al2"}, score=0.9)

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
    assert [e["kind"] for e in events_of(conn, item_of(conn, rid))] == ["approve"]


def test_album_already_monitored_is_noop_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, music_mode="queue")
    seed_album(net, "al1", "ar1", "Album 1", "Artist One")
    net.lidarr.album_lookup["al1"]["monitored"] = True
    net.lidarr.register_artist("ar1", "Artist One")
    rid = add_rec(conn, "album", "Album 1", {"mbid": "al1"},
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
    rid = add_rec(conn, "album", "Album 1", {"mbid": "al1"},
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
    rid = add_rec(conn, "album", "Ghost Album", {"mbid": "nope"},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["albums_added"] == 0
    assert any("found nothing" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"  # deterministic 4xx-style failure


# ── video approvals ──────────────────────────────────────────────────


def test_approved_movie_added_without_event_duplication(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    # queue already wrote the approve event when the user approved
    db.add_event(conn, db.now(), item_of(conn, rid), "approve", 1.0, "user", {"rec_id": rid})

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
        "SELECT kind FROM events WHERE item_id=?", (item_of(conn, rid),))]
    assert kinds == ["approve"]


def test_approved_series_added_via_sonarr(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "series", "Breaking Bad", {"tvdb": 81189},
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
    rid = add_rec(conn, "series", "Some Show", {"tmdb": 1396},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 1
    assert any("tvdb" in e for e in stats["errors"])
    assert net.sonarr.posted == []
    assert rec_row(conn, rid)["status"] == "failed"


def test_approved_video_survives_arr_outage(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (503, "service unavailable")
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
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
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 0
    assert rec_row(conn, rid)["status"] == "approved"
    net.down.clear()
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


@pytest.mark.parametrize("status", [401, 403])
def test_approved_video_survives_credential_failure(conn, tmp_path, net, status):
    # credentials are service-level, not a verdict on the item: an
    # api-key rotation window must not burn approvals (enrich's taxonomy)
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (status, "invalid api key")
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert stats["video_failed"] == 0
    row = rec_row(conn, rid)
    assert row["status"] == "approved"  # retryable, not terminally failed
    assert json.loads(row["why"])["attempts"] == 1
    # key fixed: the same approval lands on the next apply
    net.radarr.fail_add = None
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_quality_profile_typo_leaves_approved(conn, tmp_path, net):
    # a typo'd quality_profile is operator config, not a verdict on the
    # item: failing the rec terminally would be unforgivable — it must
    # survive until the config is fixed
    cfg = C._build({
        "core": {"data_dir": str(tmp_path)},
        "radarr": {"url": "http://radarr.test", "api_key": "rk",
                   "quality_profile": "Ultra-4K", "root_folder": "/movies"},
    })
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert stats["video_failed"] == 0
    assert any("config" in e and "Ultra-4K" in e for e in stats["errors"])
    row = rec_row(conn, rid)
    assert row["status"] == "approved"  # retryable, not terminally failed
    assert json.loads(row["why"])["attempts"] == 1
    # operator fixes the config: the same approval lands on the next run
    stats = apply_mod.run(conn, make_cfg(tmp_path))
    assert stats["video_added"] == 1
    assert rec_row(conn, rid)["status"] == "added"


def test_approved_video_4xx_marked_failed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (400, '[{"errorMessage": "TMDb id required"}]')
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_failed"] == 1
    assert rec_row(conn, rid)["status"] == "failed"


def test_radarr_already_exists_is_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.radarr.fail_add = (400, '[{"errorMessage": "This movie has already been added"}]')
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1
    assert stats["errors"] == []
    assert rec_row(conn, rid)["status"] == "added"


def test_exists_validator_error_code_is_success(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    net.sonarr.fail_add = (
        400, '[{"errorCode": "SeriesExistsValidator", "errorMessage": "whatever"}]')
    rid = add_rec(conn, "series", "Breaking Bad", {"tvdb": 81189},
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
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert stats["video_failed"] == 1
    assert any("Path is already configured" in e for e in stats["errors"])
    assert rec_row(conn, rid)["status"] == "failed"


# ── video auto mode ──────────────────────────────────────────────────


def test_video_auto_mode_adds_proposed_within_cap(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=2)
    top = add_rec(conn, "movie", "M1", {"tmdb": 1}, score=0.9)
    mid = add_rec(conn, "series", "S2", {"tvdb": 2}, score=0.8)
    low = add_rec(conn, "movie", "M3", {"tmdb": 3}, score=0.2)

    stats = apply_mod.run(conn, cfg)

    assert stats["video_added"] == 2
    assert [b["tmdbId"] for b in net.radarr.posted] == [1]
    assert [b["tvdbId"] for b in net.sonarr.posted] == [2]
    assert rec_row(conn, top)["status"] == "auto_added"
    assert rec_row(conn, mid)["status"] == "auto_added"
    assert rec_row(conn, low)["status"] == "proposed"  # over the per-run cap
    # audit event only, invisible to training
    events = events_of(conn, item_of(conn, top))
    assert [(e["kind"], e["scale"], e["source"]) for e in events] == \
        [("auto_add", 0.0, "gustarr")]


def test_video_auto_mode_prefers_approved_and_skips_idless(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_mode="auto", video_queue_max_pending=1)
    approved = add_rec(conn, "movie", "Approved", {"tmdb": 10},
                       status="approved", score=0.1)
    # tvdb-less series must not burn the single auto slot
    idless = add_rec(conn, "series", "No tvdb", {"tmdb": 11}, score=0.9)
    auto = add_rec(conn, "movie", "Auto", {"tmdb": 12}, score=0.5)

    stats = apply_mod.run(conn, cfg)

    assert stats["video_added"] == 2
    assert rec_row(conn, approved)["status"] == "added"
    assert rec_row(conn, auto)["status"] == "auto_added"
    assert rec_row(conn, idless)["status"] == "proposed"


def test_video_queue_mode_ignores_proposed(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)  # video_mode defaults to queue
    rid = add_rec(conn, "movie", "M1", {"tmdb": 1}, score=0.9)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 0
    assert net.radarr.posted == []
    assert rec_row(conn, rid)["status"] == "proposed"


# ── crash safety ─────────────────────────────────────────────────────


def test_successful_adds_committed_immediately(conn, tmp_path, net):
    # an arr add is irreversible: its record must survive the caller
    # never reaching conn.commit() (crash later in the run)
    cfg = make_cfg(tmp_path)
    rid = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
                  status="approved")
    apply_mod.run(conn, cfg)
    other = db.connect(tmp_path / "t.db")
    try:
        row = other.execute(
            "SELECT status FROM recommendations WHERE id=?", (rid,)).fetchone()
        assert row["status"] == "added"
    finally:
        other.close()


def test_record_add_survives_racing_reject_and_keeps_its_event(conn, tmp_path, net):
    """The arr add is irreversible reality: a reject racing in mid-HTTP
    must not leave the store denying it — _record_add forces the row to
    'added'. The reject's taste event SURVIVES: taste and state are
    separate ledgers, and the verdict still trains the model."""
    rid = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"}, status="approved")
    row = rec_row(conn, rid)  # apply's snapshot, taken before the race
    queue.set_status(conn, rid, "rejected")  # the verdict lands mid-HTTP

    apply_mod._record_add(conn, row, db.now())

    assert rec_row(conn, rid)["status"] == "added"
    kinds = [e["kind"] for e in events_of(conn, item_of(conn, rid))]
    assert kinds == ["reject"]  # the racing verdict's ledger entry stands


def test_video_queue_overflow_expires_lowest_scores(conn, tmp_path, net):
    cfg = make_cfg(tmp_path, video_queue_max_pending=2)
    keep1 = add_rec(conn, "movie", "M1", {"tmdb": 1}, score=0.9)
    keep2 = add_rec(conn, "movie", "M2", {"tmdb": 2}, score=0.8)
    drop1 = add_rec(conn, "movie", "M3", {"tmdb": 3}, score=0.2)
    drop2 = add_rec(conn, "series", "S4", {"tvdb": 4}, score=0.1)
    stats = apply_mod.run(conn, cfg)
    assert stats["overflow_expired"] == 2
    assert rec_row(conn, keep1)["status"] == "proposed"
    assert rec_row(conn, keep2)["status"] == "proposed"
    assert rec_row(conn, drop1)["status"] == "expired"
    assert rec_row(conn, drop2)["status"] == "expired"


def test_stale_proposals_expire(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)  # proposal_ttl_days default 30
    rid = add_rec(conn, "movie", "Old Prop", {"tmdb": 603},
                  ts="2020-01-01T00:00:00Z")
    stats = apply_mod.run(conn, cfg)
    assert stats["expired"] == 1
    assert rec_row(conn, rid)["status"] == "expired"


# ── dry run ──────────────────────────────────────────────────────────


def test_dry_run_mutates_nothing(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    r1 = add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"}, score=0.9)
    r2 = add_rec(conn, "movie", "The Matrix", {"tmdb": 603},
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
    add_rec(conn, "movie", "M1", {"tmdb": 1}, status="approved", score=0.9)
    add_rec(conn, "movie", "M2", {"tmdb": 2}, status="approved", score=0.8)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 2
    tag_calls = [m for m, h, p in net.log if h == "radarr.test" and p == "/api/v3/tag"]
    assert tag_calls == ["GET", "POST"]  # miss once, create once, then cached
    assert [b["tags"] for b in net.radarr.posted] == [[1], [1]]


def test_quality_profile_error_lists_available_names(net):
    client = arr_client.RadarrClient(
        ArrConfig(url="http://radarr.test", api_key="rk", quality_profile="Ultra-4K"))
    # ArrConfigError, not plain ArrError: apply keys retryability on it
    with pytest.raises(arr_client.ArrConfigError) as exc:
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
    with pytest.raises(arr_client.ArrConfigError) as exc:
        client.root_folder_path()
    msg = str(exc.value)
    assert "/missing" in msg
    assert "/other" in msg


# ── jellyfin collections ─────────────────────────────────────────────


def test_jellyfin_collections_created_then_idempotent(conn, tmp_path, net):
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie", "The Matrix", {"tmdb": 603}, status="added",
            acted_at=db.now())
    add_rec(conn, "artist", "Artist 1", {"mbid": "mb1"},
            status="auto_added", acted_at=db.now())
    net.jellyfin.add_library_item("jf-603", "The Matrix", "Movie", {"Tmdb": "603"})
    net.jellyfin.add_library_item("jf-mb1", "Artist 1", "MusicArtist",
                                  {"MusicBrainzArtist": "mb1"})

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
    add_rec(conn, "movie", "M2", {"tmdb": 2}, status="added")
    net.jellyfin.add_library_item("jf-2", "M2", "Movie", {"Tmdb": "2"})
    stats = jellyfin_collections.sync_collections(conn, cfg)
    assert stats["collections_created"] == 0
    assert stats["collection_adds"] == 1
    assert net.jellyfin.collections["coll-pre"]["members"] == ["jf-1", "jf-2"]


def test_jellyfin_identity_skips_provider_search(conn, tmp_path, net):
    # a 'jellyfin' identity IS the Jellyfin item id: no search round-trip,
    # and it even routes items with no provider mapping at all (albums)
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie", "The Matrix", {"tmdb": 603, "jellyfin": "jf-direct"},
            status="added", acted_at=db.now())
    add_rec(conn, "album", "In Rainbows", {"mbid": "al1", "jellyfin": "jf-album"},
            status="auto_added", acted_at=db.now())

    stats = jellyfin_collections.sync_collections(conn, cfg)

    assert stats == {"checked": 2, "matched": 2, "collections_created": 2,
                     "collection_adds": 2}
    assert net.jellyfin.item_searches == []  # no /Items title lookups at all
    by_name = {c["name"]: c["members"] for c in net.jellyfin.collections.values()}
    assert by_name["Gustarr Discover: Movies"] == ["jf-direct"]
    assert by_name["Gustarr Discover: Music"] == ["jf-album"]


def test_jellyfin_skipped_without_config(conn, tmp_path, net):
    cfg = C._build({"core": {"data_dir": str(tmp_path)}})
    assert jellyfin_collections.sync_collections(conn, cfg) == {"skipped": True}


def test_jellyfin_failure_is_best_effort(conn, tmp_path, net, monkeypatch):
    cfg = make_cfg(tmp_path)
    add_rec(conn, "movie", "The Matrix", {"tmdb": 603}, status="approved")

    def boom(*a, **kw):
        raise RuntimeError("jellyfin down")

    monkeypatch.setattr(jellyfin_collections, "sync_collections", boom)
    stats = apply_mod.run(conn, cfg)
    assert stats["video_added"] == 1  # arr work still landed
    assert "jellyfin down" in stats["jellyfin_error"]


# ── jellyfin weekly playlist ─────────────────────────────────────────


class FakeLB:
    """ListenBrainz created-for playlists: exactly the two endpoints
    fetch_weekly reads — the createdfor listing and the SINGULAR
    /1/playlist/{mbid} fetch. The plural path is a 404 on the real API,
    so it is deliberately unserved here and a regression back to it
    fails loudly. set_week appends, so calling it again models a new
    week arriving while the old one still sits in the listing."""

    def __init__(self):
        self.weekly = {}  # user -> [{"mbid": ..., "date": ...}]
        self.tracks = {}  # playlist mbid -> JSPF track list

    def set_week(self, user, mbid, tracks, date="2026-07-06"):
        self.weekly.setdefault(user, []).append({"mbid": mbid, "date": date})
        self.tracks[mbid] = tracks

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/1/user/") and path.endswith("/playlists/createdfor"):
            user = path.split("/")[3]
            wrappers = [{"playlist": {
                "title": f"Weekly Exploration for {user}, week of {w['date']}",
                "identifier": f"https://listenbrainz.org/playlist/{w['mbid']}",
                "date": w["date"]}} for w in self.weekly.get(user, [])]
            return httpx.Response(200, json={"playlists": wrappers})
        if path.startswith("/1/playlist/"):  # singular, like the real API
            tracks = self.tracks.get(path.split("/")[3])
            if tracks is None:
                return httpx.Response(404, text="no such playlist")
            return httpx.Response(200, json={"playlist": {"track": tracks}})
        return httpx.Response(404, text=f"unhandled listenbrainz {request.method} {path}")


class FakeJellyfinPlaylists:
    """Jellyfin 10.x as the playlist sync sees it: a user list, one typed
    /Items listing (Ids, ParentId, IncludeItemTypes, case-insensitive
    SearchTerm on Name, Fields=ProviderIds, Limit), POST /Playlists and
    DELETE /Items/{id}. Two real-server behaviours are pinned by what is
    NOT served: AnyProviderIdEquals is no filter (the real server ignores
    the unknown parameter and returns everything of the requested type,
    so a regression to it matches garbage), and /Playlists/{id}/Items
    errors under API-key auth, so here it 404s like any unhandled route.
    Separate from FakeJellyfin so the collection tests' fake stays lean."""

    def __init__(self, api_key="jk"):
        self.api_key = api_key
        self.users = [{"Id": "u-ldx", "Name": "ldx"}]
        self.audio = []  # library: {"Id", "Name", "Type", "Artists", "ProviderIds"}
        self.playlists = {}  # id -> {"Name": ..., "items": [item ids]}
        self.created = []  # POST /Playlists JSON bodies
        self.deleted = []  # playlist ids removed via DELETE /Items/{id}
        self.next_id = 1
        self.log = []  # (method, path) of every authenticated request

    def add_track(self, jf_id, name, artists=(), mbid=None):
        self.audio.append({
            "Id": jf_id, "Name": name, "Type": "Audio", "Artists": list(artists),
            "ProviderIds": {"MusicBrainzTrack": mbid} if mbid else {}})

    def seed_playlist(self, name, item_ids):
        """A playlist that exists server-side but not in gustarr's state —
        i.e. one the user made themselves."""
        pid = f"pl{self.next_id}"
        self.next_id += 1
        self.playlists[pid] = {"Name": name, "items": list(item_ids)}
        return pid

    def playlist_items(self, pid):
        return list(self.playlists[pid]["items"])

    def writes(self):
        return [(m, p) for m, p in self.log if m != "GET"]

    def _list_items(self, q) -> httpx.Response:
        if "ParentId" in q:
            pl = self.playlists.get(q["ParentId"])
            members = pl["items"] if pl else []
            return httpx.Response(200, json={"Items": [{"Id": m} for m in members]})
        # playlists are items too: the alive check finds them by Ids
        items = self.audio + [
            {"Id": pid, "Name": pl["Name"], "Type": "Playlist", "ProviderIds": {}}
            for pid, pl in self.playlists.items()]
        if "Ids" in q:
            wanted = set(q["Ids"].split(","))
            items = [i for i in items if i["Id"] in wanted]
        if q.get("IncludeItemTypes"):
            types = q["IncludeItemTypes"].split(",")
            items = [i for i in items if i["Type"] in types]
        if "SearchTerm" in q:
            term = q["SearchTerm"].casefold()
            items = [i for i in items if term in i["Name"].casefold()]
        if "Limit" in q:
            items = items[: int(q["Limit"])]
        with_pids = "ProviderIds" in (q.get("Fields") or "").split(",")
        out = []
        for i in items:
            dto = {"Id": i["Id"], "Name": i["Name"], "Type": i["Type"]}
            if i["Type"] == "Audio":  # audio DTOs always carry Artists
                dto["Artists"] = list(i.get("Artists") or [])
            if with_pids:
                dto["ProviderIds"] = dict(i["ProviderIds"])
            out.append(dto)
        return httpx.Response(200, json={"Items": out})

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Emby-Token") != self.api_key:
            return httpx.Response(401, text="bad token")
        path, q = request.url.path, request.url.params
        self.log.append((request.method, path))
        if path == "/Users" and request.method == "GET":
            return httpx.Response(200, json=self.users)
        if path == "/Items" and request.method == "GET":
            return self._list_items(q)
        if path == "/Playlists" and request.method == "POST":
            body = json.loads(request.content)
            self.created.append(body)
            pid = f"pl{self.next_id}"
            self.next_id += 1
            self.playlists[pid] = {"Name": body["Name"],
                                   "items": list(body.get("Ids") or [])}
            return httpx.Response(200, json={"Id": pid})
        if request.method == "DELETE" and path.startswith("/Items/"):
            pid = path.removeprefix("/Items/")
            if pid not in self.playlists:
                return httpx.Response(404, text="no such item")
            del self.playlists[pid]
            self.deleted.append(pid)
            return httpx.Response(204)
        return httpx.Response(404, text=f"unhandled jellyfin {request.method} {path}")


class FakePlaylistNet:
    def __init__(self):
        self.lb = FakeLB()
        self.jellyfin = FakeJellyfinPlaylists()
        self.log = []  # (method, host, path)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.log.append((request.method, request.url.host, request.url.path))
        if request.url.host == "api.listenbrainz.org":
            return self.lb.handle(request)
        if request.url.host == "jellyfin.test":
            return self.jellyfin.handle(request)
        return httpx.Response(404, text=f"no such host {request.url.host}")


@pytest.fixture
def plnet(monkeypatch):
    monkeypatch.setattr(http, "HOST_DELAYS", {})
    monkeypatch.setattr(http.time, "sleep", lambda _s: None)
    fake = FakePlaylistNet()
    transport = httpx.MockTransport(fake.handler)
    orig = http.request_json

    def patched(method, url, **kw):
        kw["transport"] = transport
        return orig(method, url, **kw)

    monkeypatch.setattr(http, "request_json", patched)
    return fake


def make_playlist_cfg(tmp_path, profiles=None, listenbrainz=None):
    raw = {
        "core": {"data_dir": str(tmp_path)},
        "jellyfin": {"url": "http://jellyfin.test", "api_key": "jk"},
        "profiles": profiles or {
            "ldx": {"jellyfin_user": "ldx", "listenbrainz_user": "ldx"}},
    }
    if listenbrainz is not None:
        raw["listenbrainz"] = listenbrainz
    return C._build(raw)


def jspf(rec_mbid, title=None, creator=None):
    track = {"identifier": f"https://musicbrainz.org/recording/{rec_mbid}"}
    if title:
        track["title"] = title
    if creator:
        track["creator"] = creator
    return track


def test_weekly_playlist_first_run_creates(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-b", "Beta", ["Artist B"], mbid="rec-b")
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [
        jspf("rec-b", "Beta", "Artist B"),
        jspf("rec-a", "Alpha", "Artist A"),
        jspf("rec-c", "Gamma", "Artist C"),  # nowhere in the library
    ])

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats == {"profiles": 1, "matched": 2, "missing": 1,
                     "created": 1, "rebuilt": 0, "unchanged": 0}
    # created for the profile's Jellyfin user, tracks in LB running order,
    # matched via MusicBrainzTrack provider ids on the title-search hits
    assert jf.created == [{"Name": "Weekly Exploration", "Ids": ["jf-b", "jf-a"],
                           "UserId": "u-ldx", "MediaType": "Audio"}]
    (pid,) = jf.playlists
    assert jf.playlist_items(pid) == ["jf-b", "jf-a"]
    # the ONLY state is the profile-scoped playlist id: reconciliation is
    # content-driven now, so no week/mbid key exists
    assert db.get_state(conn, "p:ldx:jellyfin:weekly_playlist_id") == pid
    assert db.get_state(conn, "p:ldx:jellyfin:weekly_playlist_mbid") is None
    # the playlist exists in Jellyfin, so its record must already be
    # committed — it has to survive a crash later in the run
    other = db.connect(tmp_path / "t.db")
    try:
        assert db.pget_state(other, "ldx", "jellyfin:weekly_playlist_id") == pid
    finally:
        other.close()


def test_weekly_playlist_unchanged_rerun_is_noop(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [jspf("rec-a", "Alpha", "Artist A")])
    assert jellyfin_playlist.sync_playlists(conn, cfg)["created"] == 1
    pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    jf.log.clear()

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats == {"profiles": 1, "matched": 1, "missing": 0,
                     "created": 0, "rebuilt": 0, "unchanged": 1}
    assert jf.writes() == []  # contents already match: no POST, no DELETE
    assert jf.playlist_items(pid) == ["jf-a"]
    assert db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id") == pid


def test_weekly_playlist_library_growth_rebuilds(conn, tmp_path, plnet):
    """Weekly Exploration is mostly music the user doesn't have yet, and
    gustarr itself feeds it to Lidarr: when a missing track lands mid-week
    the playlist must grow — by delete+recreate, since Jellyfin 10.x gives
    an API key no workable in-place playlist edit."""
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [jspf("rec-a", "Alpha", "Artist A"),
                                        jspf("rec-b", "Beta", "Artist B")])
    first = jellyfin_playlist.sync_playlists(conn, cfg)
    assert first["created"] == 1 and first["missing"] == 1
    old_pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    # mid-week, the missing track arrives in the library
    jf.add_track("jf-b", "Beta", ["Artist B"], mbid="rec-b")
    jf.log.clear()

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats == {"profiles": 1, "matched": 2, "missing": 0,
                     "created": 0, "rebuilt": 1, "unchanged": 0}
    # rebuilt = DELETE /Items/{old} then POST /Playlists, LB order restored
    assert jf.writes() == [("DELETE", f"/Items/{old_pid}"), ("POST", "/Playlists")]
    assert jf.deleted == [old_pid]
    new_pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    assert new_pid != old_pid
    assert list(jf.playlists) == [new_pid]  # the old playlist is gone
    assert jf.playlist_items(new_pid) == ["jf-a", "jf-b"]


def test_weekly_playlist_zero_matches_never_wipes(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [jspf("rec-a", "Alpha", "Artist A")],
                      date="2026-06-29")
    jellyfin_playlist.sync_playlists(conn, cfg)
    pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    # a new week arrives of which the library holds nothing yet
    plnet.lb.set_week("ldx", "week-2", [jspf("rec-x", "Xenon", "Artist X"),
                                        jspf("rec-y", "Yttrium", "Artist Y")],
                      date="2026-07-06")
    jf.log.clear()

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats == {"profiles": 1, "matched": 0, "missing": 2,
                     "created": 0, "rebuilt": 0, "unchanged": 0}
    assert jf.writes() == []  # never wipe: an empty rebuild helps nobody
    assert jf.deleted == []
    assert jf.playlist_items(pid) == ["jf-a"]  # last week stays playable
    assert db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id") == pid


def test_weekly_playlist_deleted_serverside_recreated_fresh(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [jspf("rec-a", "Alpha", "Artist A")])
    # the user hand-made a playlist with the exact same name...
    user_pid = jf.seed_playlist("Weekly Exploration", ["jf-user-pick"])
    # ...while gustarr's own playlist from a previous sync is gone server-side
    db.pset_state(conn, "ldx", "jellyfin:weekly_playlist_id", "vanished")

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats["created"] == 1
    assert stats["rebuilt"] == 0 and stats["unchanged"] == 0
    new_pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    assert new_pid and new_pid not in ("vanished", user_pid)
    assert jf.playlist_items(new_pid) == ["jf-a"]
    # the user's own playlist was never adopted, emptied or deleted
    assert jf.playlist_items(user_pid) == ["jf-user-pick"]
    assert jf.deleted == []


def test_weekly_playlist_name_artist_fallback(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    # no MusicBrainzTrack provider ids anywhere: only the artist-verified
    # exact-title fallback can match
    jf.add_track("jf-wrong", "Same Song", ["Impostor"])  # decoy first
    jf.add_track("jf-right", "SAME SONG", ["Right Artist", "Guest"])
    plnet.lb.set_week("ldx", "week-1", [
        jspf("rec-x", "Same Song", "Right Artist"),
        jspf("rec-y", "Same Song", "Someone Else"),  # title exists, artist absent
    ])

    stats = jellyfin_playlist.sync_playlists(conn, cfg)

    assert stats == {"profiles": 1, "matched": 1, "missing": 1,
                     "created": 1, "rebuilt": 0, "unchanged": 0}
    pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    assert jf.playlist_items(pid) == ["jf-right"]  # never the wrong-artist hit


def test_weekly_playlist_opt_out_and_scoping(conn, tmp_path, plnet):
    # explicit opt-out: nothing is even fetched
    off = make_playlist_cfg(tmp_path, listenbrainz={"weekly_playlist": False})
    assert jellyfin_playlist.sync_playlists(conn, off) == {"skipped": True}
    assert plnet.log == []

    # a profile with no listenbrainz_user contributes nothing, and LB with
    # no weekly playlist generated yet (empty createdfor) writes nothing
    cfg = make_playlist_cfg(tmp_path, profiles={
        "ldx": {"jellyfin_user": "ldx", "listenbrainz_user": "ldx"},
        "video-only": {"jellyfin_user": "someone"}})
    stats = jellyfin_playlist.sync_playlists(conn, cfg)
    assert stats == {"profiles": 0, "matched": 0, "missing": 0,
                     "created": 0, "rebuilt": 0, "unchanged": 0}
    assert plnet.jellyfin.log == []  # Jellyfin never touched
    assert plnet.jellyfin.playlists == {}
    assert db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id") is None
    # only ldx was consulted: the LB-less profile never reached the API
    lb_paths = [p for _m, h, p in plnet.log if h == "api.listenbrainz.org"]
    assert lb_paths == ["/1/user/ldx/playlists/createdfor"]


def test_weekly_playlist_dry_run_makes_no_jellyfin_calls(conn, tmp_path, plnet):
    cfg = make_playlist_cfg(tmp_path)
    jf = plnet.jellyfin
    jf.add_track("jf-a", "Alpha", ["Artist A"], mbid="rec-a")
    plnet.lb.set_week("ldx", "week-1", [jspf("rec-a", "Alpha", "Artist A"),
                                        jspf("rec-b", "Beta", "Artist B")])
    jellyfin_playlist.sync_playlists(conn, cfg)
    pid = db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id")
    # a real run would now rebuild: the second track just hit the library
    jf.add_track("jf-b", "Beta", ["Artist B"], mbid="rec-b")
    jf.log.clear()
    before = len(plnet.log)

    stats = jellyfin_playlist.sync_playlists(conn, cfg, dry_run=True)

    assert stats == {"profiles": 1, "matched": 0, "missing": 0, "created": 0,
                     "rebuilt": 0, "unchanged": 0, "would_sync": 1}
    # ZERO Jellyfin requests, by both request logs
    assert jf.log == []
    assert [p for _m, h, p in plnet.log[before:] if h == "jellyfin.test"] == []
    assert jf.playlist_items(pid) == ["jf-a"]  # existing playlist untouched
    assert db.pget_state(conn, "ldx", "jellyfin:weekly_playlist_id") == pid
