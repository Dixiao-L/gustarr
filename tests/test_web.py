"""Web approval UI: FastAPI layer over a fabricated store — plus the
standalone `gustarr schedule` process."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from gustarr import config as C
from gustarr import db, scheduler
from gustarr.web.app import create_app

# assert_mbid validates the UUID shape now, so the fixture mbids the
# identify tests post must be real MusicBrainz-shaped ids.
RADIOHEAD_MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
NEW_MBID = "53f7a13b-0f75-4b31-9c30-2ed40de6dc35"
OTHER_MBID = "b0b2c1ff-1111-4222-8333-444455556666"


@pytest.fixture
def web(tmp_path):
    # TestClient sends Host: testserver, which the Host guard would 403;
    # allowlist it through config rather than hardcoding it in prod code.
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "web": {"allowed_hosts": ["testserver"]}})
    conn = db.connect(cfg.db_path)
    # resolve_item is THE write path since schema v3: items carry surrogate
    # int ids; each external id becomes an identities row.
    matrix = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix", year=1999,
                             meta={"genres": ["Action", "Science Fiction"]})
    blade = db.resolve_item(conn, "movie", "tmdb", "78", title="Blade Runner", year=1982,
                            meta={"genres": ["Science Fiction"]})
    radiohead = db.resolve_item(conn, "artist", "mbid", RADIOHEAD_MBID, title="Radiohead",
                                meta={"genres": ["rock"]})
    ts = db.now()
    # why JSON references neighbours by surrogate int id; /why joins the
    # titles back in server-side.
    why = json.dumps({"neighbors": [{"item_id": blade, "sim": 0.82}],
                      "sources": ["tmdb_similar"]})
    movie_rec = conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why)"
        " VALUES ('r1',?,?,?,?,?)", (ts, "movie", matrix, 0.91, why)).lastrowid
    artist_rec = conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why)"
        " VALUES ('r1',?,?,?,?,?)",
        (ts, "artist", radiohead, 0.62, '{"sources": ["lastfm_similar"]}')).lastrowid
    conn.commit()
    conn.close()
    items = {"matrix": matrix, "blade": blade, "radiohead": radiohead}
    return TestClient(create_app(cfg)), cfg, movie_rec, artist_rec, items


def _fetch(cfg, sql, *params):
    conn = db.connect(cfg.db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _add_album_rec(cfg):
    # Not in the shared fixture: most tests pin exact fixture row sets,
    # so the album rec is opt-in for the tests that need one.
    conn = db.connect(cfg.db_path)
    try:
        album = db.resolve_item(conn, "album", "mbid", "rg-1", title="In Rainbows", year=2007,
                                meta={"artist": "Radiohead", "type": "Album",
                                      "image": "https://img.example/ir.jpg"})
        rec = conn.execute(
            "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why)"
            " VALUES ('r1',?,?,?,?,?)",
            (db.now(), "album", album, 0.7, "{}")).lastrowid
        conn.commit()
        return rec
    finally:
        conn.close()


def test_list_recs(web):
    client, _, movie_rec, artist_rec, items = web
    resp = client.get("/api/recs")
    assert resp.status_code == 200
    rows = resp.json()
    assert {r["id"] for r in rows} == {movie_rec, artist_rec}
    assert all(r["status"] == "proposed" for r in rows)
    by_id = {r["id"]: r for r in rows}
    assert by_id[movie_rec]["title"] == "The Matrix"
    assert by_id[movie_rec]["year"] == 1999
    assert by_id[movie_rec]["genres"] == ["Action", "Science Fiction"]
    # v3 row shape: surrogate int item_id plus the external-ids dict
    # actuation reads (tvdb for Sonarr, mbid for Lidarr, ...)
    assert by_id[movie_rec]["item_id"] == items["matrix"]
    assert isinstance(by_id[movie_rec]["item_id"], int)
    assert by_id[movie_rec]["ids"].get("tmdb") == "603"

    movies = client.get("/api/recs", params={"domain": "movie"}).json()
    assert [r["id"] for r in movies] == [movie_rec]
    # empty domain param means "all domains"
    assert len(client.get("/api/recs", params={"domain": ""}).json()) == 2


def test_music_domain_alias_and_album_fields(web):
    client, cfg, _, artist_rec, _ = web
    album_rec = _add_album_rec(cfg)

    rows = client.get("/api/recs", params={"domain": "music"}).json()
    assert {r["id"] for r in rows} == {artist_rec, album_rec}  # movie filtered out
    by_id = {r["id"]: r for r in rows}
    assert by_id[album_rec]["title"] == "In Rainbows"
    assert by_id[album_rec]["artist"] == "Radiohead"
    assert by_id[album_rec]["type"] == "Album"
    assert by_id[album_rec]["image"] == "https://img.example/ir.jpg"
    # release-group mbid rides along for Lidarr's foreignAlbumId
    assert by_id[album_rec]["ids"]["mbid"] == "rg-1"
    # artists without an enriched image still list cleanly
    assert by_id[artist_rec]["image"] is None

    # exact-domain filtering is unchanged by the alias
    albums = client.get("/api/recs", params={"domain": "album"}).json()
    assert [r["id"] for r in albums] == [album_rec]


def test_approve_flips_status_and_records_event(web):
    client, cfg, movie_rec, _, items = web
    resp = client.post(f"/api/recs/{movie_rec}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    (row,) = _fetch(cfg, "SELECT status, acted_at FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "approved"
    assert row["acted_at"]
    events = _fetch(cfg, "SELECT scale, source FROM events WHERE item_id=? AND kind='approve'",
                    items["matrix"])
    assert len(events) == 1
    assert events[0]["scale"] == 1.0
    assert events[0]["source"] == "user"


def test_reject_flips_status_and_records_event(web):
    client, cfg, _, artist_rec, items = web
    assert client.post(f"/api/recs/{artist_rec}/reject").status_code == 200
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", artist_rec)
    assert row["status"] == "rejected"
    events = _fetch(cfg, "SELECT scale FROM events WHERE item_id=? AND kind='reject'",
                    items["radiohead"])
    assert len(events) == 1
    # scale multiplies signals.WEIGHTS["reject"] at train time — the
    # verdict's sign lives in the table now, so the stored scale is +1
    assert events[0]["scale"] == 1.0


def test_double_act_conflicts(web):
    client, _, movie_rec, *_ = web
    assert client.post(f"/api/recs/{movie_rec}/approve").status_code == 200
    assert client.post(f"/api/recs/{movie_rec}/approve").status_code == 409
    assert client.post("/api/recs/99999/approve").status_code == 409


def test_why_text(web):
    client, _, movie_rec, *_ = web
    resp = client.get(f"/api/recs/{movie_rec}/why")
    assert resp.status_code == 200
    text = resp.json()["text"]
    assert isinstance(text, str)
    assert "The Matrix" in text
    assert "tmdb_similar" in text
    # the neighbour arrived as a bare int item id; the endpoint must have
    # joined the title back in — a raw surrogate id is meaningless to people
    assert "Blade Runner" in text


def test_stats(web):
    client, *_ = web
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["recs_by_status"]["proposed"] == 2
    assert stats["tables"]["items"] == 3


def test_index_served(web):
    client, *_ = web
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Gustarr" in resp.text
    assert "/api/recs" in resp.text
    for symbol in ("i-check", "i-x", "i-info", "i-compass", "i-sparkle",
                   "i-film", "i-tv", "i-music", "i-clock", "i-chart",
                   "i-play", "i-external", "i-gear", "i-user", "i-tag", "i-retry"):
        assert f'id="{symbol}"' in resp.text
    for label in ("All", "Movies", "Series", "Music", "History", "Identities"):
        assert label in resp.text
    assert "aria-selected" in resp.text
    # Trailer/preview affordances ship as static strings in the page script.
    assert "youtube.com/watch" in resp.text
    assert "itunes.apple.com/search" in resp.text


def test_index_head_and_controls(web):
    client, *_ = web
    text = client.get("/").text
    # inline SVG favicon, url-encoded (not base64)
    assert 'rel="icon"' in text
    assert "data:image/svg+xml," in text
    assert "base64" not in text
    assert text.count('name="theme-color"') == 2
    assert 'media="(prefers-color-scheme: light)"' in text
    assert 'media="(prefers-color-scheme: dark)"' in text
    assert 'name="description"' in text
    for label in ("Run Now", "Settings", "Not Now", "Forgive", "Resume",
                  "Automation Paused", "Snoozed for 30 days"):
        assert label in text
    assert "/api/settings" in text
    assert "/api/run" in text
    assert "<dialog" in text
    # the settings dialog says out loud that it is not per-profile
    assert "Operator-level" in text


def test_index_profile_chip_plumbing(web):
    text = web[0].get("/").text
    # the page asks who it is serving and propagates ?profile= to API calls
    assert "/api/profile" in text
    assert "'profile'" in text
    # the chip renders only for multi-profile instances
    assert ".length > 1" in text
    assert "chip('user', 'Profile'" in text


def test_index_album_and_music_markup(web):
    client, *_ = web
    text = client.get("/").text
    # music cards label themselves via a muted type chip (meta.type or domain)
    assert 'badge dim type' in text
    assert "r.type || r.domain" in text
    # album cards carry a "by <artist>" byline under the title
    assert "byline" in text
    assert "by ${esc(r.artist)}" in text
    # meta.image paints over the monogram; its URL is used verbatim (no TMDB prefix)
    assert "r.image ? r.image" in text
    # the Music tab spans the audio domains client-side, mirroring the
    # server's domain=music alias (one status=all fetch feeds every tab)
    assert "'artist', 'album', 'track'" in text
    assert "domain=music" in text
    # the album weekly cap is editable next to the artist one
    assert "music_max_albums_per_week" in text
    assert "Weekly Album Cap" in text


def test_index_identity_panel_and_retry_markup(web):
    client, *_ = web
    text = client.get("/").text
    # the Identities tab fetches the pending list and proxies MB search
    assert "'identify'" in text
    assert "/api/identify" in text
    assert "/api/identify?q=" in text
    # rows show spellings + listen counts; hits show name/disambiguation/id
    assert "spellings" in text
    assert "listens" in text
    assert "disambiguation" in text
    assert "Assert" in text
    assert "Search MusicBrainz" in text
    # failed history items carry the human retry back to approved
    assert "'failed'" in text
    assert "Retry" in text
    assert "/retry" in text


def test_settings_get_defaults(web):
    client, *_ = web
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"paused", "music_mode", "music_max_artists_per_week",
                         "music_max_albums_per_week", "video_queue_max_pending",
                         "exploration_frac"}
    for entry in body.values():
        assert set(entry) == {"value", "overridden", "default"}
        assert entry["overridden"] is False
        assert entry["value"] == entry["default"]
    assert body["paused"]["value"] is False
    assert body["music_mode"]["value"] == "auto"
    assert body["music_max_artists_per_week"]["value"] == 3
    assert body["music_max_albums_per_week"]["value"] == 10
    assert body["video_queue_max_pending"]["value"] == 20
    assert body["exploration_frac"]["value"] == pytest.approx(0.15)


def test_settings_put_overrides_and_delete_clears(web):
    client, *_ = web
    resp = client.put("/api/settings/paused", json={"value": True})
    assert resp.status_code == 200
    assert resp.json() == {"key": "paused", "value": True}
    body = client.get("/api/settings").json()
    assert body["paused"] == {"value": True, "overridden": True, "default": False}

    # values are coerced on the way in ("5" → 5)
    resp = client.put("/api/settings/music_max_artists_per_week", json={"value": "5"})
    assert resp.status_code == 200
    assert resp.json()["value"] == 5
    body = client.get("/api/settings").json()
    assert body["music_max_artists_per_week"]["value"] == 5
    assert body["music_max_artists_per_week"]["default"] == 3

    assert client.delete("/api/settings/paused").status_code == 200
    body = client.get("/api/settings").json()
    assert body["paused"] == {"value": False, "overridden": False, "default": False}
    # clearing an already-clear key is a no-op, not an error
    assert client.delete("/api/settings/paused").status_code == 200


def test_settings_put_invalid_returns_400(web):
    client, *_ = web
    assert client.put("/api/settings/music_mode", json={"value": "chaos"}).status_code == 400
    assert client.put("/api/settings/no_such_key", json={"value": 1}).status_code == 400
    assert client.put("/api/settings/exploration_frac", json={"value": 2.0}).status_code == 400
    assert client.put("/api/settings/video_queue_max_pending", json={"value": 0}).status_code == 400
    assert client.put("/api/settings/paused", json={"nope": True}).status_code == 400
    assert client.delete("/api/settings/no_such_key").status_code == 400
    body = client.get("/api/settings").json()
    assert all(not entry["overridden"] for entry in body.values())


def test_snooze_flips_status_without_event(web):
    client, cfg, movie_rec, _, items = web
    resp = client.post(f"/api/recs/{movie_rec}/snooze")
    assert resp.status_code == 200
    assert resp.json()["status"] == "snoozed"

    (row,) = _fetch(cfg, "SELECT status, acted_at FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "snoozed"
    assert row["acted_at"]
    assert _fetch(cfg, "SELECT 1 FROM events WHERE item_id=?", items["matrix"]) == []
    # snoozed is not re-snoozable; and only proposed recs can be snoozed
    assert client.post(f"/api/recs/{movie_rec}/snooze").status_code == 409


def test_snooze_only_from_proposed(web):
    client, _, _, artist_rec, _ = web
    assert client.post(f"/api/recs/{artist_rec}/approve").status_code == 200
    assert client.post(f"/api/recs/{artist_rec}/snooze").status_code == 409
    assert client.post("/api/recs/99999/snooze").status_code == 409


def test_forgive_expires_rec_and_deletes_reject_event(web):
    client, cfg, _, artist_rec, items = web
    assert client.post(f"/api/recs/{artist_rec}/reject").status_code == 200
    assert len(_fetch(cfg, "SELECT 1 FROM events WHERE item_id=? AND kind='reject'",
                      items["radiohead"])) == 1

    resp = client.post(f"/api/recs/{artist_rec}/forgive")
    assert resp.status_code == 200
    assert resp.json()["status"] == "expired"
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", artist_rec)
    assert row["status"] == "expired"
    assert _fetch(cfg, "SELECT 1 FROM events WHERE item_id=? AND kind='reject'",
                  items["radiohead"]) == []


def test_forgive_requires_rejected(web):
    client, _, movie_rec, *_ = web
    assert client.post(f"/api/recs/{movie_rec}/forgive").status_code == 409
    assert client.post(f"/api/recs/{movie_rec}/approve").status_code == 200
    assert client.post(f"/api/recs/{movie_rec}/forgive").status_code == 409
    assert client.post("/api/recs/99999/forgive").status_code == 409


# ── retry ────────────────────────────────────────────────────────────


def _add_failed_rec(cfg, profile="default"):
    conn = db.connect(cfg.db_path)
    try:
        album = db.resolve_item(conn, "album", "mbid", "rg-lost", title="Lost Album")
        rec = conn.execute(
            "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score,"
            " why, status) VALUES (?, 'r1', ?, 'album', ?, 0.5, '{}', 'failed')",
            (profile, db.now(), album)).lastrowid
        conn.commit()
        return rec
    finally:
        conn.close()


def test_retry_requeues_failed_without_event(web):
    client, cfg, *_ = web
    failed = _add_failed_rec(cfg)

    resp = client.post(f"/api/recs/{failed}/retry")
    assert resp.status_code == 200
    assert resp.json() == {"id": failed, "status": "approved", "updated": 1, "events": 0}

    (row,) = _fetch(cfg, "SELECT status, acted_at FROM recommendations WHERE id=?", failed)
    assert row["status"] == "approved"
    assert row["acted_at"]
    # a retry is plumbing, not a verdict: the model must learn nothing
    assert _fetch(cfg, "SELECT 1 FROM events") == []


def test_retry_requires_failed(web):
    client, cfg, movie_rec, *_ = web
    assert client.post(f"/api/recs/{movie_rec}/retry").status_code == 409
    assert client.post("/api/recs/99999/retry").status_code == 409
    # retrying an already-retried rec is a state answer too, not a mutation
    failed = _add_failed_rec(cfg)
    assert client.post(f"/api/recs/{failed}/retry").status_code == 200
    assert client.post(f"/api/recs/{failed}/retry").status_code == 409


def test_cross_profile_retry_denied_with_403(multi):
    client, cfg, *_ = multi
    failed = _add_failed_rec(cfg, profile="alice")

    resp = client.post(f"/api/recs/{failed}/retry", headers={"Remote-User": "bob"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "recommendation belongs to another profile"
    # the header-less 'default' identity is a stranger to alice's rec too
    assert client.post(f"/api/recs/{failed}/retry").status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", failed)
    assert row["status"] == "failed"
    # the guard never locks the owner out of her own retry
    assert client.post(f"/api/recs/{failed}/retry",
                       headers={"Remote-User": "alice"}).status_code == 200


# ── identities ───────────────────────────────────────────────────────


def _add_nameonly_artist(cfg, name="KinokoTeikoku", listens=2):
    conn = db.connect(cfg.db_path)
    try:
        artist = db.resolve_item(conn, "artist", "name", name, title=name)
        for i in range(listens):
            db.add_event(conn, f"2026-01-0{i + 1}T00:00:00Z", artist, "scrobble",
                         1.0, "lastfm")
        conn.commit()
        return artist
    finally:
        conn.close()


def test_identify_lists_pending_artists(web):
    client, cfg, *_ = web
    artist = _add_nameonly_artist(cfg)

    resp = client.get("/api/identify")
    assert resp.status_code == 200
    rows = resp.json()
    # radiohead holds an mbid and never shows; the name-only artist does
    assert [r["item_id"] for r in rows] == [artist]
    assert rows[0] == {"item_id": artist, "title": "KinokoTeikoku",
                       "spellings": ["kinokoteikoku"], "events": 2}


def test_identify_q_proxies_musicbrainz_search(web, monkeypatch):
    from gustarr import http

    client, *_ = web
    calls = []

    def fake(method, url, *, params=None, **kw):
        calls.append((url, dict(params or {})))
        return {"artists": [{"id": "mb-1", "name": "Kinoko Teikoku", "score": 100,
                             "type": "Group", "country": "JP"}]}

    monkeypatch.setattr(http, "request_json", fake)

    resp = client.get("/api/identify", params={"q": "Kinoko Teikoku"})
    assert resp.status_code == 200
    (hit,) = resp.json()
    assert (hit["id"], hit["name"]) == ("mb-1", "Kinoko Teikoku")
    assert hit["disambiguation"] == ""
    url, params = calls[0]
    assert "musicbrainz.org" in url
    assert params["query"] == 'artist:"Kinoko Teikoku"'


def test_identify_assert_attaches_and_reopens_enrichment(web):
    client, cfg, *_ = web
    artist = _add_nameonly_artist(cfg)

    resp = client.post("/api/identify", json={"item_id": artist, "mbid": NEW_MBID})
    assert resp.status_code == 200
    assert resp.json() == {"item_id": artist, "mbid": NEW_MBID}

    (row,) = _fetch(cfg, "SELECT item_id FROM identities WHERE ns='mbid' AND key=?", NEW_MBID)
    assert row["item_id"] == artist  # committed
    (row,) = _fetch(cfg, "SELECT enriched_at FROM items WHERE id=?", artist)
    assert row["enriched_at"] is None
    assert client.get("/api/identify").json() == []  # off the pending list


def test_identify_assert_merge_reports_survivor(web):
    client, cfg, _, _, items = web
    twin = _add_nameonly_artist(cfg, name="radiohead uk")

    # asserting radiohead's mbid merges the spelling twin into it; the
    # response carries the survivor the UI must key off from now on
    resp = client.post("/api/identify", json={"item_id": twin, "mbid": RADIOHEAD_MBID})
    assert resp.status_code == 200
    assert resp.json()["item_id"] == items["radiohead"]
    assert _fetch(cfg, "SELECT 1 FROM items WHERE id=?", twin) == []


def test_identify_assert_refuses_identified_artist(web):
    client, cfg, _, _, items = web
    # radiohead already holds its mbid: the cross-entity refusal rule
    # holds on the human path, as a clear 409, and nothing is written
    resp = client.post("/api/identify",
                       json={"item_id": items["radiohead"], "mbid": OTHER_MBID})
    assert resp.status_code == 409
    assert "already has MusicBrainz id" in resp.json()["detail"]
    assert _fetch(cfg, "SELECT 1 FROM identities WHERE ns='mbid' AND key=?", OTHER_MBID) == []


def test_identify_assert_validates_body(web):
    client, *_ = web
    for bad in ({}, {"item_id": 1}, {"mbid": "x"}, {"item_id": "1", "mbid": "x"},
                {"item_id": 1, "mbid": ""}):
        assert client.post("/api/identify", json=bad).status_code == 400
    assert client.post("/api/identify",
                       json={"item_id": 999, "mbid": NEW_MBID}).status_code == 409


def test_identify_assert_malformed_mbid_is_400(web):
    client, cfg, *_ = web
    artist = _add_nameonly_artist(cfg)
    # 'banana', a pasted artist name, and a truncated uuid: bad requests,
    # never store-state conflicts — and nothing is written
    for bad in ("banana", "Kinoko Teikoku", NEW_MBID[:-4]):
        resp = client.post("/api/identify", json={"item_id": artist, "mbid": bad})
        assert resp.status_code == 400
        assert "not a MusicBrainz id" in resp.json()["detail"]
    assert _fetch(cfg, "SELECT 1 FROM identities WHERE ns='mbid' AND item_id=?", artist) == []


def test_identify_assert_guarded_like_every_post(web):
    client, cfg, *_ = web
    artist = _add_nameonly_artist(cfg)
    resp = client.post("/api/identify", json={"item_id": artist, "mbid": NEW_MBID},
                       headers={"Origin": "https://evil.example.com"})
    assert resp.status_code == 403
    assert _fetch(cfg, "SELECT 1 FROM identities WHERE ns='mbid' AND key=?", NEW_MBID) == []


def test_run_now_touches_sentinel(web):
    client, cfg, *_ = web
    sentinel = cfg.data_dir / "run-requested"
    assert not sentinel.exists()
    resp = client.post("/api/run")
    assert resp.status_code == 200
    assert resp.json() == {"requested": True}
    assert sentinel.is_file()
    # re-requesting while a sentinel is pending stays fine (touch is idempotent)
    assert client.post("/api/run").status_code == 200


def test_foreign_host_rejected(web):
    client, cfg, movie_rec, *_ = web
    assert client.get("/api/recs", headers={"Host": "evil.example.com"}).status_code == 403
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Host": "attacker.rebind.net"})
    assert resp.status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "proposed"


def test_cross_origin_post_rejected(web):
    client, cfg, movie_rec, _, items = web
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Origin": "https://evil.example.com"})
    assert resp.status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "proposed"
    events = _fetch(cfg, "SELECT 1 FROM events WHERE item_id=? AND kind='approve'",
                    items["matrix"])
    assert events == []


def test_no_origin_and_allowlisted_origin_posts_allowed(web):
    client, _, movie_rec, artist_rec, _ = web
    # CLI clients and same-origin navigations send no Origin header.
    assert client.post(f"/api/recs/{artist_rec}/reject").status_code == 200
    # Same-origin fetch from the served UI carries an allowlisted Origin.
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Origin": "http://localhost:8790"})
    assert resp.status_code == 200


# ── profiles ─────────────────────────────────────────────────────────


def _profiles_cfg(tmp_path, **web_extra):
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "web": {"allowed_hosts": ["testserver"], **web_extra},
        "profiles": {"alice": {"jellyfin_user": "alice"},
                     "bob": {"lastfm_user": "bob-fm"}},
    })


@pytest.fixture
def multi(tmp_path):
    cfg = _profiles_cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    matrix = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix", year=1999)
    blade = db.resolve_item(conn, "movie", "tmdb", "78", title="Blade Runner", year=1982)
    # Same-item recs under two profiles are legal: the open-rec unique
    # index is (profile, item_id) — each person gets their own verdict.
    alice_rec = conn.execute(
        "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score, why)"
        " VALUES ('alice','r1',?,?,?,?,?)", (db.now(), "movie", matrix, 0.9, "{}")).lastrowid
    bob_rec = conn.execute(
        "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score, why)"
        " VALUES ('bob','r1',?,?,?,?,?)", (db.now(), "movie", blade, 0.8, "{}")).lastrowid
    conn.commit()
    conn.close()
    return TestClient(create_app(cfg)), cfg, alice_rec, bob_rec


def test_profile_from_forward_auth_header(multi):
    client, *_ = multi
    resp = client.get("/api/profile", headers={"Remote-User": "alice"})
    assert resp.status_code == 200
    assert resp.json() == {"name": "alice", "profiles": ["alice", "bob"]}


def test_profile_from_query_param(multi):
    client, *_ = multi
    assert client.get("/api/profile", params={"profile": "bob"}).json()["name"] == "bob"
    # the auth proxy's header outranks the query param
    resp = client.get("/api/profile", params={"profile": "bob"},
                      headers={"Remote-User": "alice"})
    assert resp.json()["name"] == "alice"


def test_profile_sole_configured_fallback(web):
    # Legacy configs synthesize a single 'default' profile; no header or
    # param is needed for anything to work.
    client, *_ = web
    assert client.get("/api/profile").json() == {"name": "default", "profiles": ["default"]}


def test_profile_multi_without_hint_is_default(multi):
    client, *_ = multi
    assert client.get("/api/profile").json()["name"] == "default"


def test_unknown_profile_403(multi):
    client, *_ = multi
    for req in ({"headers": {"Remote-User": "mallory"}},
                {"params": {"profile": "mallory"}}):
        resp = client.get("/api/profile", **req)
        assert resp.status_code == 403
        assert "mallory" in resp.json()["detail"]
    assert client.get("/api/recs", params={"profile": "mallory"}).status_code == 403
    assert client.get("/api/stats", headers={"Remote-User": "mallory"}).status_code == 403


def test_profile_header_name_configurable(tmp_path):
    cfg = _profiles_cfg(tmp_path, profile_header="X-Forwarded-User")
    client = TestClient(create_app(cfg))
    assert client.get("/api/profile",
                      headers={"X-Forwarded-User": "bob"}).json()["name"] == "bob"
    # the default header name carries no meaning once another is configured
    resp = client.get("/api/profile", headers={"Remote-User": "nobody"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "default"


def test_recs_scoped_to_active_profile(multi):
    client, _, alice_rec, bob_rec = multi
    rows = client.get("/api/recs", headers={"Remote-User": "alice"}).json()
    assert [r["id"] for r in rows] == [alice_rec]
    rows = client.get("/api/recs", params={"profile": "bob"}).json()
    assert [r["id"] for r in rows] == [bob_rec]


def test_cross_profile_act_denied_with_403(multi):
    client, cfg, alice_rec, bob_rec = multi
    # rec ids are guessable smallints, so acting on another profile's rec
    # must be authorization (403), not a state answer (409) or a mutation
    for action in ("approve", "reject", "snooze"):
        resp = client.post(f"/api/recs/{alice_rec}/{action}", headers={"Remote-User": "bob"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "recommendation belongs to another profile"
    # the header-less 'default' identity is a stranger to alice's rec too
    assert client.post(f"/api/recs/{alice_rec}/approve").status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", alice_rec)
    assert row["status"] == "proposed"
    assert _fetch(cfg, "SELECT 1 FROM events") == []
    # the guard never locks anyone out of their own queue
    assert client.post(f"/api/recs/{alice_rec}/approve",
                       headers={"Remote-User": "alice"}).status_code == 200
    assert client.post(f"/api/recs/{bob_rec}/reject",
                       params={"profile": "bob"}).status_code == 200


def test_cross_profile_forgive_denied_with_403(multi):
    client, cfg, alice_rec, bob_rec = multi
    assert client.post(f"/api/recs/{alice_rec}/reject",
                       headers={"Remote-User": "alice"}).status_code == 200
    resp = client.post(f"/api/recs/{alice_rec}/forgive", headers={"Remote-User": "bob"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "recommendation belongs to another profile"
    # ownership outranks the state machine: a foreign *proposed* rec is
    # still 403, never the state-leaking "only rejected recs" 409
    assert client.post(f"/api/recs/{bob_rec}/forgive",
                       headers={"Remote-User": "alice"}).status_code == 403
    # the denied forgive left the reject verdict (and its event) intact
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", alice_rec)
    assert row["status"] == "rejected"
    assert len(_fetch(cfg, "SELECT 1 FROM events WHERE kind='reject'")) == 1
    assert client.post(f"/api/recs/{alice_rec}/forgive",
                       headers={"Remote-User": "alice"}).status_code == 200


def test_cross_profile_why_403(multi):
    client, _, alice_rec, _ = multi
    # explanations are taste data: reading across profiles is denied like
    # acting is, and with the same non-leaking detail
    resp = client.get(f"/api/recs/{alice_rec}/why", headers={"Remote-User": "bob"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "recommendation belongs to another profile"
    ok = client.get(f"/api/recs/{alice_rec}/why", headers={"Remote-User": "alice"})
    assert ok.status_code == 200
    assert "The Matrix" in ok.json()["text"]
    # a genuinely unknown id stays a plain 404 for everyone
    assert client.get("/api/recs/99999/why", headers={"Remote-User": "bob"}).status_code == 404


# ── built-in scheduler ───────────────────────────────────────────────


class FakeProc:
    def __init__(self):
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode


def test_scheduler_fires_once_and_skips_while_running():
    procs = []

    def popen(cmd):
        assert cmd[-2:] == ["run", "nightly"]
        procs.append(FakeProc())
        return procs[-1]

    sched = scheduler.Scheduler("04:30", popen=popen)
    assert sched.tick(datetime(2026, 7, 12, 4, 29)) is False
    assert sched.tick(datetime(2026, 7, 12, 4, 30)) is True
    # same minute / later the same day: one fire per day, ever
    assert sched.tick(datetime(2026, 7, 12, 4, 30)) is False
    assert sched.tick(datetime(2026, 7, 12, 18, 0)) is False
    assert len(procs) == 1
    # next day's slot arrives while the run is still alive: skip, don't queue
    assert sched.tick(datetime(2026, 7, 13, 4, 30)) is False
    assert len(procs) == 1
    # run finished → the following day fires again
    procs[0].returncode = 0
    assert sched.tick(datetime(2026, 7, 14, 4, 30)) is True
    assert len(procs) == 2
    # a tick that drifted past the target minute still fires that day
    procs[1].returncode = 1
    assert sched.tick(datetime(2026, 7, 15, 4, 47)) is True
    assert len(procs) == 3


def test_scheduler_prime_spends_todays_past_slot():
    # Booting the web UI at noon must not instantly fire an 04:30 pipeline.
    sched = scheduler.Scheduler("04:30", popen=lambda cmd: FakeProc())
    sched.prime(datetime(2026, 7, 12, 12, 0))
    assert sched.tick(datetime(2026, 7, 12, 12, 1)) is False
    assert sched.tick(datetime(2026, 7, 13, 4, 30)) is True


def test_scheduler_consumes_run_now_sentinel(tmp_path, capsys):
    procs = []

    def popen(cmd):
        assert cmd[-2:] == ["run", "nightly"]
        procs.append(FakeProc())
        return procs[-1]

    sentinel = tmp_path / "run-requested"
    sched = scheduler.Scheduler("04:30", popen=popen, sentinel=sentinel)
    # spend today's clock slot so only the sentinel can fire below
    sched.prime(datetime(2026, 7, 12, 12, 0))
    assert sched.tick(datetime(2026, 7, 12, 12, 1)) is False  # nothing requested
    sentinel.touch()
    assert sched.tick(datetime(2026, 7, 12, 12, 2)) is True
    assert not sentinel.exists()
    assert len(procs) == 1
    # a second request while the run is alive: consumed, but skipped — a
    # leftover file would otherwise re-fire every minute forever
    sentinel.touch()
    assert sched.tick(datetime(2026, 7, 12, 12, 3)) is False
    assert not sentinel.exists()
    assert len(procs) == 1
    assert "still alive" in capsys.readouterr().out
    # once the run finishes, RUN NOW fires again the same day
    procs[0].returncode = 0
    sentinel.touch()
    assert sched.tick(datetime(2026, 7, 12, 12, 4)) is True
    assert len(procs) == 2


def test_scheduler_sentinel_run_makes_clock_slot_skip(tmp_path):
    procs = []

    def popen(cmd):
        procs.append(FakeProc())
        return procs[-1]

    sentinel = tmp_path / "run-requested"
    sched = scheduler.Scheduler("04:30", popen=popen, sentinel=sentinel)
    sched.prime(datetime(2026, 7, 12, 3, 0))
    sentinel.touch()
    assert sched.tick(datetime(2026, 7, 12, 3, 1)) is True
    # the nightly slot arrives while the manual run is alive: skip, not queue
    assert sched.tick(datetime(2026, 7, 12, 4, 30)) is False
    assert len(procs) == 1
    # the skipped slot stays spent for the day; the next day fires normally
    procs[0].returncode = 0
    assert sched.tick(datetime(2026, 7, 12, 4, 31)) is False
    assert sched.tick(datetime(2026, 7, 13, 4, 30)) is True
    assert len(procs) == 2


def test_schedule_command_requires_config(tmp_path):
    cfg = C._build({"core": {"data_dir": str(tmp_path)}})
    with pytest.raises(SystemExit):
        scheduler.main(cfg)


def test_schedule_main_wires_sentinel(tmp_path, monkeypatch):
    # `gustarr schedule` must watch the very file web's POST /api/run
    # touches, or RUN NOW silently no-ops in Docker deployments
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "scheduler": {"nightly": "04:30"}})
    seen = {}

    def fake_run_forever(self, *args, **kwargs):
        seen["sentinel"] = self.sentinel
        seen["at"] = (self.hour, self.minute)

    monkeypatch.setattr(scheduler.Scheduler, "run_forever", fake_run_forever)
    scheduler.main(cfg)
    assert seen["sentinel"] == tmp_path / "run-requested"
    assert seen["at"] == (4, 30)


def test_web_process_has_no_scheduler(tmp_path):
    # one process, one job: the web app must never own scheduling
    import gustarr.web.app as app_mod
    assert not hasattr(app_mod, "scheduler")


def test_configured_allowed_hosts_honored(tmp_path):
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "web": {"allowed_hosts": ["gustarr.pit21.net"]}})
    client = TestClient(create_app(cfg))
    ok = {"Host": "gustarr.pit21.net"}
    assert client.get("/api/recs", headers=ok).status_code == 200
    # Host matching is port-insensitive.
    assert client.get("/api/recs", headers={"Host": "gustarr.pit21.net:8790"}).status_code == 200
    # Default localhost aliases stay allowed alongside configured hosts.
    assert client.get("/api/recs", headers={"Host": "127.0.0.1:8790"}).status_code == 200
    assert client.get("/api/recs", headers={"Host": "localhost"}).status_code == 200
    # TestClient's default Host (testserver) is not allowlisted here.
    assert client.get("/api/recs").status_code == 403
    resp = client.post("/api/recs/1/approve",
                       headers={"Host": "gustarr.pit21.net",
                                "Origin": "https://gustarr.pit21.net"})
    assert resp.status_code == 409  # passed the guard; no such rec
