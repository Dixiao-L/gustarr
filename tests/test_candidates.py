"""Offline tests for candidates.run: seeding, fan-out, exclusions, caps.

Identity v3: fixtures mint items through db.resolve_item and tests find
them back through db.lookup_item — candidate/exclusion plumbing is all
plain integer item ids now.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gustarr import config as C
from gustarr import db, http
from gustarr.candidates import SNOOZE_DAYS, _excluded_ids, _tmdb_result, run

TMDB = "https://api.themoviedb.org/3"


def iso(days_ago: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "tmdb": {"api_key": "tk"},
        "lastfm": {"api_key": "lk", "user": "u"},
    })


class Router:
    """Fake gustarr.http.request_json: routes on URL fragment (or last.fm
    method) and records every call. Unmocked requests fail the test."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def route(self, fragment, payload):
        self.routes[fragment] = payload

    def __call__(self, method, url, *, params=None, **kw):
        params = dict(params or {})
        self.calls.append((url, params))
        key = url
        if "audioscrobbler" in url:
            key = f"lastfm:{params.get('method')}"
        for fragment, payload in self.routes.items():
            if fragment in key:
                return payload(params) if callable(payload) else payload
        raise AssertionError(f"unmocked request: {key} {params}")


@pytest.fixture
def router(monkeypatch):
    r = Router()
    monkeypatch.setattr(http, "HOST_DELAYS", {})
    monkeypatch.setattr(http, "request_json", r)
    return r


def item(conn, domain, ns, key):
    """Int id of an item the test expects to exist under this identity."""
    iid = db.lookup_item(conn, domain, ns, key)
    assert iid is not None, f"no item for {domain}:{ns}:{key}"
    return iid


def seed_movie(conn, tmdb_id, title, genres=None, ts=None, year=None, profile="default"):
    item_id = db.resolve_item(conn, "movie", "tmdb", str(tmdb_id), title=title, year=year,
                              meta={"genres": genres} if genres else None)
    db.add_event(conn, ts or iso(1), item_id, "complete", 0.8, "jellyfin", profile=profile)
    return item_id


def movie_result(tmdb_id, vote=7.0):
    return {"id": tmdb_id, "title": f"M{tmdb_id}", "release_date": "2021-06-01",
            "vote_average": vote, "popularity": 12.3, "overview": f"about {tmdb_id}",
            "poster_path": f"/p{tmdb_id}.jpg"}


def candidate_rows(conn):
    return {(r["item_id"], r["source"]): r for r in conn.execute("SELECT * FROM candidates")}


def test_movie_similar_discover_and_exclusions(conn, cfg, router):
    seed = seed_movie(conn, 603, "The Matrix", genres=["Science Fiction", "Action"])
    excluded = {}
    for tmdb_id, reason in ((100, "library"), (101, "reject"), (102, "open_rec")):
        iid = db.resolve_item(conn, "movie", "tmdb", str(tmdb_id), title=f"M{tmdb_id}")
        excluded[reason] = iid
        if reason == "library":
            conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'radarr')", (iid,))
        elif reason == "reject":
            db.add_event(conn, iso(2), iid, "reject", -1.0, "user")
        else:
            conn.execute(
                "INSERT INTO recommendations (run_id, ts, domain, item_id, score)"
                " VALUES ('r1', ?, 'movie', ?, 0.5)", (iso(2), iid))

    router.route("/movie/603/recommendations", {"results": [
        movie_result(100), movie_result(101), movie_result(102), movie_result(200, vote=7.5)]})
    router.route("/movie/603/similar", {"results": [
        movie_result(200, vote=8.0), movie_result(201)]})
    router.route("/genre/movie/list", {"genres": [
        {"id": 878, "name": "Science Fiction"}, {"id": 28, "name": "Action"}]})

    def discover(params):
        if params["sort_by"] == "popularity.desc":
            return {"results": [movie_result(300)]}
        return {"results": []}  # serendipity probes covered in their own tests
    router.route("/discover/movie", discover)

    stats = run(conn, cfg, domain="movie")

    rows = candidate_rows(conn)
    m200 = item(conn, "movie", "tmdb", "200")
    assert rows[(m200, "tmdb_similar")]["external_score"] == 8.0  # max of 7.5 and 8.0
    assert rows[(m200, "tmdb_similar")]["seed_item_id"] == seed
    assert (item(conn, "movie", "tmdb", "201"), "tmdb_similar") in rows
    disc = rows[(item(conn, "movie", "tmdb", "300"), "tmdb_discover")]
    assert disc["seed_item_id"] is None
    for iid in excluded.values():
        assert not any(k[0] == iid for k in rows)

    row = conn.execute("SELECT * FROM items WHERE id=?", (m200,)).fetchone()
    assert row["title"] == "M200" and row["year"] == 2021
    meta = json.loads(row["meta"])
    assert meta["popularity"] == 12.3
    assert meta["overview"] == "about 200"
    assert meta["poster_path"] == "/p200.jpg"

    bubble = [p for u, p in router.calls
              if "/discover/movie" in u and p["sort_by"] == "popularity.desc"]
    assert len(bubble) == 1
    assert set(bubble[0]["with_genres"].split("|")) == {"878", "28"}
    assert bubble[0]["vote_count.gte"] == 100
    # both taste genres sit exactly at the uniform share (never below),
    # so no genre probes fire — only the decade probe
    seren = [p for u, p in router.calls
             if "/discover/movie" in u and p["sort_by"] == "vote_average.desc"]
    assert len(seren) == 1 and "with_genres" not in seren[0]

    assert stats["seeds"]["movie"] == 1
    assert stats["new"] == {"tmdb_similar": 2, "tmdb_discover": 1}
    assert stats["updated"] == {"tmdb_similar": 1}
    assert stats["skipped"] == 3
    assert stats["serendipity"] == 0
    assert stats["profiles"] == 1
    # legacy single-user path: every row belongs to the synthesized default
    assert {r["profile"] for r in conn.execute("SELECT profile FROM candidates")} == {"default"}
    # the genre cache is runtime state, not taste — stays globally keyed
    assert db.get_state(conn, "tmdb:genres:movie") is not None


def test_rerun_idempotent_score_max_and_genre_cache(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix", genres=["Action"])
    router.route("/movie/603/recommendations", {"results": [movie_result(200, vote=7.0)]})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": [{"id": 28, "name": "Action"}]})

    def discover(params):
        if params["sort_by"] == "popularity.desc":
            return {"results": [movie_result(300)]}
        return {"results": []}
    router.route("/discover/movie", discover)

    run(conn, cfg, domain="movie")
    m200 = item(conn, "movie", "tmdb", "200")
    first = conn.execute(
        "SELECT * FROM candidates WHERE item_id=? AND source='tmdb_similar'", (m200,)).fetchone()

    # second run reports a lower score: max() must keep 7.0, first_seen sticks
    router.route("/movie/603/recommendations", {"results": [movie_result(200, vote=6.0)]})
    stats = run(conn, cfg, domain="movie")

    assert conn.execute("SELECT count(*) c FROM candidates").fetchone()["c"] == 2
    again = conn.execute(
        "SELECT * FROM candidates WHERE item_id=? AND source='tmdb_similar'", (m200,)).fetchone()
    assert again["first_seen"] == first["first_seen"]
    assert again["seed_item_id"] == first["seed_item_id"]
    assert again["external_score"] == 7.0
    assert stats["new"] == {}
    assert stats["updated"] == {"tmdb_similar": 1, "tmdb_discover": 1}
    # re-resolving tmdb id 200 must land on the same item row, not mint one
    assert conn.execute(
        "SELECT count(*) c FROM items WHERE domain='movie'").fetchone()["c"] == 3

    genre_calls = [u for u, _ in router.calls if "/genre/movie/list" in u]
    assert len(genre_calls) == 1  # second run served from state cache


def test_cap_200_new_rows_per_source(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix")  # no genres -> no discover pass
    router.route("/movie/603/recommendations",
                 {"results": [movie_result(1000 + i) for i in range(250)]})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", {"results": []})  # serendipity decade probe

    stats = run(conn, cfg, domain="movie")

    n = conn.execute(
        "SELECT count(*) c FROM candidates WHERE source='tmdb_similar'").fetchone()["c"]
    assert n == 200
    assert stats["new"] == {"tmdb_similar": 200}
    assert stats["capped"] == ["tmdb_similar"]
    # capped results must not mint items rank will never see (seed + 200)
    assert conn.execute("SELECT count(*) c FROM items").fetchone()["c"] == 201


def test_artist_lastfm_similar(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed_mb = db.resolve_item(conn, "artist", "mbid", mb, title="Nirvana")
    db.add_event(conn, iso(1), seed_mb, "loved", 1.0, "lastfm")
    seed_name = db.resolve_item(conn, "artist", "name", "Radiohead", title="Radiohead")
    db.add_event(conn, iso(2), seed_name, "loved", 1.0, "lastfm")
    rejected = db.resolve_item(conn, "artist", "name", "Nickelback", title="Nickelback")
    db.add_event(conn, iso(3), rejected, "reject", -1.0, "user")

    def similar(params):
        if params["limit"] == 20:  # hop-2 serendipity fan-out, covered elsewhere
            return {"similarartists": {"artist": []}}
        assert params["limit"] == 50
        if params.get("mbid") == mb:
            return {"similarartists": {"artist": [
                {"name": "Hole", "mbid": "abc-123", "match": "0.87"},
                {"name": "Bush", "mbid": "", "match": "0.5"},
                {"name": "Nickelback", "mbid": "", "match": "0.4"},
            ]}}
        assert params.get("artist") == "Radiohead"  # no mbid -> name lookup
        return {"similarartists": {"artist": {"name": "Thom Yorke", "match": "0.9"}}}

    router.route("lastfm:artist.getsimilar", similar)
    stats = run(conn, cfg, domain="artist")

    rows = candidate_rows(conn)
    hole_id = item(conn, "artist", "mbid", "abc-123")
    hole = rows[(hole_id, "lastfm_similar")]
    assert hole["external_score"] == pytest.approx(0.87)
    assert hole["seed_item_id"] == seed_mb
    assert (item(conn, "artist", "name", "Bush"), "lastfm_similar") in rows
    assert (item(conn, "artist", "name", "Thom Yorke"), "lastfm_similar") in rows
    assert not any(k[0] == rejected for k in rows)
    # the mbid entry also taught the item its name spelling (normalized)
    assert db.identities_of(conn, hole_id) == {"mbid": "abc-123", "name": "hole"}
    assert stats["seeds"]["artist"] == 2
    assert stats["skipped"] == 1


def test_tmdb_result_truncates_overview_and_skips_missing_art():
    full = _tmdb_result("movie", {"id": 42, "title": "M42", "release_date": "2020-01-01",
                                  "overview": "x" * 400, "poster_path": "/p42.jpg",
                                  "vote_average": 7.0})
    assert full is not None
    key, _title, _year, meta, _score = full
    assert key == "42"
    assert meta["overview"] == "x" * 300  # capped so items.meta stays lean
    assert meta["poster_path"] == "/p42.jpg"

    bare = _tmdb_result("movie", {"id": 43, "title": "M43", "poster_path": None})
    assert bare is not None
    assert "poster_path" not in bare[3] and "overview" not in bare[3]


def test_series_seeds_resolve_tmdb_id(conn, cfg, router):
    s1 = db.resolve_item(conn, "series", "tvdb", "81189", title="Breaking Bad",
                         meta={"genres": ["Drama"]})
    db.attach_identity(conn, s1, "tmdb", "1396")
    db.add_event(conn, iso(1), s1, "complete", 0.8, "jellyfin")
    # positive but no tmdb identity anywhere: must be skipped, not fetched
    s2 = db.resolve_item(conn, "series", "tvdb", "999", title="No Tmdb")
    db.add_event(conn, iso(1), s2, "complete", 0.8, "jellyfin")

    router.route("/tv/1396/recommendations", {"results": [
        {"id": 60059, "name": "Better Call Saul", "first_air_date": "2015-02-08",
         "vote_average": 8.6, "popularity": 45.0, "overview": "spinoff",
         "poster_path": "/bcs.jpg"}]})
    router.route("/genre/tv/list", {"genres": [{"id": 18, "name": "Drama"}]})

    def discover(params):
        if params["sort_by"] == "popularity.desc":
            return {"results": [
                {"id": 1399, "name": "Game of Thrones", "first_air_date": "2011-04-17",
                 "vote_average": 8.4}]}
        return {"results": []}
    router.route("/discover/tv", discover)

    stats = run(conn, cfg, domain="series")

    assert stats["seeds"]["series"] == 1
    rows = candidate_rows(conn)
    bcs = item(conn, "series", "tmdb", "60059")
    assert rows[(bcs, "tmdb_similar")]["seed_item_id"] == s1
    assert (item(conn, "series", "tmdb", "1399"), "tmdb_discover") in rows
    row = conn.execute("SELECT * FROM items WHERE id=?", (bcs,)).fetchone()
    assert row["title"] == "Better Call Saul" and row["year"] == 2015
    meta = json.loads(row["meta"])
    assert meta["poster_path"] == "/bcs.jpg" and meta["overview"] == "spinoff"
    assert not any("/tv/999" in u for u, _ in router.calls)
    # tv decade probe uses first_air_date; no positive years -> tie
    # resolves to the earliest decade
    seren = [p for u, p in router.calls
             if "/discover/tv" in u and p["sort_by"] == "vote_average.desc"]
    assert len(seren) == 1
    assert seren[0]["first_air_date.gte"] == "1960-01-01"
    assert seren[0]["first_air_date.lte"] == "1969-12-31"


def test_seed_threshold_and_top25_limit(conn, cfg, router):
    # 25 recent completes (~0.8) beat 5 year-old ones (~0.37, still
    # positive); a skip-only item never seeds. Router raises on any
    # unmocked fetch, so only the 25 winners may fan out.
    for i in range(25):
        seed_movie(conn, 1000 + i, f"R{i}", ts=iso(1))
    for i in range(5):
        seed_movie(conn, 2000 + i, f"O{i}", ts=iso(400))
    neg = db.resolve_item(conn, "movie", "tmdb", "3000", title="Skipped")
    db.add_event(conn, iso(1), neg, "skip", -0.1, "jellyfin")

    empty = {"results": []}
    for i in range(25):
        router.route(f"/movie/{1000 + i}/recommendations", empty)
        router.route(f"/movie/{1000 + i}/similar", empty)
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", empty)  # serendipity decade probe

    stats = run(conn, cfg, domain="movie")

    assert stats["seeds"]["movie"] == 25
    assert not any("/movie/200" in u for u, _ in router.calls)
    assert not any("/movie/3000" in u for u, _ in router.calls)


def test_unconfigured_apis_do_nothing(conn, tmp_path, router):
    bare = C._build({"core": {"data_dir": str(tmp_path)}})
    seed_movie(conn, 603, "The Matrix")
    stats = run(conn, bare)
    assert stats["new"] == {} and router.calls == []


def test_failed_rec_never_reenters_pool(conn, cfg, router):
    # 'failed' is terminal (un-actuatable add): re-proposing would just
    # re-fail in apply, so the item must be blocked at insert time.
    seed_movie(conn, 603, "The Matrix")
    failed = db.resolve_item(conn, "movie", "tmdb", "700", title="M700")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('r1', ?, 'movie', ?, 0.5, 'failed')", (iso(2), failed))
    router.route("/movie/603/recommendations", {"results": [movie_result(700)]})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", {"results": []})

    stats = run(conn, cfg, domain="movie")

    assert candidate_rows(conn) == {}
    assert stats["skipped"] == 1


def add_snoozed_rec(conn, tmdb_id, days_ago):
    iid = db.resolve_item(conn, "movie", "tmdb", str(tmdb_id), title=f"M{tmdb_id}")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status, acted_at)"
        " VALUES ('r1', ?, 'movie', ?, 0.5, 'snoozed', ?)",
        (iso(days_ago), iid, iso(days_ago)))
    return iid


def test_excluded_ids_snooze_window(conn):
    active = add_snoozed_rec(conn, 800, days_ago=SNOOZE_DAYS - 1)
    lapsed = add_snoozed_rec(conn, 801, days_ago=SNOOZE_DAYS + 1)

    excluded = _excluded_ids(conn, "default")

    assert active in excluded
    assert lapsed not in excluded  # rank will expire it; re-proposable


def test_snoozed_item_blocked_only_while_active(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix")
    active = add_snoozed_rec(conn, 800, days_ago=2)
    lapsed = add_snoozed_rec(conn, 801, days_ago=SNOOZE_DAYS + 1)
    router.route("/movie/603/recommendations", {"results": [
        movie_result(800), movie_result(801)]})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", {"results": []})

    stats = run(conn, cfg, domain="movie")

    rows = candidate_rows(conn)
    assert (lapsed, "tmdb_similar") in rows
    assert not any(k[0] == active for k in rows)
    assert stats["skipped"] == 1


def test_excluded_ids_plain_ints_and_profile_scope(conn):
    owned = db.resolve_item(conn, "series", "tvdb", "81189", title="Breaking Bad")
    db.attach_identity(conn, owned, "tmdb", "1396")
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'sonarr')", (owned,))
    rejected = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    db.add_event(conn, iso(1), rejected, "reject", -1.0, "user")

    excluded = _excluded_ids(conn, "default")

    # identity resolution keeps one item per entity, so the int block
    # list needs no alternate-namespace synthesis: every spelling of the
    # owned series resolves to the same excluded id
    assert {owned, rejected} <= excluded
    assert db.lookup_item(conn, "series", "tmdb", "1396") == owned
    # library rows block every profile; the reject only blocks its own
    assert owned in _excluded_ids(conn, "other")
    assert rejected not in _excluded_ids(conn, "other")


def test_owned_series_excluded_across_namespaces(conn, cfg, router):
    # Sonarr keys series by tvdb; TMDb mints tmdb ids — an owned show must
    # not slip back into the pool under the other namespace.
    seed = db.resolve_item(conn, "series", "tvdb", "1", title="Seed")
    db.attach_identity(conn, seed, "tmdb", "500")
    db.add_event(conn, iso(1), seed, "complete", 0.8, "jellyfin")
    owned = db.resolve_item(conn, "series", "tvdb", "81189", title="Breaking Bad")
    db.attach_identity(conn, owned, "tmdb", "1396")
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'sonarr')", (owned,))
    router.route("/tv/500/recommendations", {"results": [
        {"id": 1396, "name": "Breaking Bad", "first_air_date": "2008-01-20",
         "vote_average": 8.9},
        {"id": 60059, "name": "Better Call Saul", "first_air_date": "2015-02-08",
         "vote_average": 8.6},
    ]})
    router.route("/genre/tv/list", {"genres": []})
    router.route("/discover/tv", {"results": []})

    stats = run(conn, cfg, domain="series")

    rows = candidate_rows(conn)
    assert (item(conn, "series", "tmdb", "60059"), "tmdb_similar") in rows
    assert not any(k[0] == owned for k in rows)
    assert stats["skipped"] == 1


def test_lastfm_mbid_merges_name_keyed_twin(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = db.resolve_item(conn, "artist", "mbid", mb, title="Nirvana")
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    # scrobbles minted a name-keyed twin before any mbid was known
    twin = db.resolve_item(conn, "artist", "name", "Hole", title="Hole")
    db.add_event(conn, iso(2), twin, "scrobble", 0.3, "lastfm")

    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "Hole", "mbid": "abc-123", "match": "0.87"}]}})
    run(conn, cfg, domain="artist")

    winner = item(conn, "artist", "mbid", "abc-123")
    # the twin merged into the mbid holder: its row is gone, both
    # identities and its history now point at the winner
    assert winner != twin
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (twin,)).fetchone() is None
    assert db.lookup_item(conn, "artist", "name", "Hole") == winner
    ev = conn.execute("SELECT item_id FROM events WHERE kind='scrobble'").fetchone()
    assert ev["item_id"] == winner
    rows = candidate_rows(conn)
    assert (winner, "lastfm_similar") in rows
    assert not any(k[0] == twin for k in rows)


def test_name_keyed_entry_lands_on_existing_item(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = db.resolve_item(conn, "artist", "mbid", mb, title="Nirvana")
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    # enrich already taught the canonical item its name; a similar-artist
    # entry without mbid (any spelling of it) must resolve to that row
    # instead of minting a twin
    canon = db.resolve_item(conn, "artist", "mbid", "xyz-1", title="Bush")
    db.attach_identity(conn, canon, "name", "Bush")

    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "BUSH", "mbid": "", "match": "0.5"}]}})
    stats = run(conn, cfg, domain="artist")

    rows = candidate_rows(conn)
    assert (canon, "lastfm_similar") in rows
    assert stats["new"] == {"lastfm_similar": 1}
    assert conn.execute(
        "SELECT count(*) c FROM items WHERE domain='artist'").fetchone()["c"] == 2


def test_rejected_artist_stays_excluded_when_mbid_appears(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = db.resolve_item(conn, "artist", "mbid", mb, title="Nirvana")
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    rejected = db.resolve_item(conn, "artist", "name", "Nickelback", title="Nickelback")
    db.add_event(conn, iso(2), rejected, "reject", -1.0, "user")

    # the reject lives on the name-keyed item; the freshly revealed mbid
    # must not smuggle the artist back into the pool post-merge
    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "Nickelback", "mbid": "nb-1", "match": "0.4"}]}})
    stats = run(conn, cfg, domain="artist")

    assert candidate_rows(conn) == {}
    assert stats["skipped"] == 1
    winner = item(conn, "artist", "mbid", "nb-1")
    assert db.lookup_item(conn, "artist", "name", "Nickelback") == winner
    ev = conn.execute("SELECT item_id FROM events WHERE kind='reject'").fetchone()
    assert ev["item_id"] == winner


# ── top albums ───────────────────────────────────────────────────────


def seed_artist(conn, name, mb=None, ts=None):
    if mb:
        iid = db.resolve_item(conn, "artist", "mbid", mb, title=name)
    else:
        iid = db.resolve_item(conn, "artist", "name", name, title=name)
    db.add_event(conn, ts or iso(1), iid, "loved", 1.0, "lastfm")
    return iid


def album_entry(mbid, name, playcount, artist="Nirvana", artist_mbid=""):
    return {"name": name, "mbid": mbid, "playcount": playcount,
            "artist": {"name": artist, "mbid": artist_mbid}}


def test_album_top_albums_fanout_rank_normalized(conn, cfg, router):
    mb1 = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    mb2 = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
    seed1 = seed_artist(conn, "Nirvana", mb=mb1, ts=iso(1))
    seed2 = seed_artist(conn, "Radiohead", mb=mb2, ts=iso(2))
    seed_artist(conn, "Name Only", ts=iso(3))  # no mbid anywhere: never queried
    owned = db.resolve_item(conn, "album", "mbid", "alb-own", title="Owned")
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'lidarr')", (owned,))

    def top_albums(params):
        assert params["limit"] == 8
        if params["mbid"] == mb1:
            return {"topalbums": {"album": [
                album_entry("alb-1", "Nevermind", 900, artist_mbid=mb1),
                album_entry("", "No Mbid", 800),  # unactionable, but keeps its rank
                album_entry("alb-2", "In Utero", 700),  # no artist mbid: seed's is used
                album_entry("alb-own", "Owned", 600, artist_mbid=mb1),
            ]}}
        assert params["mbid"] == mb2
        return {"topalbums": {"album":  # single-element list collapse
            album_entry("alb-3", "OK Computer", 500, artist="Radiohead", artist_mbid=mb2)}}

    router.route("lastfm:artist.gettopalbums", top_albums)
    stats = run(conn, cfg, domain="album")

    # mbid seeds queried best-label first; name-only artist skipped entirely
    assert [p["mbid"] for _, p in router.calls] == [mb1, mb2]
    assert stats["seeds"]["album"] == 2

    rows = candidate_rows(conn)
    alb1 = item(conn, "album", "mbid", "alb-1")
    top = rows[(alb1, "lastfm_top_albums")]
    assert top["external_score"] == pytest.approx(1.0)  # playcount #1
    assert top["seed_item_id"] == seed1
    # rank includes the skipped mbid-less entry: In Utero is playcount #3
    alb2 = item(conn, "album", "mbid", "alb-2")
    assert rows[(alb2, "lastfm_top_albums")]["external_score"] == pytest.approx(0.8)
    third = rows[(item(conn, "album", "mbid", "alb-3"), "lastfm_top_albums")]
    assert third["external_score"] == pytest.approx(1.0)
    assert third["seed_item_id"] == seed2
    assert not any(k[0] == owned for k in rows)  # item exclusions apply unchanged
    albums = [k for k in rows if k[1] == "lastfm_top_albums"]
    assert len(albums) == 3

    row = conn.execute("SELECT * FROM items WHERE id=?", (alb1,)).fetchone()
    assert row["domain"] == "album" and row["title"] == "Nevermind"
    # the album's own mbid is its identity; artist_mbid is a relation to
    # another item, so it lives in meta only
    assert db.identities_of(conn, alb1) == {"mbid": "alb-1"}
    meta = json.loads(row["meta"])
    assert meta["artist"] == "Nirvana" and meta["artist_mbid"] == mb1
    # entry carried no artist mbid: the seed artist's fills in
    in_utero = conn.execute("SELECT meta FROM items WHERE id=?", (alb2,)).fetchone()
    assert json.loads(in_utero["meta"])["artist_mbid"] == mb1
    # the mbid-less entry never became an item
    assert conn.execute(
        "SELECT count(*) c FROM items WHERE domain='album'").fetchone()["c"] == 4

    assert stats["new"] == {"lastfm_top_albums": 3}
    assert stats["skipped"] == 1


def test_album_cap_100_and_rank_floor(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed_artist(conn, "Nirvana", mb=mb)
    # oversized payload (the live API honors limit=8; the cap is our guard)
    router.route("lastfm:artist.gettopalbums", {"topalbums": {"album": [
        album_entry(f"alb-{i}", f"A{i}", 1000 - i, artist_mbid=mb) for i in range(120)]}})

    stats = run(conn, cfg, domain="album")

    n = conn.execute(
        "SELECT count(*) c FROM candidates WHERE source='lastfm_top_albums'").fetchone()["c"]
    assert n == 100
    assert stats["capped"] == ["lastfm_top_albums"]
    rows = candidate_rows(conn)
    # deep ranks clamp at the 0.3 floor instead of going negative
    assert rows[(item(conn, "album", "mbid", "alb-50"), "lastfm_top_albums")][
        "external_score"] == pytest.approx(0.3)


def test_album_domain_in_default_run(conn, cfg, router):
    # a full run (no domain filter) reaches the album source; the artist
    # sources fire too since the same key configures them
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed_artist(conn, "Nirvana", mb=mb)
    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": []}})
    router.route("lastfm:artist.gettopalbums", {"topalbums": {"album": [
        album_entry("alb-1", "Nevermind", 900, artist_mbid=mb)]}})

    stats = run(conn, cfg)

    assert stats["new"] == {"lastfm_top_albums": 1}
    assert (item(conn, "album", "mbid", "alb-1"), "lastfm_top_albums") in candidate_rows(conn)


# ── serendipity ──────────────────────────────────────────────────────


def test_serendipity_prefers_under_represented_genres(conn, cfg, router):
    # 6 action + 1 comedy positives against a 5-genre map: uniform share
    # is 1/5; action (6/7) is over, comedy (1/7) is under, the rest absent
    for i in range(6):
        seed_movie(conn, 600 + i, f"A{i}", genres=["Action"])
    seed_movie(conn, 610, "C0", genres=["Comedy"])
    router.route("/genre/movie/list", {"genres": [
        {"id": 28, "name": "Action"}, {"id": 35, "name": "Comedy"},
        {"id": 18, "name": "Drama"}, {"id": 14, "name": "Fantasy"},
        {"id": 27, "name": "Horror"}]})
    empty = {"results": []}
    for tmdb_id in (*range(600, 606), 610):
        router.route(f"/movie/{tmdb_id}/recommendations", empty)
        router.route(f"/movie/{tmdb_id}/similar", empty)

    def discover(params):
        if params["sort_by"] == "popularity.desc":
            return empty
        if params.get("with_genres") == "14":
            return {"results": [movie_result(900, vote=8.2)]}
        return empty
    router.route("/discover/movie", discover)

    stats = run(conn, cfg, domain="movie")

    seren = [p for u, p in router.calls
             if "/discover/movie" in u and p["sort_by"] == "vote_average.desc"]
    probes = [p for p in seren if "with_genres" in p]
    # absent genres first (rarest, tie by name: drama/fantasy/horror),
    # then once-seen comedy; over-represented action never probed
    assert [p["with_genres"] for p in probes] == ["18", "14", "27", "35"]
    assert all(p["vote_count.gte"] == 500 and p["page"] == 1 for p in seren)
    rows = candidate_rows(conn)
    row = rows[(item(conn, "movie", "tmdb", "900"), "serendipity_tmdb")]
    assert row["external_score"] == 8.2
    assert row["seed_item_id"] is None
    assert stats["new"]["serendipity_tmdb"] == 1
    assert stats["serendipity"] == 1


def test_serendipity_decade_probe_targets_least_seen_decade(conn, cfg, router):
    # one positive per decade except the 2010s -> probe the 2010s
    for i, year in enumerate((1965, 1975, 1985, 1995, 2005)):
        seed_movie(conn, 700 + i, f"D{i}", year=year)
    owned = db.resolve_item(conn, "movie", "tmdb", "902", title="M902")
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'radarr')", (owned,))
    empty = {"results": []}
    for i in range(5):
        router.route(f"/movie/{700 + i}/recommendations", empty)
        router.route(f"/movie/{700 + i}/similar", empty)
    router.route("/genre/movie/list", {"genres": []})

    def discover(params):
        assert params["sort_by"] == "vote_average.desc"  # only serendipity discovers here
        return {"results": [movie_result(901, vote=8.4), movie_result(902)]}
    router.route("/discover/movie", discover)

    stats = run(conn, cfg, domain="movie")

    probes = [p for u, p in router.calls if "/discover/movie" in u]
    assert len(probes) == 1  # empty genre map -> decade probe only
    assert probes[0]["primary_release_date.gte"] == "2010-01-01"
    assert probes[0]["primary_release_date.lte"] == "2019-12-31"
    assert probes[0]["vote_count.gte"] == 500
    rows = candidate_rows(conn)
    m901 = item(conn, "movie", "tmdb", "901")
    assert rows[(m901, "serendipity_tmdb")]["external_score"] == 8.4
    assert not any(k[0] == owned for k in rows)  # exclusions apply unchanged
    assert stats["skipped"] == 1
    assert stats["serendipity"] == 1


def test_serendipity_skipped_without_positives(conn, cfg, router):
    # nothing positive yet: no taste baseline to diverge from
    neg = db.resolve_item(conn, "movie", "tmdb", "3000", title="Skipped")
    db.add_event(conn, iso(1), neg, "skip", -0.1, "jellyfin")

    stats = run(conn, cfg, domain="movie")

    assert router.calls == []
    assert stats["serendipity"] == 0


def test_serendipity_cap_100_new_rows(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix", genres=["Action"])
    router.route("/movie/603/recommendations", {"results": []})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": [
        {"id": 28, "name": "Action"}, {"id": 35, "name": "Comedy"}]})

    def discover(params):
        if params["sort_by"] == "popularity.desc":
            return {"results": []}
        if params.get("with_genres") == "35":
            return {"results": [movie_result(5000 + i) for i in range(80)]}
        return {"results": [movie_result(6000 + i) for i in range(80)]}
    router.route("/discover/movie", discover)

    stats = run(conn, cfg, domain="movie")

    n = conn.execute(
        "SELECT count(*) c FROM candidates WHERE source='serendipity_tmdb'").fetchone()["c"]
    assert n == 100  # 160 fetched across genre + decade probes, tighter cap holds
    assert stats["capped"] == ["serendipity_tmdb"]
    assert stats["serendipity"] == 100


def test_serendipity_lastfm_two_hop_damped_and_deduped(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed_artist(conn, "Nirvana", mb=mb)

    def similar(params):
        if params.get("mbid") == mb:
            assert params["limit"] == 50
            return {"similarartists": {"artist": [
                {"name": "Hole", "mbid": "h-1", "match": "0.9"},
                {"name": "Melvins", "mbid": "m-1", "match": "0.7"},
            ]}}
        assert params["limit"] == 20  # hop-2 fan-out
        if params.get("mbid") == "h-1":
            return {"similarartists": {"artist": [
                {"name": "Babes in Toyland", "mbid": "b-1", "match": "0.6"},
                {"name": "Melvins", "mbid": "m-1", "match": "0.95"},
            ]}}
        assert params.get("mbid") == "m-1"
        return {"similarartists": {"artist": []}}

    router.route("lastfm:artist.getsimilar", similar)
    stats = run(conn, cfg, domain="artist")

    hop2 = [p for _, p in router.calls if p.get("limit") == 20]
    assert [p["mbid"] for p in hop2] == ["h-1", "m-1"]  # best hop-1 first
    rows = candidate_rows(conn)
    babes = rows[(item(conn, "artist", "mbid", "b-1"), "serendipity_lastfm")]
    assert babes["external_score"] == pytest.approx(0.48)  # 0.6 match damped by 0.8
    assert babes["seed_item_id"] == item(conn, "artist", "mbid", "h-1")
    # melvins is hop-1 reachable: bubble-adjacent, never serendipity
    melvins = item(conn, "artist", "mbid", "m-1")
    assert (melvins, "lastfm_similar") in rows
    assert (melvins, "serendipity_lastfm") not in rows
    assert stats["new"] == {"lastfm_similar": 2, "serendipity_lastfm": 1}
    assert stats["serendipity"] == 1


def test_serendipity_hop1_skips_library_seeds_and_caps_at_5(conn, cfg, router):
    # pre-existing hop-1 pool from earlier runs: 8 lastfm_similar rows
    def hop1(name, score):
        iid = db.resolve_item(conn, "artist", "name", name, title=name)
        conn.execute(
            "INSERT INTO candidates (item_id, source, external_score, first_seen, last_seen)"
            " VALUES (?, 'lastfm_similar', ?, ?, ?)", (iid, score, iso(3), iso(3)))
        return iid

    owned = hop1("Owned", 0.95)
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'lidarr')", (owned,))
    loved = hop1("Loved", 0.9)
    db.add_event(conn, iso(1), loved, "loved", 1.0, "lastfm")  # candidate turned seed
    for i, score in enumerate((0.8, 0.7, 0.6, 0.5, 0.4, 0.3)):
        hop1(f"N{i}", score)
    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": []}})

    run(conn, cfg, domain="artist")

    hop2 = [p["artist"] for _, p in router.calls if p.get("limit") == 20]
    assert hop2 == ["N0", "N1", "N2", "N3", "N4"]  # library/seed skipped, top 5 by score


def test_serendipity_honors_domain_filter(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix")
    artist = db.resolve_item(conn, "artist", "name", "Nirvana", title="Nirvana")
    db.add_event(conn, iso(1), artist, "loved", 1.0, "lastfm")
    hop = db.resolve_item(conn, "artist", "name", "Hole", title="Hole")
    conn.execute(
        "INSERT INTO candidates (item_id, source, external_score, first_seen, last_seen)"
        " VALUES (?, 'lastfm_similar', 0.9, ?, ?)", (hop, iso(3), iso(3)))
    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": []}})

    run(conn, cfg, domain="artist")
    assert router.calls and not any("themoviedb" in u for u, _ in router.calls)

    router.calls.clear()
    router.route("/movie/603/recommendations", {"results": []})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", {"results": []})
    run(conn, cfg, domain="movie")
    assert router.calls and not any("audioscrobbler" in u for u, _ in router.calls)


# ── multi-profile ────────────────────────────────────────────────────


def test_two_profiles_independent_seeds_and_exclusions(conn, tmp_path, router):
    cfg = C._build({
        "core": {"data_dir": str(tmp_path)},
        "tmdb": {"api_key": "tk"},
        "profiles": {"alice": {"jellyfin_user": "a"}, "bob": {"jellyfin_user": "b"}},
    })
    seed_movie(conn, 603, "The Matrix", profile="alice")
    seed_movie(conn, 700, "Heat", profile="bob")
    # bob rejected 200; alice holds no grudge, so it may still reach her pool
    rejected = db.resolve_item(conn, "movie", "tmdb", "200", title="M200")
    db.add_event(conn, iso(2), rejected, "reject", -1.0, "user", profile="bob")

    router.route("/movie/603/recommendations",
                 {"results": [movie_result(200), movie_result(201)]})
    router.route("/movie/700/recommendations",
                 {"results": [movie_result(200), movie_result(202)]})
    empty = {"results": []}
    router.route("/movie/603/similar", empty)
    router.route("/movie/700/similar", empty)
    router.route("/genre/movie/list", {"genres": []})
    router.route("/discover/movie", empty)

    stats = run(conn, cfg, domain="movie")

    rows = {(r["profile"], r["item_id"]) for r in conn.execute(
        "SELECT profile, item_id FROM candidates WHERE source='tmdb_similar'")}
    assert rows == {
        ("alice", rejected), ("alice", item(conn, "movie", "tmdb", "201")),
        ("bob", item(conn, "movie", "tmdb", "202")),
    }
    # each profile fanned out only from its own seed, exactly once
    recs = [u for u, _ in router.calls if u.endswith("/recommendations")]
    assert recs.count(f"{TMDB}/movie/603/recommendations") == 1
    assert recs.count(f"{TMDB}/movie/700/recommendations") == 1
    assert stats["profiles"] == 2
    assert stats["seeds"]["movie"] == 2
    assert stats["skipped"] == 1  # bob's reject blocked 200 for bob alone
    assert stats["new"] == {"tmdb_similar": 3}
