"""Offline tests for candidates.run: seeding, fan-out, exclusions, caps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gustarr import config as C
from gustarr import db, http, ids
from gustarr.candidates import _excluded_ids, run

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


def seed_movie(conn, tmdb_id, title, genres=None, ts=None):
    item_id = ids.make("movie", "tmdb", str(tmdb_id))
    db.upsert_item(conn, item_id, "movie", title=title, ids={"tmdb": tmdb_id},
                   meta={"genres": genres} if genres else None)
    db.add_event(conn, ts or iso(1), item_id, "complete", 0.8, "jellyfin")
    return item_id


def movie_result(tmdb_id, vote=7.0):
    return {"id": tmdb_id, "title": f"M{tmdb_id}", "release_date": "2021-06-01",
            "vote_average": vote, "popularity": 12.3, "overview": f"about {tmdb_id}"}


def candidate_rows(conn):
    return {(r["item_id"], r["source"]): r for r in conn.execute("SELECT * FROM candidates")}


def test_movie_similar_discover_and_exclusions(conn, cfg, router):
    seed = seed_movie(conn, 603, "The Matrix", genres=["Science Fiction", "Action"])
    excluded = {}
    for tmdb_id, reason in ((100, "library"), (101, "reject"), (102, "open_rec")):
        iid = ids.make("movie", "tmdb", str(tmdb_id))
        db.upsert_item(conn, iid, "movie", title=f"M{tmdb_id}")
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
    router.route("/discover/movie", {"results": [movie_result(300)]})

    stats = run(conn, cfg, domain="movie")

    rows = candidate_rows(conn)
    m200 = ids.make("movie", "tmdb", "200")
    assert rows[(m200, "tmdb_similar")]["external_score"] == 8.0  # max of 7.5 and 8.0
    assert rows[(m200, "tmdb_similar")]["seed_item_id"] == seed
    assert (ids.make("movie", "tmdb", "201"), "tmdb_similar") in rows
    disc = rows[(ids.make("movie", "tmdb", "300"), "tmdb_discover")]
    assert disc["seed_item_id"] is None
    for iid in excluded.values():
        assert not any(k[0] == iid for k in rows)

    item = conn.execute("SELECT * FROM items WHERE id=?", (m200,)).fetchone()
    assert item["title"] == "M200" and item["year"] == 2021
    assert '"popularity"' in item["meta"] and '"overview"' in item["meta"]

    discover_calls = [p for u, p in router.calls if "/discover/movie" in u]
    assert len(discover_calls) == 1
    assert set(discover_calls[0]["with_genres"].split("|")) == {"878", "28"}
    assert discover_calls[0]["vote_count.gte"] == 100
    assert discover_calls[0]["sort_by"] == "popularity.desc"

    assert stats["seeds"]["movie"] == 1
    assert stats["new"] == {"tmdb_similar": 2, "tmdb_discover": 1}
    assert stats["updated"] == {"tmdb_similar": 1}
    assert stats["skipped"] == 3
    assert db.get_state(conn, "tmdb:genres:movie") is not None


def test_rerun_idempotent_score_max_and_genre_cache(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix", genres=["Action"])
    router.route("/movie/603/recommendations", {"results": [movie_result(200, vote=7.0)]})
    router.route("/movie/603/similar", {"results": []})
    router.route("/genre/movie/list", {"genres": [{"id": 28, "name": "Action"}]})
    router.route("/discover/movie", {"results": [movie_result(300)]})

    run(conn, cfg, domain="movie")
    m200 = ids.make("movie", "tmdb", "200")
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

    genre_calls = [u for u, _ in router.calls if "/genre/movie/list" in u]
    assert len(genre_calls) == 1  # second run served from state cache


def test_cap_200_new_rows_per_source(conn, cfg, router):
    seed_movie(conn, 603, "The Matrix")  # no genres -> no discover pass
    router.route("/movie/603/recommendations",
                 {"results": [movie_result(1000 + i) for i in range(250)]})
    router.route("/movie/603/similar", {"results": []})

    stats = run(conn, cfg, domain="movie")

    n = conn.execute(
        "SELECT count(*) c FROM candidates WHERE source='tmdb_similar'").fetchone()["c"]
    assert n == 200
    assert stats["new"] == {"tmdb_similar": 200}
    assert stats["capped"] == ["tmdb_similar"]


def test_artist_lastfm_similar(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed_mb = ids.make("artist", "mbid", mb)
    db.upsert_item(conn, seed_mb, "artist", title="Nirvana", ids={"mbid": mb})
    db.add_event(conn, iso(1), seed_mb, "loved", 1.0, "lastfm")
    seed_name = ids.make("artist", "lastfm", "Radiohead")
    db.upsert_item(conn, seed_name, "artist", title="Radiohead")
    db.add_event(conn, iso(2), seed_name, "loved", 1.0, "lastfm")
    rejected = ids.make("artist", "lastfm", "Nickelback")
    db.upsert_item(conn, rejected, "artist", title="Nickelback")
    db.add_event(conn, iso(3), rejected, "reject", -1.0, "user")

    def similar(params):
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
    hole = rows[(ids.make("artist", "mbid", "abc-123"), "lastfm_similar")]
    assert hole["external_score"] == pytest.approx(0.87)
    assert hole["seed_item_id"] == seed_mb
    assert (ids.make("artist", "lastfm", "Bush"), "lastfm_similar") in rows
    assert (ids.make("artist", "lastfm", "Thom Yorke"), "lastfm_similar") in rows
    assert not any(k[0] == rejected for k in rows)
    assert conn.execute(
        "SELECT ids FROM items WHERE id=?",
        (ids.make("artist", "mbid", "abc-123"),)).fetchone()["ids"] == '{"mbid": "abc-123"}'
    assert stats["seeds"]["artist"] == 2
    assert stats["skipped"] == 1


def test_series_seeds_resolve_tmdb_id(conn, cfg, router):
    s1 = ids.make("series", "tvdb", "81189")
    db.upsert_item(conn, s1, "series", title="Breaking Bad",
                   ids={"tvdb": 81189, "tmdb": 1396}, meta={"genres": ["Drama"]})
    db.add_event(conn, iso(1), s1, "complete", 0.8, "jellyfin")
    # positive but no tmdb id anywhere: must be skipped, not fetched
    s2 = ids.make("series", "tvdb", "999")
    db.upsert_item(conn, s2, "series", title="No Tmdb", ids={"tvdb": 999})
    db.add_event(conn, iso(1), s2, "complete", 0.8, "jellyfin")

    router.route("/tv/1396/recommendations", {"results": [
        {"id": 60059, "name": "Better Call Saul", "first_air_date": "2015-02-08",
         "vote_average": 8.6, "popularity": 45.0, "overview": "spinoff"}]})
    router.route("/genre/tv/list", {"genres": [{"id": 18, "name": "Drama"}]})
    router.route("/discover/tv", {"results": [
        {"id": 1399, "name": "Game of Thrones", "first_air_date": "2011-04-17",
         "vote_average": 8.4}]})

    stats = run(conn, cfg, domain="series")

    assert stats["seeds"]["series"] == 1
    rows = candidate_rows(conn)
    bcs = ids.make("series", "tmdb", "60059")
    assert rows[(bcs, "tmdb_similar")]["seed_item_id"] == s1
    assert (ids.make("series", "tmdb", "1399"), "tmdb_discover") in rows
    item = conn.execute("SELECT * FROM items WHERE id=?", (bcs,)).fetchone()
    assert item["title"] == "Better Call Saul" and item["year"] == 2015
    assert not any("/tv/999" in u for u, _ in router.calls)


def test_seed_threshold_and_top25_limit(conn, cfg, router):
    # 25 recent completes (~0.8) beat 5 year-old ones (~0.37, still
    # positive); a skip-only item never seeds. Router raises on any
    # unmocked fetch, so only the 25 winners may fan out.
    for i in range(25):
        seed_movie(conn, 1000 + i, f"R{i}", ts=iso(1))
    for i in range(5):
        seed_movie(conn, 2000 + i, f"O{i}", ts=iso(400))
    neg = ids.make("movie", "tmdb", "3000")
    db.upsert_item(conn, neg, "movie", title="Skipped")
    db.add_event(conn, iso(1), neg, "skip", -0.1, "jellyfin")

    empty = {"results": []}
    for i in range(25):
        router.route(f"/movie/{1000 + i}/recommendations", empty)
        router.route(f"/movie/{1000 + i}/similar", empty)

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
    failed = ids.make("movie", "tmdb", "700")
    db.upsert_item(conn, failed, "movie", title="M700")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('r1', ?, 'movie', ?, 0.5, 'failed')", (iso(2), failed))
    router.route("/movie/603/recommendations", {"results": [movie_result(700)]})
    router.route("/movie/603/similar", {"results": []})

    stats = run(conn, cfg, domain="movie")

    assert candidate_rows(conn) == {}
    assert stats["skipped"] == 1


def test_excluded_ids_cover_alternate_namespaces(conn):
    owned = ids.make("series", "tvdb", "81189")
    db.upsert_item(conn, owned, "series", title="Breaking Bad",
                   ids={"tvdb": 81189, "tmdb": 1396})
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'sonarr')", (owned,))
    rejected = ids.make("movie", "tmdb", "603")
    db.upsert_item(conn, rejected, "movie", title="The Matrix",
                   ids={"tmdb": 603, "imdb": "tt0133093"})
    db.add_event(conn, iso(1), rejected, "reject", -1.0, "user")

    excluded = _excluded_ids(conn)

    assert {owned, ids.make("series", "tmdb", "1396"),
            rejected, ids.make("movie", "imdb", "tt0133093")} <= excluded


def test_owned_series_excluded_across_namespaces(conn, cfg, router):
    # Sonarr keys series by tvdb; TMDb mints tmdb ids — an owned show must
    # not slip back into the pool under the other namespace.
    seed = ids.make("series", "tvdb", "1")
    db.upsert_item(conn, seed, "series", title="Seed", ids={"tvdb": 1, "tmdb": 500})
    db.add_event(conn, iso(1), seed, "complete", 0.8, "jellyfin")
    owned = ids.make("series", "tvdb", "81189")
    db.upsert_item(conn, owned, "series", title="Breaking Bad",
                   ids={"tvdb": 81189, "tmdb": 1396})
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'sonarr')", (owned,))
    router.route("/tv/500/recommendations", {"results": [
        {"id": 1396, "name": "Breaking Bad", "first_air_date": "2008-01-20",
         "vote_average": 8.9},
        {"id": 60059, "name": "Better Call Saul", "first_air_date": "2015-02-08",
         "vote_average": 8.6},
    ]})

    stats = run(conn, cfg, domain="series")

    rows = candidate_rows(conn)
    assert (ids.make("series", "tmdb", "60059"), "tmdb_similar") in rows
    assert not any(k[0] == ids.make("series", "tmdb", "1396") for k in rows)
    assert stats["skipped"] == 1


def test_lastfm_mbid_merges_name_keyed_twin(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = ids.make("artist", "mbid", mb)
    db.upsert_item(conn, seed, "artist", title="Nirvana", ids={"mbid": mb})
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    # scrobbles minted a name-keyed twin before any mbid was known
    twin = ids.make("artist", "lastfm", "Hole")
    db.upsert_item(conn, twin, "artist", title="Hole")
    db.add_event(conn, iso(2), twin, "scrobble", 0.3, "lastfm")

    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "Hole", "mbid": "abc-123", "match": "0.87"}]}})
    run(conn, cfg, domain="artist")

    canon = ids.make("artist", "mbid", "abc-123")
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (twin,)).fetchone() is None
    assert db.canonical_id(conn, twin) == canon
    ev = conn.execute("SELECT item_id FROM events WHERE kind='scrobble'").fetchone()
    assert ev["item_id"] == canon
    rows = candidate_rows(conn)
    assert (canon, "lastfm_similar") in rows
    assert not any(k[0] == twin for k in rows)


def test_candidate_insert_follows_alias(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = ids.make("artist", "mbid", mb)
    db.upsert_item(conn, seed, "artist", title="Nirvana", ids={"mbid": mb})
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    # enrich previously merged the name-keyed item away; a similar-artist
    # entry without mbid re-mints the fallback id, which must redirect to
    # the canonical row instead of FK-failing on the candidates insert.
    fallback = ids.make("artist", "lastfm", "Bush")
    canon = ids.make("artist", "mbid", "xyz-1")
    db.upsert_item(conn, fallback, "artist", title="Bush")
    db.merge_item(conn, fallback, canon)

    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "Bush", "mbid": "", "match": "0.5"}]}})
    stats = run(conn, cfg, domain="artist")

    rows = candidate_rows(conn)
    assert (canon, "lastfm_similar") in rows
    assert not any(k[0] == fallback for k in rows)
    assert stats["new"] == {"lastfm_similar": 1}


def test_rejected_artist_stays_excluded_when_mbid_appears(conn, cfg, router):
    mb = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    seed = ids.make("artist", "mbid", mb)
    db.upsert_item(conn, seed, "artist", title="Nirvana", ids={"mbid": mb})
    db.add_event(conn, iso(1), seed, "loved", 1.0, "lastfm")
    rejected = ids.make("artist", "lastfm", "Nickelback")
    db.upsert_item(conn, rejected, "artist", title="Nickelback")
    db.add_event(conn, iso(2), rejected, "reject", -1.0, "user")

    # the reject lives on the name-keyed id; the freshly revealed mbid
    # must not smuggle the artist back into the pool post-merge
    router.route("lastfm:artist.getsimilar", {"similarartists": {"artist": [
        {"name": "Nickelback", "mbid": "nb-1", "match": "0.4"}]}})
    stats = run(conn, cfg, domain="artist")

    assert candidate_rows(conn) == {}
    assert stats["skipped"] == 1
    canon = ids.make("artist", "mbid", "nb-1")
    assert db.canonical_id(conn, rejected) == canon
    ev = conn.execute("SELECT item_id FROM events WHERE kind='reject'").fetchone()
    assert ev["item_id"] == canon
