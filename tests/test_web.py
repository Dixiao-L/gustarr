"""Web approval UI: FastAPI layer over a fabricated store."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gustarr import config as C
from gustarr import db
from gustarr.web.app import create_app

MATRIX = "movie:tmdb:603"
BLADE = "movie:tmdb:78"
RADIOHEAD = "artist:mbid:a74b1b7f"


@pytest.fixture
def web(tmp_path):
    # TestClient sends Host: testserver, which the Host guard would 403;
    # allowlist it through config rather than hardcoding it in prod code.
    cfg = C._build({"core": {"data_dir": str(tmp_path)},
                    "web": {"allowed_hosts": ["testserver"]}})
    conn = db.connect(cfg.db_path)
    db.upsert_item(conn, MATRIX, "movie", "The Matrix", 1999,
                   {"tmdb": 603}, {"genres": ["Action", "Science Fiction"]})
    db.upsert_item(conn, BLADE, "movie", "Blade Runner", 1982,
                   {"tmdb": 78}, {"genres": ["Science Fiction"]})
    db.upsert_item(conn, RADIOHEAD, "artist", "Radiohead", None,
                   {"mbid": "a74b1b7f"}, {"genres": ["rock"]})
    ts = db.now()
    why = json.dumps({"neighbors": [[BLADE, 0.82]], "sources": ["tmdb_similar"]})
    movie_rec = conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why)"
        " VALUES ('r1',?,?,?,?,?)", (ts, "movie", MATRIX, 0.91, why)).lastrowid
    artist_rec = conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why)"
        " VALUES ('r1',?,?,?,?,?)",
        (ts, "artist", RADIOHEAD, 0.62, '{"sources": ["lastfm_similar"]}')).lastrowid
    conn.commit()
    conn.close()
    return TestClient(create_app(cfg)), cfg, movie_rec, artist_rec


def _fetch(cfg, sql, *params):
    conn = db.connect(cfg.db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_list_recs(web):
    client, _, movie_rec, artist_rec = web
    resp = client.get("/api/recs")
    assert resp.status_code == 200
    rows = resp.json()
    assert {r["id"] for r in rows} == {movie_rec, artist_rec}
    assert all(r["status"] == "proposed" for r in rows)
    by_id = {r["id"]: r for r in rows}
    assert by_id[movie_rec]["title"] == "The Matrix"
    assert by_id[movie_rec]["year"] == 1999
    assert by_id[movie_rec]["genres"] == ["Action", "Science Fiction"]

    movies = client.get("/api/recs", params={"domain": "movie"}).json()
    assert [r["id"] for r in movies] == [movie_rec]
    # empty domain param means "all domains"
    assert len(client.get("/api/recs", params={"domain": ""}).json()) == 2


def test_approve_flips_status_and_records_event(web):
    client, cfg, movie_rec, _ = web
    resp = client.post(f"/api/recs/{movie_rec}/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"

    (row,) = _fetch(cfg, "SELECT status, acted_at FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "approved"
    assert row["acted_at"]
    events = _fetch(cfg, "SELECT weight, source FROM events WHERE item_id=? AND kind='approve'",
                    MATRIX)
    assert len(events) == 1
    assert events[0]["weight"] == 1.0
    assert events[0]["source"] == "user"


def test_reject_flips_status_and_records_event(web):
    client, cfg, _, artist_rec = web
    assert client.post(f"/api/recs/{artist_rec}/reject").status_code == 200
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", artist_rec)
    assert row["status"] == "rejected"
    events = _fetch(cfg, "SELECT weight FROM events WHERE item_id=? AND kind='reject'", RADIOHEAD)
    assert len(events) == 1
    assert events[0]["weight"] == -1.0


def test_double_act_conflicts(web):
    client, _, movie_rec, _ = web
    assert client.post(f"/api/recs/{movie_rec}/approve").status_code == 200
    assert client.post(f"/api/recs/{movie_rec}/approve").status_code == 409
    assert client.post("/api/recs/99999/approve").status_code == 409


def test_why_text(web):
    client, _, movie_rec, _ = web
    resp = client.get(f"/api/recs/{movie_rec}/why")
    assert resp.status_code == 200
    text = resp.json()["text"]
    assert isinstance(text, str)
    assert "The Matrix" in text
    assert "tmdb_similar" in text
    assert "Blade Runner" in text


def test_stats(web):
    client, _, *_ = web
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
                   "i-play", "i-external"):
        assert f'id="{symbol}"' in resp.text
    for label in ("All", "Movies", "Series", "Music", "History"):
        assert label in resp.text
    assert "aria-selected" in resp.text
    # Trailer/preview affordances ship as static strings in the page script.
    assert "youtube.com/watch" in resp.text
    assert "itunes.apple.com/search" in resp.text


def test_foreign_host_rejected(web):
    client, cfg, movie_rec, _ = web
    assert client.get("/api/recs", headers={"Host": "evil.example.com"}).status_code == 403
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Host": "attacker.rebind.net"})
    assert resp.status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "proposed"


def test_cross_origin_post_rejected(web):
    client, cfg, movie_rec, _ = web
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Origin": "https://evil.example.com"})
    assert resp.status_code == 403
    (row,) = _fetch(cfg, "SELECT status FROM recommendations WHERE id=?", movie_rec)
    assert row["status"] == "proposed"
    events = _fetch(cfg, "SELECT 1 FROM events WHERE item_id=? AND kind='approve'", MATRIX)
    assert events == []


def test_no_origin_and_allowlisted_origin_posts_allowed(web):
    client, _, movie_rec, artist_rec = web
    # CLI clients and same-origin navigations send no Origin header.
    assert client.post(f"/api/recs/{artist_rec}/reject").status_code == 200
    # Same-origin fetch from the served UI carries an allowlisted Origin.
    resp = client.post(f"/api/recs/{movie_rec}/approve",
                       headers={"Origin": "http://localhost:8790"})
    assert resp.status_code == 200


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
