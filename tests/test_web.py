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
    cfg = C._build({"core": {"data_dir": str(tmp_path)}})
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
    assert "gustarr" in resp.text
    assert "/api/recs" in resp.text
