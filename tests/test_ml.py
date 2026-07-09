"""Offline tests for the ml slice: embed doc building + selection,
head/centroid training on synthetic separable vectors, and ranking
end-to-end on a fabricated store. Never imports sentence-transformers."""

from __future__ import annotations

import base64
import json
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gustarr import config as C
from gustarr import db
from gustarr.ml import embed as embed_mod
from gustarr.ml import rank as rank_mod
from gustarr.ml import train as train_mod

DIM = 8


def make_cfg(tmp_path, **model_kw):
    raw = {"core": {"data_dir": str(tmp_path)}}
    if model_kw:
        raw["model"] = model_kw
    return C._build(raw)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def axis(i, dim=DIM):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def put_vec(conn, item_id, vec, model):
    v16 = np.asarray(vec, dtype=np.float16)
    db.put_embedding(conn, item_id, model, v16.tobytes(), len(v16))


def add_candidate(conn, item_id, source="tmdb_similar", seed=None, ext=None):
    ts = db.now()
    conn.execute(
        "INSERT OR REPLACE INTO candidates"
        " (item_id, source, seed_item_id, external_score, first_seen, last_seen)"
        " VALUES (?,?,?,?,?,?)",
        (item_id, source, seed, ext, ts, ts))


def b64(vec):
    return base64.b64encode(np.asarray(vec, dtype=np.float32).tobytes()).decode("ascii")


# ── embed ────────────────────────────────────────────────────────────


def test_build_doc_movie():
    doc = embed_mod.build_doc({
        "domain": "movie", "title": "The Matrix", "year": 1999,
        "meta": {"genres": ["Action", "Sci-Fi"], "keywords": ["cyberpunk", "dystopia"],
                 "tags": ["mind-bending"], "original_language": "en",
                 "overview": "x" * 2000},
    })
    lines = doc.split("\n")
    assert lines[0] == "movie: The Matrix (1999)"
    assert "genres: Action, Sci-Fi" in lines
    assert "keywords: cyberpunk, dystopia" in lines
    assert "tags: mind-bending" in lines
    assert "language: en" in lines
    overview = [ln for ln in lines if ln.startswith("overview: ")][0]
    assert len(overview) == len("overview: ") + 1200


def test_build_doc_artist_and_json_meta():
    doc = embed_mod.build_doc({
        "domain": "artist", "title": "Radiohead", "year": None,
        "meta": json.dumps({"tags": ["rock", "electronic"],
                            "similar": ["Muse", "Portishead"], "bio": "y" * 1000}),
    })
    lines = doc.split("\n")
    assert lines[0] == "artist: Radiohead"
    assert "tags: rock, electronic" in lines
    assert "similar: Muse, Portishead" in lines
    bio = [ln for ln in lines if ln.startswith("bio: ")][0]
    assert len(bio) == len("bio: ") + 800
    assert embed_mod.build_doc({"domain": "movie", "title": "", "meta": {}}) == ""


def test_embed_run_selects_and_stores(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    calls = {}

    class FakeST:
        def __init__(self, name, device=None, cache_folder=None):
            calls["model"] = name
            calls["device"] = device

        def encode(self, docs, **kw):
            calls["docs"] = list(docs)
            calls["kw"] = kw
            rng = np.random.default_rng(1)
            v = rng.normal(size=(len(docs), DIM)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    meta = {"genres": ["Drama"], "overview": "A film."}
    ts = db.now()
    # a: enriched + event → embedded
    db.upsert_item(conn, "movie:tmdb:a", "movie", title="A", year=2000, meta=meta, enriched=True)
    db.add_event(conn, ts, "movie:tmdb:a", "play", 0.3, "jellyfin")
    # b: enriched + candidate → embedded
    db.upsert_item(conn, "movie:tmdb:b", "movie", title="B", meta=meta, enriched=True)
    add_candidate(conn, "movie:tmdb:b")
    # c: enriched but not relevant anywhere → ignored
    db.upsert_item(conn, "movie:tmdb:c", "movie", title="C", meta=meta, enriched=True)
    # d: relevant but not enriched → ignored
    db.upsert_item(conn, "movie:tmdb:d", "movie", title="D", enriched=False)
    db.add_event(conn, ts, "movie:tmdb:d", "play", 0.3, "jellyfin")
    # e: relevant + enriched but already embedded → not re-embedded
    db.upsert_item(conn, "movie:tmdb:e", "movie", title="E", meta=meta, enriched=True)
    add_candidate(conn, "movie:tmdb:e")
    put_vec(conn, "movie:tmdb:e", axis(0), model)

    stats = embed_mod.run(conn, cfg)
    assert stats == {"embedded": 2, "skipped": 0}
    assert calls["model"] == model
    assert calls["kw"]["batch_size"] == 32
    assert calls["kw"]["normalize_embeddings"] is True
    assert any(d.startswith("movie: A (2000)") for d in calls["docs"])
    rows = {r["item_id"]: r for r in conn.execute(
        "SELECT item_id, dim, vec FROM embeddings WHERE model=?", (model,))}
    assert set(rows) == {"movie:tmdb:a", "movie:tmdb:b", "movie:tmdb:e"}
    assert rows["movie:tmdb:a"]["dim"] == DIM
    assert len(rows["movie:tmdb:a"]["vec"]) == DIM * 2  # float16 bytes


def test_embed_requires_ml_extra(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    db.upsert_item(conn, "movie:tmdb:a", "movie", title="A",
                   meta={"overview": "x"}, enriched=True)
    db.add_event(conn, db.now(), "movie:tmdb:a", "play", 0.3, "jellyfin")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        embed_mod.run(conn, cfg)


# ── train ────────────────────────────────────────────────────────────


def test_train_head_separates_and_persists(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    rng = np.random.default_rng(42)
    ts = db.now()

    for i in range(10):  # positives cluster on +e0
        iid = f"movie:tmdb:p{i}"
        db.upsert_item(conn, iid, "movie", title=f"P{i}")
        put_vec(conn, iid, unit(axis(0) + 0.1 * rng.normal(size=DIM)), model)
        db.add_event(conn, ts, iid, "loved", 1.0, "jellyfin")
    for i in range(3):  # explicit negatives cluster on -e0
        iid = f"movie:tmdb:n{i}"
        db.upsert_item(conn, iid, "movie", title=f"N{i}")
        put_vec(conn, iid, unit(-axis(0) + 0.1 * rng.normal(size=DIM)), model)
        db.add_event(conn, ts, iid, "reject", -1.0, "user")
    for i in range(25):  # event-less candidates → weak negative pool
        iid = f"movie:tmdb:w{i}"
        db.upsert_item(conn, iid, "movie", title=f"W{i}")
        put_vec(conn, iid, unit(rng.normal(size=DIM)), model)
        add_candidate(conn, iid)

    stats = train_mod.run(conn, cfg)
    assert stats["movie"] == {"pos": 10, "neg": 3, "weak": 20, "head": 1}

    head = json.loads(db.get_state(conn, "model:movie"))
    w = np.frombuffer(base64.b64decode(head["w"]), dtype=np.float32)
    assert w.shape == (DIM,)
    assert head["dim"] == DIM
    assert head["embed_model"] == model
    assert (head["n_pos"], head["n_neg"], head["n_weak"]) == (10, 3, 20)
    assert isinstance(head["b"], float) and "trained_at" in head

    def s(v):
        return float(1.0 / (1.0 + np.exp(-(v @ w + head["b"]))))

    assert s(axis(0)) > s(-axis(0)) + 0.2  # separates the taste axis
    assert s(axis(0)) > s(axis(1))

    cent = json.loads(db.get_state(conn, "centroid:movie"))
    pos_c = np.frombuffer(base64.b64decode(cent["pos"]), dtype=np.float32)
    assert float(unit(pos_c) @ axis(0)) > 0.9
    neg_c = np.frombuffer(base64.b64decode(cent["neg"]), dtype=np.float32)
    assert float(unit(neg_c) @ axis(0)) < -0.9
    assert len(cent["exemplars"]) == 10
    assert all(len(e) == 2 for e in cent["exemplars"])
    labels = [e[1] for e in cent["exemplars"]]
    assert labels == sorted(labels, reverse=True)
    assert db.get_state(conn, "model:artist") is None


def test_train_few_positives_centroid_only(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    ts = db.now()
    for i in range(3):  # below the 8-positive head guard
        iid = f"series:tvdb:{i}"
        db.upsert_item(conn, iid, "series", title=f"S{i}")
        put_vec(conn, iid, unit(axis(1) + 0.05 * axis(i + 2)), model)
        db.add_event(conn, ts, iid, "library_add", 0.6, "arr")

    stats = train_mod.run(conn, cfg)
    assert stats["series"] == {"pos": 3, "neg": 0, "weak": 0, "head": 0}
    assert db.get_state(conn, "model:series") is None
    cent = json.loads(db.get_state(conn, "centroid:series"))
    assert cent["neg"] is None
    assert len(cent["exemplars"]) == 3


# ── rank ─────────────────────────────────────────────────────────────


def test_rank_end_to_end(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, exploration_frac=0.2, diversity_lambda=0.5)
    model = cfg.model.embed_model
    e = [axis(i) for i in range(DIM)]
    r = float(np.sqrt(1 - 0.49))

    vecs = {
        "m1": e[0],
        "m2": unit(e[0] + 0.05 * e[1]),  # near-duplicate of m1, near-top score
        "m3": unit(0.8 * e[0] + 0.6 * e[2]),
        "m4": unit(0.7 * e[0] + r * e[3]),
        "m5": unit(0.7 * e[0] + r * e[4]),
        "m6": unit(0.6 * e[0] + 0.8 * e[5]),
        "m7": e[1],
        "m8": e[2],
        "m9": unit(-e[0] + 0.2 * e[5]),
        "m10": e[6],
    }
    db.upsert_item(conn, "movie:tmdb:seed1", "movie", title="Seed Movie")
    for name, v in vecs.items():
        iid = f"movie:tmdb:{name}"
        db.upsert_item(conn, iid, "movie", title=name.upper())
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, seed="movie:tmdb:seed1")

    # exclusions: in library / rejected / already openly recommended —
    # all e0-aligned so they would top the ranking if the filter leaked
    for name in ("lib1", "rej1", "open1"):
        iid = f"movie:tmdb:{name}"
        db.upsert_item(conn, iid, "movie", title=name)
        put_vec(conn, iid, e[0], model)
        add_candidate(conn, iid)
    conn.execute("INSERT INTO library (item_id, arr) VALUES ('movie:tmdb:lib1','radarr')")
    db.add_event(conn, db.now(), "movie:tmdb:rej1", "reject", -1.0, "user")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('old', ?, 'movie', 'movie:tmdb:open1', 0.9, 'proposed')", (db.now(),))

    # stale proposal → should expire (default TTL 30 days)
    db.upsert_item(conn, "movie:tmdb:stale", "movie", title="stale")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('old2', ?, 'movie', 'movie:tmdb:stale', 0.5, 'proposed')", (old_ts,))

    db.set_state(conn, "model:movie", json.dumps({
        "w": b64(4.0 * e[0]), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    exemplars = []
    for i, v in enumerate([e[0], unit(0.9 * e[0] + 0.436 * e[1]),
                           unit(0.8 * e[0] + 0.6 * e[3])], 1):
        iid = f"movie:tmdb:ex{i}"
        db.upsert_item(conn, iid, "movie", title=f"Liked {i}")
        put_vec(conn, iid, v, model)
        exemplars.append([iid, round(1.0 - 0.1 * i, 2)])
    db.set_state(conn, "centroid:movie", json.dumps({
        "pos": b64(e[0]), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": exemplars}))

    stats = rank_mod.run(conn, cfg, top=5)

    rows = conn.execute(
        "SELECT * FROM recommendations WHERE domain='movie' AND status='proposed'"
        " AND run_id NOT IN ('old','old2')").fetchall()
    assert len(rows) == 5
    assert stats["proposed"] == 5
    assert stats["movie"] == 5
    picked = {r["item_id"] for r in rows}
    assert "movie:tmdb:m1" in picked
    assert "movie:tmdb:m2" not in picked  # MMR suppresses the near-duplicate
    assert not picked & {"movie:tmdb:lib1", "movie:tmdb:rej1", "movie:tmdb:open1"}

    pv = [vecs[i.split(":")[-1]] for i in picked]
    for a in range(len(pv)):
        for b_ in range(a + 1, len(pv)):
            assert float(pv[a] @ pv[b_]) < 0.9  # no near-duplicate pair got through

    flags = []
    for row in rows:
        why = json.loads(row["why"])
        assert why["sources"] == ["tmdb_similar"]
        assert why["seeds"] == ["Seed Movie"]
        assert 1 <= len(why["neighbors"]) <= 3
        for n in why["neighbors"]:
            assert set(n) == {"item_id", "title", "sim"}
            assert n["title"].startswith("Liked ")
        sims = [n["sim"] for n in why["neighbors"]]
        assert sims == sorted(sims, reverse=True)
        flags.append(why["exploration"])
        assert 0.0 <= row["score"] <= 1.2
    assert sum(flags) == 1  # round(5 * 0.2) exploration slot

    assert stats["expired"] == 1
    assert conn.execute("SELECT status FROM recommendations WHERE item_id='movie:tmdb:stale'")
    stale = conn.execute(
        "SELECT status, acted_at FROM recommendations WHERE item_id='movie:tmdb:stale'"
    ).fetchone()
    assert stale["status"] == "expired" and stale["acted_at"]
    still_open = conn.execute(
        "SELECT status FROM recommendations WHERE item_id='movie:tmdb:open1'").fetchone()
    assert still_open["status"] == "proposed"


def test_rank_centroid_fallback_artist(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    for name, v in {"a1": axis(0), "a2": axis(1), "a3": -axis(0)}.items():
        iid = f"artist:mbid:{name}"
        db.upsert_item(conn, iid, "artist", title=name)
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, source="lastfm_similar")
    db.set_state(conn, "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["proposed"] == 3  # pool smaller than the 10 artist slots
    scores = {r["item_id"]: r["score"] for r in conn.execute(
        "SELECT item_id, score FROM recommendations WHERE domain='artist'")}
    s1, s2, s3 = (scores[f"artist:mbid:a{i}"] for i in (1, 2, 3))
    assert s1 > s2 > s3
    assert abs(s1 - 1.03) < 0.02  # (cos+1)/2 mapping + one-source bonus
    assert abs(s2 - 0.53) < 0.02
    why = json.loads(conn.execute(
        "SELECT why FROM recommendations WHERE item_id='artist:mbid:a1'").fetchone()["why"])
    assert why["neighbors"] == [] and why["sources"] == ["lastfm_similar"]
