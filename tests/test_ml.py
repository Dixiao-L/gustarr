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


def test_rank_serendipity_wins_exploration_slots(tmp_path):
    """Serendipity-sourced candidates get first claim on exploration slots
    even though they score below the old 40-90 band; leftover slots fill
    from the band; the bottom-decile sanity floor still applies."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, exploration_frac=0.4, diversity_lambda=0.3)
    model = cfg.model.embed_model
    vecs = {
        "s1": axis(0),                                          # MMR pick
        "s2": unit(0.8 * axis(0) + 0.6 * axis(2)),              # MMR pick
        "s3": unit(0.8 * axis(0) + 0.6 * axis(3)),              # MMR pick
        "b1": unit(0.45 * axis(0) + 0.893 * axis(6)),           # in the 40-90 band
        "b2": unit(0.5 * axis(0) + 0.866 * axis(1)),            # above the band's p90
        "sr": unit(-0.4 * axis(0) + 0.917 * axis(5)),           # serendipity, sub-band score
        "sj": -axis(0),                                         # serendipity, bottom decile
    }
    for name, v in vecs.items():
        iid = f"movie:tmdb:{name}"
        db.upsert_item(conn, iid, "movie", title=name.upper())
        put_vec(conn, iid, v, model)
        source = "serendipity_tmdb" if name in ("sr", "sj") else "tmdb_similar"
        add_candidate(conn, iid, source=source)
    db.set_state(conn, "model:movie", json.dumps({
        "w": b64(4.0 * axis(0)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.set_state(conn, "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=5)
    assert stats["proposed"] == 5
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    # 3 MMR + 2 exploration: sr beats b1 (band member with the higher
    # score) for the first slot; the remainder fills from the band. sj is
    # serendipity too but scores in the bottom decile — sanity floor.
    assert set(whys) == {f"movie:tmdb:{n}" for n in ("s1", "s2", "s3", "sr", "b1")}
    sr = whys["movie:tmdb:sr"]
    assert sr["exploration"] is True and sr["serendipity"] is True
    assert sr["sources"] == ["serendipity_tmdb"]
    b1 = whys["movie:tmdb:b1"]
    assert b1["exploration"] is True and "serendipity" not in b1
    for name in ("s1", "s2", "s3"):
        w = whys[f"movie:tmdb:{name}"]
        assert w["exploration"] is False and "serendipity" not in w


def test_rank_exploration_novelty_gate_excludes_near_core(tmp_path):
    """An in-band, maximally-distant-from-picked item that still hugs the
    positive centroid loses the exploration slot to a genuinely novel one."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, exploration_frac=0.34)
    model = cfg.model.embed_model
    # Taste axis (head) is e1; core axis (centroid) is e0, so scores and
    # centroid similarity decouple. "core" is the most distant in-band
    # item — the old max-distance rule would pick it — but sits at
    # cos 0.9 to the centroid, far above the pool's 60th percentile.
    vecs = {
        "h1": axis(1) + 0.1 * axis(0),
        "h2": 0.8 * axis(1) + 0.6 * axis(2) + 0.15 * axis(0),
        "mid": 0.5 * axis(1) + 0.866 * axis(5) + 0.2 * axis(0),
        "novel": 0.45 * axis(1) + 0.893 * axis(4) - 0.1 * axis(0),
        "core": 0.9 * axis(0) + 0.3 * axis(1) + 0.316 * axis(3),
        "j1": -axis(1) - 0.05 * axis(0),
        "j2": -axis(1) + 0.05 * axis(0) + 0.1 * axis(6),
    }
    for name, v in vecs.items():
        iid = f"movie:tmdb:{name}"
        db.upsert_item(conn, iid, "movie", title=name.upper())
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid)
    db.set_state(conn, "model:movie", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.set_state(conn, "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=3)
    assert stats["proposed"] == 3
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    assert set(whys) == {"movie:tmdb:h1", "movie:tmdb:h2", "movie:tmdb:novel"}
    assert whys["movie:tmdb:novel"]["exploration"] is True
    assert "serendipity" not in whys["movie:tmdb:novel"]


def test_rank_exploration_fallback_without_serendipity(tmp_path):
    """No serendipity candidates and a novelty gate that filters every
    remaining item (all sit exactly on the pool's 60th centroid-sim
    percentile): the old band max-distance logic still fills the slot."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, exploration_frac=0.34)
    model = cfg.model.embed_model
    vecs = {  # all orthogonal to the e0 centroid → cent_sims all equal
        "f1": axis(1),
        "f2": unit(0.8 * axis(1) + 0.6 * axis(2)),
        "f3": unit(0.5 * axis(1) + 0.866 * axis(3)),
        "f4": axis(4),
        "f5": -axis(1),
    }
    for name, v in vecs.items():
        iid = f"movie:tmdb:{name}"
        db.upsert_item(conn, iid, "movie", title=name.upper())
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid)
    db.set_state(conn, "model:movie", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.set_state(conn, "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=3)
    assert stats["proposed"] == 3  # gate never leaves slots unfilled
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    # f4 is the pre-serendipity band pick: in the 40-90 band, max distance.
    assert set(whys) == {"movie:tmdb:f1", "movie:tmdb:f2", "movie:tmdb:f4"}
    assert whys["movie:tmdb:f4"]["exploration"] is True
    assert "serendipity" not in whys["movie:tmdb:f4"]


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


def test_rank_rescores_open_proposals_with_current_scorer(tmp_path):
    """Frozen scores from earlier runs/scorers are re-scored on the current
    model's scale so apply's cross-row comparisons compare like with like."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    ts = db.now()

    # Open proposal from an old centroid-era run with an inflated frozen score.
    db.upsert_item(conn, "movie:tmdb:old", "movie", title="Old")
    put_vec(conn, "movie:tmdb:old", unit(0.5 * axis(0) + float(np.sqrt(0.75)) * axis(1)), model)
    old_why = json.dumps({"sources": ["tmdb_similar"], "exploration": False})
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why, status)"
        " VALUES ('oldrun', ?, 'movie', 'movie:tmdb:old', 0.99, ?, 'proposed')", (ts, old_why))
    # Open proposal without an embedding: keeps its old score.
    db.upsert_item(conn, "movie:tmdb:noemb", "movie", title="NoEmb")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('oldrun', ?, 'movie', 'movie:tmdb:noemb', 0.42, 'proposed')", (ts,))
    # Approved rec: not touched by the re-score (only 'proposed' rows are).
    db.upsert_item(conn, "movie:tmdb:appr", "movie", title="Appr")
    put_vec(conn, "movie:tmdb:appr", axis(0), model)
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('oldrun', ?, 'movie', 'movie:tmdb:appr', 0.91, 'approved')", (ts,))

    db.upsert_item(conn, "movie:tmdb:new", "movie", title="New")
    put_vec(conn, "movie:tmdb:new", axis(0), model)
    add_candidate(conn, "movie:tmdb:new")
    db.set_state(conn, "model:movie", json.dumps({
        "w": b64(4.0 * axis(0)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": ts, "n_pos": 10, "n_neg": 3, "n_weak": 20}))

    stats = rank_mod.run(conn, cfg, top=5)
    assert stats["rescored"] == 1
    rows = {r["item_id"]: r for r in conn.execute("SELECT * FROM recommendations")}

    old = rows["movie:tmdb:old"]
    # sigmoid(4*0.5 - 1), the current head's base score — not the frozen 0.99.
    assert old["score"] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)), abs=1e-3)
    assert old["status"] == "proposed" and old["run_id"] == "oldrun"
    assert old["why"] == old_why  # why untouched
    assert rows["movie:tmdb:noemb"]["score"] == 0.42
    assert rows["movie:tmdb:appr"]["score"] == 0.91
    # New proposal scored by the same head (+ one-source bonus): comparable.
    new_score = rows["movie:tmdb:new"]["score"]
    assert new_score == pytest.approx(1.0 / (1.0 + np.exp(-3.0)) + 0.03, abs=1e-3)
    assert old["score"] < new_score  # frozen 0.99 would have outranked it


def test_rank_ignores_head_from_other_embed_model(tmp_path):
    """A head trained in a different embedding space (same dim) must be
    treated as absent so the current-model centroid takes over."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    for name, v in {"a1": axis(0), "a2": axis(1)}.items():
        iid = f"artist:mbid:{name}"
        db.upsert_item(conn, iid, "artist", title=name)
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, source="lastfm_similar")
    # Stale head points at axis(1): if it were used, a2 would win big.
    db.set_state(conn, "model:artist", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": "other/embedder",
        "trained_at": db.now(), "n_pos": 8, "n_neg": 0, "n_weak": 16}))
    db.set_state(conn, "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["stale_state"] == 1
    assert stats["proposed"] == 2
    scores = {r["item_id"]: r["score"] for r in conn.execute(
        "SELECT item_id, score FROM recommendations WHERE domain='artist'")}
    # Centroid (cos+1)/2 mapping + one-source bonus, not the stale head's sigmoid.
    assert abs(scores["artist:mbid:a1"] - 1.03) < 0.02
    assert abs(scores["artist:mbid:a2"] - 0.53) < 0.02


def test_rank_skips_domain_when_all_state_is_stale(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    db.upsert_item(conn, "artist:mbid:a1", "artist", title="a1")
    put_vec(conn, "artist:mbid:a1", axis(0), cfg.model.embed_model)
    add_candidate(conn, "artist:mbid:a1", source="lastfm_similar")
    db.set_state(conn, "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": "other/embedder",
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["stale_state"] == 1
    assert stats["proposed"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM recommendations").fetchone()["n"] == 0


def test_rank_respects_video_queue_cap(tmp_path):
    """Rank must stop proposing video items at the cap instead of letting
    apply mass-expire the overflow (churned 180 enriched items once)."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    cfg.autonomy.video_queue_max_pending = 1
    model = cfg.model.embed_model
    for i in range(4):
        iid = f"movie:tmdb:c{i}"
        db.upsert_item(conn, iid, "movie", title=f"C{i}")
        put_vec(conn, iid, axis(i), model)
        add_candidate(conn, iid)
    # one open movie proposal occupies the whole video budget
    db.upsert_item(conn, "movie:tmdb:open1", "movie", title="Open")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why, status)"
        " VALUES ('r0', '2026-07-10T00:00:00Z', 'movie', 'movie:tmdb:open1', 0.5, '{}',"
        " 'proposed')")
    rank_mod.run(conn, cfg, top=5)
    open_video = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE domain IN ('movie','series')"
        " AND status='proposed'").fetchone()[0]
    assert open_video == 1  # cap already spent; nothing new proposed
