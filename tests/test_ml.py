"""Offline tests for the ml slice: embed doc building + selection,
head/centroid training on synthetic separable vectors, and ranking
end-to-end on a fabricated store. Never imports sentence-transformers.

Identity v3: fixtures mint items through db.resolve_item and carry the
returned integer ids everywhere — embeddings, candidates, exemplars and
why-json all speak ints now.
"""

from __future__ import annotations

import base64
import json
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gustarr import config as C
from gustarr import db, settings, signals
from gustarr.ml import embed as embed_mod
from gustarr.ml import rank as rank_mod
from gustarr.ml import train as train_mod

DIM = 8


def make_cfg(tmp_path, profiles=None, **model_kw):
    # no profiles section → config synthesizes the single 'default' profile
    raw = {"core": {"data_dir": str(tmp_path)}}
    if model_kw:
        raw["model"] = model_kw
    if profiles:
        raw["profiles"] = {name: {} for name in profiles}
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


def movie(conn, key, title=None):
    return db.resolve_item(conn, "movie", "tmdb", key, title=title or key.upper())


def artist(conn, key, title=None):
    return db.resolve_item(conn, "artist", "mbid", key, title=title or key)


def album(conn, key, title=None):
    return db.resolve_item(conn, "album", "mbid", key, title=title or key.upper())


def add_candidate(conn, item_id, source="tmdb_similar", seed=None, ext=None,
                  profile="default"):
    ts = db.now()
    conn.execute(
        "INSERT OR REPLACE INTO candidates"
        " (profile, item_id, source, seed_item_id, external_score, first_seen, last_seen)"
        " VALUES (?,?,?,?,?,?,?)",
        (profile, item_id, source, seed, ext, ts, ts))


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

    def enriched_movie(key, title, year=None):
        iid = db.resolve_item(conn, "movie", "tmdb", key, title=title, year=year, meta=meta)
        db.upsert_item_fields(conn, iid, enriched=True)
        return iid

    # a: enriched + event → embedded
    a = enriched_movie("a", "A", year=2000)
    db.add_event(conn, ts, a, "play", 1.0, "jellyfin")
    # b: enriched + candidate → embedded
    b = enriched_movie("b", "B")
    add_candidate(conn, b)
    # c: enriched but not relevant anywhere → ignored
    enriched_movie("c", "C")
    # d: relevant but not enriched → ignored
    d = db.resolve_item(conn, "movie", "tmdb", "d", title="D")
    db.add_event(conn, ts, d, "play", 1.0, "jellyfin")
    # e: relevant + enriched but already embedded → not re-embedded
    e = enriched_movie("e", "E")
    add_candidate(conn, e)
    put_vec(conn, e, axis(0), model)

    stats = embed_mod.run(conn, cfg)
    assert stats == {"embedded": 2, "skipped": 0}
    assert calls["model"] == model
    assert calls["kw"]["batch_size"] == 32
    assert calls["kw"]["normalize_embeddings"] is True
    assert any(d_.startswith("movie: A (2000)") for d_ in calls["docs"])
    rows = {r["item_id"]: r for r in conn.execute(
        "SELECT item_id, dim, vec FROM embeddings WHERE model=?", (model,))}
    assert set(rows) == {a, b, e}
    assert rows[a]["dim"] == DIM
    assert len(rows[a]["vec"]) == DIM * 2  # float16 bytes


def test_embed_requires_ml_extra(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    iid = db.resolve_item(conn, "movie", "tmdb", "a", title="A", meta={"overview": "x"})
    db.upsert_item_fields(conn, iid, enriched=True)
    db.add_event(conn, db.now(), iid, "play", 1.0, "jellyfin")
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

    pos_ids = []
    for i in range(10):  # positives cluster on +e0
        iid = movie(conn, f"p{i}", f"P{i}")
        pos_ids.append(iid)
        put_vec(conn, iid, unit(axis(0) + 0.1 * rng.normal(size=DIM)), model)
        db.add_event(conn, ts, iid, "loved", 1.0, "jellyfin")
    for i in range(3):  # explicit negatives cluster on -e0
        iid = movie(conn, f"n{i}", f"N{i}")
        put_vec(conn, iid, unit(-axis(0) + 0.1 * rng.normal(size=DIM)), model)
        db.add_event(conn, ts, iid, "reject", 1.0, "user")
    for i in range(25):  # event-less candidates → weak negative pool
        iid = movie(conn, f"w{i}", f"W{i}")
        put_vec(conn, iid, unit(rng.normal(size=DIM)), model)
        add_candidate(conn, iid)

    stats = train_mod.run(conn, cfg)
    assert stats["default"]["movie"] == {"pos": 10, "neg": 3, "weak": 20, "head": 1}

    head = json.loads(db.pget_state(conn, "default", "model:movie"))
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

    cent = json.loads(db.pget_state(conn, "default", "centroid:movie"))
    pos_c = np.frombuffer(base64.b64decode(cent["pos"]), dtype=np.float32)
    assert float(unit(pos_c) @ axis(0)) > 0.9
    neg_c = np.frombuffer(base64.b64decode(cent["neg"]), dtype=np.float32)
    assert float(unit(neg_c) @ axis(0)) < -0.9
    assert len(cent["exemplars"]) == 10
    assert all(len(e) == 2 for e in cent["exemplars"])
    # exemplars persist plain int item ids — json numbers round-trip
    assert {e[0] for e in cent["exemplars"]} == set(pos_ids)
    assert all(isinstance(e[0], int) for e in cent["exemplars"])
    labels = [e[1] for e in cent["exemplars"]]
    assert labels == sorted(labels, reverse=True)
    assert db.pget_state(conn, "default", "model:artist") is None
    # nothing may leak into the old un-namespaced key
    assert db.get_state(conn, "model:movie") is None


def test_retuned_weights_reprice_old_events(tmp_path, monkeypatch):
    """Debt item 2's whole point: events store a scale multiplier, so
    retuning signals.WEIGHTS moves the labels of history already on disk
    at the next train — no re-collection, nothing frozen at write time."""
    conn = db.connect(tmp_path / "t.db")
    iid = movie(conn, "old", "Old")
    db.add_event(conn, db.now(), iid, "play", 1.0, "jellyfin")
    before = train_mod._item_labels(conn, "default", "movie")[iid]
    assert before == pytest.approx(signals.WEIGHTS["play"], rel=1e-3)

    monkeypatch.setitem(signals.WEIGHTS, "play", 3 * signals.WEIGHTS["play"])
    after = train_mod._item_labels(conn, "default", "movie")[iid]
    # same stored event, same recency — only the table's price changed
    assert after == pytest.approx(3 * before)


def test_train_few_positives_centroid_only(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    ts = db.now()
    for i in range(3):  # below the 8-positive head guard
        iid = db.resolve_item(conn, "series", "tvdb", str(i), title=f"S{i}")
        put_vec(conn, iid, unit(axis(1) + 0.05 * axis(i + 2)), model)
        db.add_event(conn, ts, iid, "library_add", 1.0, "arr")

    stats = train_mod.run(conn, cfg)
    assert stats["default"]["series"] == {"pos": 3, "neg": 0, "weak": 0, "head": 0}
    assert db.pget_state(conn, "default", "model:series") is None
    cent = json.loads(db.pget_state(conn, "default", "centroid:series"))
    assert cent["neg"] is None
    assert len(cent["exemplars"]) == 3


def test_train_two_profiles_independent_centroids(tmp_path):
    """Shared items/embeddings, personal labels: each profile's centroid
    must be built from its own events only."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"])
    model = cfg.model.embed_model
    ts = db.now()
    ids = {"alice": set(), "bob": set()}
    for profile, ax in (("alice", 0), ("bob", 1)):
        for i in range(3):
            iid = movie(conn, f"{profile}{i}", f"{profile}-{i}")
            ids[profile].add(iid)
            put_vec(conn, iid, unit(axis(ax) + 0.05 * axis(i + 2)), model)
            db.add_event(conn, ts, iid, "loved", 1.0, "jellyfin", profile=profile)

    stats = train_mod.run(conn, cfg)

    assert stats["alice"]["movie"]["pos"] == 3
    assert stats["bob"]["movie"]["pos"] == 3
    alice = json.loads(db.pget_state(conn, "alice", "centroid:movie"))
    bob = json.loads(db.pget_state(conn, "bob", "centroid:movie"))
    a_pos = np.frombuffer(base64.b64decode(alice["pos"]), dtype=np.float32)
    b_pos = np.frombuffer(base64.b64decode(bob["pos"]), dtype=np.float32)
    assert float(unit(a_pos) @ axis(0)) > 0.9  # alice's taste axis, untainted by bob's
    assert float(unit(b_pos) @ axis(1)) > 0.9
    # exemplars are the training evidence — one profile's must never
    # surface in the other's "because you liked ..." explanations
    assert {e[0] for e in alice["exemplars"]} == ids["alice"]
    assert {e[0] for e in bob["exemplars"]} == ids["bob"]
    assert db.get_state(conn, "centroid:movie") is None  # no global leftovers


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
    seed1 = movie(conn, "seed1", "Seed Movie")
    iids: dict[str, int] = {}
    for name, v in vecs.items():
        iid = movie(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, seed=seed1)

    # exclusions: in library / rejected / already openly recommended —
    # all e0-aligned so they would top the ranking if the filter leaked
    blocked = {}
    for name in ("lib1", "rej1", "open1"):
        iid = movie(conn, name)
        blocked[name] = iid
        put_vec(conn, iid, e[0], model)
        add_candidate(conn, iid)
    conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'radarr')",
                 (blocked["lib1"],))
    db.add_event(conn, db.now(), blocked["rej1"], "reject", 1.0, "user")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('old', ?, 'movie', ?, 0.9, 'proposed')", (db.now(), blocked["open1"]))

    # stale proposal → should expire (default TTL 30 days)
    stale = movie(conn, "stale")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('old2', ?, 'movie', ?, 0.5, 'proposed')", (old_ts, stale))

    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * e[0]), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    exemplars = []
    for i, v in enumerate([e[0], unit(0.9 * e[0] + 0.436 * e[1]),
                           unit(0.8 * e[0] + 0.6 * e[3])], 1):
        iid = movie(conn, f"ex{i}", f"Liked {i}")
        put_vec(conn, iid, v, model)
        exemplars.append([iid, round(1.0 - 0.1 * i, 2)])
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
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
    assert iids["m1"] in picked
    assert iids["m2"] not in picked  # MMR suppresses the near-duplicate
    assert not picked & set(blocked.values())

    names_by_id = {v: k for k, v in iids.items()}
    pv = [vecs[names_by_id[i]] for i in picked]
    for a in range(len(pv)):
        for b_ in range(a + 1, len(pv)):
            assert float(pv[a] @ pv[b_]) < 0.9  # no near-duplicate pair got through

    ex_ids = {ex[0] for ex in exemplars}
    flags = []
    for row in rows:
        why = json.loads(row["why"])
        assert why["sources"] == ["tmdb_similar"]
        # seeds/neighbors carry int item ids; display names resolve at read time
        assert why["seeds"] == [seed1]
        assert 1 <= len(why["neighbors"]) <= 3
        for n in why["neighbors"]:
            assert set(n) == {"item_id", "title", "sim"}
            assert n["item_id"] in ex_ids
            assert n["title"].startswith("Liked ")
        sims = [n["sim"] for n in why["neighbors"]]
        assert sims == sorted(sims, reverse=True)
        flags.append(why["exploration"])
        assert 0.0 <= row["score"] <= 1.2
    assert sum(flags) == 1  # round(5 * 0.2) exploration slot

    assert stats["expired"] == 1
    stale_row = conn.execute(
        "SELECT status, acted_at FROM recommendations WHERE item_id=?", (stale,)).fetchone()
    assert stale_row["status"] == "expired" and stale_row["acted_at"]
    still_open = conn.execute(
        "SELECT status FROM recommendations WHERE item_id=?", (blocked["open1"],)).fetchone()
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
    iids = {}
    for name, v in vecs.items():
        iid = movie(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        source = "serendipity_tmdb" if name in ("sr", "sj") else "tmdb_similar"
        add_candidate(conn, iid, source=source)
    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * axis(0)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=5)
    assert stats["proposed"] == 5
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    # 3 MMR + 2 exploration: sr beats b1 (band member with the higher
    # score) for the first slot; the remainder fills from the band. sj is
    # serendipity too but scores in the bottom decile — sanity floor.
    assert set(whys) == {iids[n] for n in ("s1", "s2", "s3", "sr", "b1")}
    sr = whys[iids["sr"]]
    assert sr["exploration"] is True and sr["serendipity"] is True
    assert sr["sources"] == ["serendipity_tmdb"]
    b1 = whys[iids["b1"]]
    assert b1["exploration"] is True and "serendipity" not in b1
    for name in ("s1", "s2", "s3"):
        w = whys[iids[name]]
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
    iids = {}
    for name, v in vecs.items():
        iid = movie(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid)
    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=3)
    assert stats["proposed"] == 3
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    assert set(whys) == {iids["h1"], iids["h2"], iids["novel"]}
    assert whys[iids["novel"]]["exploration"] is True
    assert "serendipity" not in whys[iids["novel"]]


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
    iids = {}
    for name, v in vecs.items():
        iid = movie(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid)
    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=3)
    assert stats["proposed"] == 3  # gate never leaves slots unfilled
    whys = {r["item_id"]: json.loads(r["why"]) for r in conn.execute(
        "SELECT item_id, why FROM recommendations WHERE status='proposed'")}
    # f4 is the pre-serendipity band pick: in the 40-90 band, max distance.
    assert set(whys) == {iids["f1"], iids["f2"], iids["f4"]}
    assert whys[iids["f4"]]["exploration"] is True
    assert "serendipity" not in whys[iids["f4"]]


def test_rank_album_domain_slots_and_video_cap_exemption(tmp_path):
    """Albums rank with ALBUM_TOP=10 slots (not the run-wide top) and,
    like artists, never count against or suffer the video queue cap."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    cfg.autonomy.video_queue_max_pending = 1
    model = cfg.model.embed_model
    rng = np.random.default_rng(7)

    seed_a = artist(conn, "seed-a", "Seed Artist")
    # 15 candidates: enough leftover after the 8 MMR picks that the 40-90
    # exploration band can fill both exploration slots
    for i in range(15):
        iid = album(conn, f"al{i}")
        put_vec(conn, iid, unit(axis(0) + 0.3 * rng.normal(size=DIM)), model)
        add_candidate(conn, iid, source="lastfm_top_albums", seed=seed_a,
                      ext=1.0 - 0.05 * i)
    db.pset_state(conn, "default", "centroid:album", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))
    # movies in the same run: the video cap of 1 must bite them, not albums
    for i in range(3):
        iid = movie(conn, f"m{i}")
        put_vec(conn, iid, axis(i + 1), model)
        add_candidate(conn, iid)
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(1)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg, top=20)

    assert stats["album"] == 10  # ALBUM_TOP, not top=20 or the video cap
    assert stats["movie"] == 1  # the cap only ever constrains video domains
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE domain='album' AND status='proposed'").fetchall()
    assert len(rows) == 10
    for r in rows:
        why = json.loads(r["why"])
        assert why["sources"] == ["lastfm_top_albums"]
        assert why["seeds"] == [seed_a]
        assert isinstance(why["exploration"], bool)
        assert 0.0 <= r["score"] <= 1.2


def test_rank_centroid_fallback_artist(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    iids = {}
    for name, v in {"a1": axis(0), "a2": axis(1), "a3": -axis(0)}.items():
        iid = artist(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, source="lastfm_similar")
    db.pset_state(conn, "default", "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["proposed"] == 3  # pool smaller than the 10 artist slots
    scores = {r["item_id"]: r["score"] for r in conn.execute(
        "SELECT item_id, score FROM recommendations WHERE domain='artist'")}
    s1, s2, s3 = (scores[iids[f"a{i}"]] for i in (1, 2, 3))
    assert s1 > s2 > s3
    assert abs(s1 - 1.03) < 0.02  # (cos+1)/2 mapping + one-source bonus
    assert abs(s2 - 0.53) < 0.02
    why = json.loads(conn.execute(
        "SELECT why FROM recommendations WHERE item_id=?", (iids["a1"],)).fetchone()["why"])
    assert why["neighbors"] == [] and why["sources"] == ["lastfm_similar"]


def test_rank_rescores_open_proposals_with_current_scorer(tmp_path):
    """Frozen scores from earlier runs/scorers are re-scored on the current
    model's scale so apply's cross-row comparisons compare like with like."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    ts = db.now()

    # Open proposal from an old centroid-era run with an inflated frozen score.
    old = movie(conn, "old", "Old")
    put_vec(conn, old, unit(0.5 * axis(0) + float(np.sqrt(0.75)) * axis(1)), model)
    old_why = json.dumps({"sources": ["tmdb_similar"], "exploration": False})
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why, status)"
        " VALUES ('oldrun', ?, 'movie', ?, 0.99, ?, 'proposed')", (ts, old, old_why))
    # Open proposal without an embedding: keeps its old score.
    noemb = movie(conn, "noemb", "NoEmb")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('oldrun', ?, 'movie', ?, 0.42, 'proposed')", (ts, noemb))
    # Approved rec: not touched by the re-score (only 'proposed' rows are).
    appr = movie(conn, "appr", "Appr")
    put_vec(conn, appr, axis(0), model)
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status)"
        " VALUES ('oldrun', ?, 'movie', ?, 0.91, 'approved')", (ts, appr))

    new = movie(conn, "new", "New")
    put_vec(conn, new, axis(0), model)
    add_candidate(conn, new)
    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * axis(0)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": ts, "n_pos": 10, "n_neg": 3, "n_weak": 20}))

    stats = rank_mod.run(conn, cfg, top=5)
    assert stats["rescored"] == 1
    rows = {r["item_id"]: r for r in conn.execute("SELECT * FROM recommendations")}

    old_row = rows[old]
    # sigmoid(4*0.5 - 1), the current head's base score — not the frozen 0.99.
    assert old_row["score"] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)), abs=1e-3)
    assert old_row["status"] == "proposed" and old_row["run_id"] == "oldrun"
    assert old_row["why"] == old_why  # why untouched
    assert rows[noemb]["score"] == 0.42
    assert rows[appr]["score"] == 0.91
    # New proposal scored by the same head (+ one-source bonus): comparable.
    new_score = rows[new]["score"]
    assert new_score == pytest.approx(1.0 / (1.0 + np.exp(-3.0)) + 0.03, abs=1e-3)
    assert old_row["score"] < new_score  # frozen 0.99 would have outranked it


def test_rank_ignores_head_from_other_embed_model(tmp_path):
    """A head trained in a different embedding space (same dim) must be
    treated as absent so the current-model centroid takes over."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    iids = {}
    for name, v in {"a1": axis(0), "a2": axis(1)}.items():
        iid = artist(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, source="lastfm_similar")
    # Stale head points at axis(1): if it were used, a2 would win big.
    db.pset_state(conn, "default", "model:artist", json.dumps({
        "w": b64(4.0 * axis(1)), "b": -1.0, "dim": DIM, "embed_model": "other/embedder",
        "trained_at": db.now(), "n_pos": 8, "n_neg": 0, "n_weak": 16}))
    db.pset_state(conn, "default", "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["stale_state"] == 1
    assert stats["proposed"] == 2
    scores = {r["item_id"]: r["score"] for r in conn.execute(
        "SELECT item_id, score FROM recommendations WHERE domain='artist'")}
    # Centroid (cos+1)/2 mapping + one-source bonus, not the stale head's sigmoid.
    assert abs(scores[iids["a1"]] - 1.03) < 0.02
    assert abs(scores[iids["a2"]] - 0.53) < 0.02


def test_rank_skips_domain_when_all_state_is_stale(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    a1 = artist(conn, "a1")
    put_vec(conn, a1, axis(0), cfg.model.embed_model)
    add_candidate(conn, a1, source="lastfm_similar")
    db.pset_state(conn, "default", "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": "other/embedder",
        "exemplars": []}))

    stats = rank_mod.run(conn, cfg)
    assert stats["stale_state"] == 1
    assert stats["proposed"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM recommendations").fetchone()["n"] == 0


def _artist_centroid_store(tmp_path, names):
    """Store with one centroid-scored artist candidate per (name, vec)."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    iids = {}
    for name, v in names.items():
        iid = artist(conn, name)
        iids[name] = iid
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid, source="lastfm_similar")
    db.pset_state(conn, "default", "centroid:artist", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))
    return conn, cfg, iids


def iso_days_ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_rank_active_snooze_blocks_reproposal(tmp_path):
    conn, cfg, iids = _artist_centroid_store(tmp_path, {"a1": axis(0), "a2": axis(1)})
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status, acted_at)"
        " VALUES ('r0', ?, 'artist', ?, 0.9, 'snoozed', ?)",
        (iso_days_ago(5), iids["a1"], iso_days_ago(5)))

    stats = rank_mod.run(conn, cfg)

    assert stats["unsnoozed"] == 0
    assert stats["proposed"] == 1
    proposed = {r["item_id"] for r in conn.execute(
        "SELECT item_id FROM recommendations WHERE status='proposed'")}
    assert proposed == {iids["a2"]}  # a1 would top the score if it leaked
    snoozed = conn.execute(
        "SELECT status FROM recommendations WHERE item_id=?", (iids["a1"],)).fetchone()
    assert snoozed["status"] == "snoozed"


def test_rank_never_reproposes_failed_items(tmp_path):
    """failed is terminal for automation — the architecture doc promises
    "rank never re-proposes it" (re-proposing would just re-fail in
    apply); only a human retry revives the rec."""
    conn, cfg, iids = _artist_centroid_store(tmp_path, {"a1": axis(0), "a2": axis(1)})
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status, acted_at)"
        " VALUES ('r0', ?, 'artist', ?, 0.9, 'failed', ?)",
        (iso_days_ago(5), iids["a1"], iso_days_ago(5)))

    stats = rank_mod.run(conn, cfg)

    assert stats["proposed"] == 1
    proposed = {r["item_id"] for r in conn.execute(
        "SELECT item_id FROM recommendations WHERE status='proposed'")}
    assert proposed == {iids["a2"]}  # a1 would top the score if it leaked
    failed = conn.execute(
        "SELECT status FROM recommendations WHERE item_id=?", (iids["a1"],)).fetchone()
    assert failed["status"] == "failed"


def test_rank_lapsed_snooze_expires_and_reproposes(tmp_path):
    conn, cfg, iids = _artist_centroid_store(tmp_path, {"a1": axis(0)})
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, status, acted_at)"
        " VALUES ('r0', ?, 'artist', ?, 0.9, 'snoozed', ?)",
        (iso_days_ago(45), iids["a1"], iso_days_ago(31)))

    stats = rank_mod.run(conn, cfg)

    assert stats["unsnoozed"] == 1
    assert stats["proposed"] == 1
    rows = conn.execute(
        "SELECT status, acted_at FROM recommendations WHERE item_id=?"
        " ORDER BY id", (iids["a1"],)).fetchall()
    assert [r["status"] for r in rows] == ["expired", "proposed"]
    assert rows[0]["acted_at"] > iso_days_ago(1)  # expiry stamped now, not snooze time


def test_rank_exploration_frac_override_changes_slot_split(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, exploration_frac=0.0)
    model = cfg.model.embed_model
    r = float(np.sqrt(1 - 0.49))
    vecs = {
        "m1": axis(0),
        "m2": unit(0.8 * axis(0) + 0.6 * axis(2)),
        "m3": unit(0.7 * axis(0) + r * axis(3)),
        "m4": unit(0.6 * axis(0) + 0.8 * axis(4)),
        "m5": unit(0.5 * axis(0) + 0.866 * axis(5)),
        "m6": axis(1),
        "m7": -axis(0),
        "m8": unit(0.45 * axis(0) + 0.893 * axis(6)),
        "m9": unit(0.4 * axis(0) + 0.917 * axis(7)),
    }
    for name, v in vecs.items():
        iid = movie(conn, name)
        put_vec(conn, iid, v, model)
        add_candidate(conn, iid)
    db.pset_state(conn, "default", "model:movie", json.dumps({
        "w": b64(4.0 * axis(0)), "b": -1.0, "dim": DIM, "embed_model": model,
        "trained_at": db.now(), "n_pos": 10, "n_neg": 3, "n_weak": 20}))
    settings.set(conn, "exploration_frac", 0.4)

    stats = rank_mod.run(conn, cfg, top=5)

    assert stats["proposed"] == 5
    flags = [json.loads(r["why"])["exploration"] for r in conn.execute(
        "SELECT why FROM recommendations WHERE status='proposed'")]
    # cfg alone would explore 0 slots; the override reserves round(5*0.4)
    assert sum(flags) == 2


def test_rank_respects_video_queue_cap(tmp_path):
    """Rank must stop proposing video items at the cap instead of letting
    apply mass-expire the overflow (churned 180 enriched items once)."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    cfg.autonomy.video_queue_max_pending = 1
    model = cfg.model.embed_model
    for i in range(4):
        iid = movie(conn, f"c{i}", f"C{i}")
        put_vec(conn, iid, axis(i), model)
        add_candidate(conn, iid)
    # one open movie proposal occupies the whole video budget
    open1 = movie(conn, "open1", "Open")
    conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why, status)"
        " VALUES ('r0', '2026-07-10T00:00:00Z', 'movie', ?, 0.5, '{}', 'proposed')",
        (open1,))
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))
    rank_mod.run(conn, cfg, top=5)
    open_video = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE domain IN ('movie','series')"
        " AND status='proposed'").fetchone()[0]
    assert open_video == 1  # cap already spent; nothing new proposed


def test_rank_video_cap_override_widens_budget(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    cfg.autonomy.video_queue_max_pending = 1
    model = cfg.model.embed_model
    for i in range(4):
        iid = movie(conn, f"c{i}", f"C{i}")
        put_vec(conn, iid, axis(i), model)
        add_candidate(conn, iid)
    db.pset_state(conn, "default", "centroid:movie", json.dumps({
        "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
        "exemplars": []}))
    settings.set(conn, "video_queue_max_pending", 3)
    stats = rank_mod.run(conn, cfg, top=5)
    assert stats["proposed"] == 3  # override 3 beats the cfg cap of 1


def test_album_domain_falls_back_to_artist_model(tmp_path):
    """Albums have no listen events of their own yet; they must score
    against the artist taste state instead of being skipped."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path)
    model = cfg.model.embed_model
    pos = axis(0)
    db.pset_state(conn, "default", "centroid:artist", json.dumps({
        "dim": DIM, "embed_model": model, "pos": b64(pos), "neg": None,
        "exemplars": []}))
    for i in range(3):
        iid = album(conn, f"al{i}", f"Album {i}")
        put_vec(conn, iid, unit(pos + 0.1 * axis(i + 1)), model)
        add_candidate(conn, iid, source="lastfm_top_albums")
    stats = rank_mod.run(conn, cfg, top=5)
    assert stats.get("album", 0) == 3


# ── rank: profiles ───────────────────────────────────────────────────


def _two_profile_artist_store(tmp_path):
    """One shared candidate item, per-profile pool rows + centroids."""
    conn = db.connect(tmp_path / "t.db")
    cfg = make_cfg(tmp_path, profiles=["alice", "bob"])
    model = cfg.model.embed_model
    x = artist(conn, "x", "Shared Pick")
    put_vec(conn, x, axis(0), model)
    for profile in ("alice", "bob"):
        add_candidate(conn, x, source="lastfm_similar", profile=profile)
        db.pset_state(conn, profile, "centroid:artist", json.dumps({
            "pos": b64(axis(0)), "neg": None, "dim": DIM, "embed_model": model,
            "exemplars": []}))
    return conn, cfg, x


def test_rank_proposes_same_item_to_both_profiles(tmp_path):
    """The open-rec unique index is (profile, item_id): one household,
    two queues, so both profiles may hold their own rec for one item."""
    conn, cfg, x = _two_profile_artist_store(tmp_path)

    stats = rank_mod.run(conn, cfg)

    assert stats["proposed"] == 2
    rows = conn.execute(
        "SELECT profile FROM recommendations WHERE item_id=? AND status='proposed'",
        (x,)).fetchall()
    assert sorted(r["profile"] for r in rows) == ["alice", "bob"]
    # idempotence across runs: the per-profile index blocks re-queueing
    assert rank_mod.run(conn, cfg)["proposed"] == 0


def test_rank_one_profiles_reject_does_not_block_the_other(tmp_path):
    conn, cfg, x = _two_profile_artist_store(tmp_path)
    db.add_event(conn, db.now(), x, "reject", 1.0, "user", profile="alice")

    stats = rank_mod.run(conn, cfg)

    assert stats["proposed"] == 1
    rows = conn.execute(
        "SELECT profile FROM recommendations WHERE item_id=? AND status='proposed'",
        (x,)).fetchall()
    # alice said no — that verdict is hers alone, bob still gets the pick
    assert [r["profile"] for r in rows] == ["bob"]
