"""Offline tests for the approval queue and the recipe pipeline."""

from __future__ import annotations

import json
import sys
import types

import pytest

from gustarr import config as C
from gustarr import db, pipeline, queue
from gustarr.signals import WEIGHTS, aggregate_label


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def cfg(tmp_path):
    # every sync section configured → no stage skipped
    return C._build({
        "core": {"data_dir": str(tmp_path)},
        "jellyfin": {"url": "http://jf:8096", "api_key": "jk"},
        "lastfm": {"api_key": "lk", "user": "u"},
        "listenbrainz": {"user": "u"},
        "tmdb": {"api_key": "tk"},
        "radarr": {"url": "http://radarr:7878", "api_key": "rk"},
    })


def add_rec(conn, item_id, domain="movie", title=None, year=None, score=0.5,
            why=None, status="proposed", genres=None):
    db.upsert_item(conn, item_id, domain, title, year,
                   meta={"genres": genres} if genres else None)
    cur = conn.execute(
        "INSERT INTO recommendations (run_id, ts, domain, item_id, score, why, status)"
        " VALUES ('r1', ?, ?, ?, ?, ?, ?)",
        (db.now(), domain, item_id, score, json.dumps(why or {}), status))
    return cur.lastrowid


# ── queue: list ──────────────────────────────────────────────────────


def test_list_recs_default_open_queue_score_desc(conn):
    low = add_rec(conn, "movie:tmdb:1", title="Low", year=2001, score=0.2, genres=["Drama"])
    high = add_rec(conn, "movie:tmdb:2", title="High", year=2002, score=0.9,
                   why={"sources": ["tmdb_similar"]})
    add_rec(conn, "series:tvdb:3", domain="series", title="Done", score=0.7, status="added")

    rows = queue.list_recs(conn)
    assert [r["id"] for r in rows] == [high, low]
    assert rows[0]["title"] == "High" and rows[0]["year"] == 2002
    assert rows[0]["why"] == {"sources": ["tmdb_similar"]}  # parsed, not a json string
    assert rows[1]["genres"] == ["Drama"]
    assert all(r["status"] == "proposed" for r in rows)


def test_list_recs_filters(conn):
    add_rec(conn, "movie:tmdb:1", title="M", score=0.4)
    add_rec(conn, "series:tvdb:2", domain="series", title="S", score=0.6)
    add_rec(conn, "movie:tmdb:3", title="A", score=0.5, status="approved")

    assert [r["title"] for r in queue.list_recs(conn, domain="movie")] == ["M"]
    assert len(queue.list_recs(conn, status="all")) == 3
    assert [r["title"] for r in queue.list_recs(conn, status="approved")] == ["A"]


# ── queue: approve / reject ──────────────────────────────────────────


def test_approve_flips_status_and_writes_event(conn):
    rid = add_rec(conn, "movie:tmdb:603", title="The Matrix", year=1999, score=0.9)
    stats = queue.set_status(conn, rid, "approved")
    assert stats == {"updated": 1, "events": 1}

    rec = conn.execute("SELECT * FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "approved"
    assert rec["acted_at"] is not None

    ev = conn.execute("SELECT * FROM events").fetchone()
    assert ev["item_id"] == "movie:tmdb:603"
    assert ev["kind"] == "approve"
    assert ev["source"] == "user"
    assert ev["weight"] == WEIGHTS["approve"]
    assert ev["ts"] == rec["acted_at"]
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_reject_writes_negative_event(conn):
    rid = add_rec(conn, "artist:mbid:aa", domain="artist", title="Nickelback")
    queue.set_status(conn, rid, "rejected")

    rec = conn.execute("SELECT status FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "rejected"
    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["item_id"] == "artist:mbid:aa"
    assert ev["weight"] == WEIGHTS["reject"] == -1.0
    assert ev["source"] == "user"


def test_exploration_reject_is_soft(conn):
    rid = add_rec(conn, "movie:tmdb:8", title="Wildcard", why={"exploration": True})
    queue.set_status(conn, rid, "rejected")

    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["weight"] == pytest.approx(WEIGHTS["reject"] * 0.3)
    assert ev["weight"] == pytest.approx(-0.3)
    assert json.loads(ev["meta"]) == {"rec_id": rid, "exploration": True}
    # the stored weight is what training sees — no special-casing downstream
    assert aggregate_label([(ev["ts"], ev["kind"], ev["weight"])]) == pytest.approx(-0.3, abs=1e-3)


def test_reject_with_falsy_exploration_stays_full_weight(conn):
    rid = add_rec(conn, "movie:tmdb:8", title="Safe Bet", why={"exploration": False})
    queue.set_status(conn, rid, "rejected")

    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["weight"] == WEIGHTS["reject"] == -1.0
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_exploration_approve_keeps_full_weight(conn):
    rid = add_rec(conn, "movie:tmdb:9", title="Wildcard", why={"exploration": True})
    queue.set_status(conn, rid, "approved")

    ev = conn.execute("SELECT * FROM events WHERE kind='approve'").fetchone()
    assert ev["weight"] == WEIGHTS["approve"] == 1.0
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_double_approve_raises(conn):
    rid = add_rec(conn, "movie:tmdb:1", title="M")
    queue.set_status(conn, rid, "approved")
    with pytest.raises(ValueError, match="already approved"):
        queue.set_status(conn, rid, "approved")
    # only the first verdict produced an event
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_set_status_guards(conn):
    with pytest.raises(ValueError, match="no recommendation"):
        queue.set_status(conn, 999, "approved")

    terminal = add_rec(conn, "movie:tmdb:1", title="M", status="added")
    with pytest.raises(ValueError, match="added"):
        queue.set_status(conn, terminal, "rejected")

    auto = add_rec(conn, "movie:tmdb:2", title="N", status="auto_added")
    with pytest.raises(ValueError, match="auto_added"):
        queue.set_status(conn, auto, "approved")

    open_rec = add_rec(conn, "movie:tmdb:3", title="O")
    with pytest.raises(ValueError):
        queue.set_status(conn, open_rec, "added")  # apply's business, not the queue's
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


# ── queue: explain ───────────────────────────────────────────────────


def test_explain_renders_all_sections(conn):
    db.upsert_item(conn, "movie:tmdb:604", "movie", "Heat", 1995)
    rid = add_rec(
        conn, "movie:tmdb:603", title="The Matrix", year=1999, score=0.87,
        genres=["Action", "Sci-Fi"],
        why={
            "sources": ["tmdb_similar", "tmdb_discover"],
            "seeds": ["movie:tmdb:604"],
            "neighbors": [
                {"item_id": "movie:tmdb:604", "sim": 0.82},
                {"title": "Blade Runner", "sim": 0.6},
            ],
            "exploration": True,
        })

    text = queue.explain(conn, rid)
    assert "The Matrix" in text and "1999" in text
    assert "Action, Sci-Fi" in text
    assert "+0.87" in text
    assert "tmdb_similar" in text and "tmdb_discover" in text
    assert "seeded by: Heat" in text  # seed id resolved via items table
    assert "because you liked Heat (sim .82)" in text
    assert "because you liked Blade Runner (sim .60)" in text
    assert "exploration" in text.lower()
    assert len(text.splitlines()) >= 6


def test_explain_serendipity_marker(conn):
    ser = add_rec(conn, "movie:tmdb:70", title="Left Field", year=2011,
                  why={"sources": ["serendipity"], "serendipity": True})
    plain = add_rec(conn, "movie:tmdb:71", title="Safe Bet", year=2012,
                    why={"sources": ["tmdb_similar"]})

    assert "(serendipity)" in queue.explain(conn, ser)
    assert "(serendipity)" not in queue.explain(conn, plain)


def test_explain_unknown_rec_raises(conn):
    with pytest.raises(ValueError):
        queue.explain(conn, 42)


# ── queue: store stats ───────────────────────────────────────────────


def test_store_stats(conn):
    add_rec(conn, "movie:tmdb:1", title="M", score=0.5)
    add_rec(conn, "movie:tmdb:2", title="N", score=0.6, status="added")
    db.upsert_item(conn, "artist:mbid:aa", "artist", "Radiohead")
    db.add_event(conn, "2024-01-01T00:00:00Z", "artist:mbid:aa", "scrobble",
                 WEIGHTS["scrobble"], "lastfm")
    db.add_event(conn, "2024-01-02T00:00:00Z", "artist:mbid:aa", "scrobble",
                 WEIGHTS["scrobble"], "lastfm")
    db.add_event(conn, "2024-01-03T00:00:00Z", "movie:tmdb:1", "complete",
                 WEIGHTS["complete"], "jellyfin")
    conn.execute(
        "INSERT INTO candidates (item_id, source, first_seen, last_seen)"
        " VALUES ('movie:tmdb:2', 'tmdb_similar', ?, ?)", (db.now(), db.now()))
    db.put_embedding(conn, "artist:mbid:aa", "m", b"\x00\x00", 1)
    db.set_state(conn, "lastfm:last_uts", "1700000000")
    db.set_state(conn, "arr:known:radarr", json.dumps({"movie:tmdb:1": 11}))
    db.set_state(conn, "model:movie", json.dumps({"trained_at": "2026-07-01T00:00:00Z"}))

    s = queue.store_stats(conn)
    assert s["tables"]["items"] == 3
    assert s["tables"]["events"] == 3
    assert s["tables"]["recommendations"] == 2
    assert s["tables"]["candidates"] == 1
    assert s["tables"]["embeddings"] == 1
    assert s["events_by_kind"] == {"scrobble": 2, "complete": 1}
    assert s["recs_by_status"] == {"proposed": 1, "added": 1}
    assert s["sync"]["lastfm:last_uts"] == "1700000000"
    assert s["sync"]["arr:known:radarr"] is True
    assert s["models"]["movie"] == "2026-07-01T00:00:00Z"
    # 3 items carry events/candidates; only the artist has a vector
    assert s["embedding_coverage"] == {"needed": 3, "embedded": 1}


def test_store_stats_diversity_entropy_and_share(conn):
    # recs: Drama x2, Comedy x2 — an even 2-genre split is exactly 1 bit;
    # the genre-less rec must be skipped, not counted as a genre
    add_rec(conn, "movie:tmdb:1", title="A", genres=["Drama"])
    add_rec(conn, "movie:tmdb:2", title="B", genres=["Comedy"])
    add_rec(conn, "movie:tmdb:3", title="C", genres=["Drama", "Comedy"])
    add_rec(conn, "movie:tmdb:4", title="D", why={"exploration": True})

    # library: Drama x3, Sci-Fi x1 → -(0.75*log2(0.75) + 0.25*log2(0.25)) = 0.811
    for i, genre in enumerate(["Drama", "Drama", "Drama", "Sci-Fi"]):
        item = db.upsert_item(conn, f"movie:tmdb:{100 + i}", "movie", f"L{i}",
                              meta={"genres": [genre]})
        conn.execute("INSERT INTO library (item_id, arr) VALUES (?, 'radarr')", (item,))

    d = queue.store_stats(conn)["diversity"]
    assert d["genre_entropy_recs"] == 1.0
    assert d["genre_entropy_library"] == 0.811
    assert d["exploration_share"] == 0.25
    assert d["exploration_approval_rate"] is None  # nothing acted yet


def test_store_stats_diversity_empty_store(conn):
    d = queue.store_stats(conn)["diversity"]
    assert d == {"genre_entropy_recs": 0.0, "genre_entropy_library": 0.0,
                 "exploration_share": 0.0, "exploration_approval_rate": None}


def test_store_stats_exploration_approval_rate(conn):
    def rec(i, status, exploration):
        add_rec(conn, f"movie:tmdb:{i}", title=f"T{i}", status=status,
                why={"exploration": True} if exploration else {})

    rec(1, "approved", True)
    rec(2, "rejected", True)
    rec(3, "added", True)
    rec(4, "auto_added", True)  # acted, but not a user approval
    rec(5, "proposed", True)    # not acted → excluded entirely
    rec(6, "approved", False)   # not exploration → excluded entirely

    d = queue.store_stats(conn)["diversity"]
    assert d["exploration_approval_rate"] == 0.5  # approved + added out of 4 acted


def test_store_stats_diversity_windows_last_100_recs(conn):
    # the oldest rec would poison both metrics if the 100-rec window leaked
    add_rec(conn, "movie:tmdb:1000", title="Old", genres=["Western"],
            why={"exploration": True})
    for i in range(100):
        add_rec(conn, f"movie:tmdb:{i}", title=f"T{i}",
                genres=["Drama"] if i % 2 else ["Comedy"])

    d = queue.store_stats(conn)["diversity"]
    assert d["genre_entropy_recs"] == 1.0  # 50/50 Drama-Comedy; Western aged out
    assert d["exploration_share"] == 0.0


# ── pipeline ─────────────────────────────────────────────────────────


def fake_stages(monkeypatch, fail=frozenset()):
    """Install fake modules for every stage; import_module finds them in
    sys.modules so real (possibly missing) modules are never touched."""
    calls = []
    for name, (module_path, func_name, _) in pipeline.STAGES.items():
        mod = types.ModuleType(module_path)

        def make(stage):
            def run(conn, cfg, **kw):
                calls.append((stage, kw))
                if stage in fail:
                    raise RuntimeError(f"{stage} blew up")
                return {"n": 1}
            return run

        setattr(mod, func_name, make(name))
        monkeypatch.setitem(sys.modules, module_path, mod)
    return calls


class CommitSpy:
    def __init__(self, conn):
        self._conn = conn
        self.commits = 0

    def commit(self):
        self.commits += 1
        self._conn.commit()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_nightly_runs_stages_in_order_and_commits_each(conn, cfg, monkeypatch):
    calls = fake_stages(monkeypatch)
    spy = CommitSpy(conn)
    stats = pipeline.run_recipe(spy, cfg, "nightly")

    assert [c[0] for c in calls] == pipeline.NIGHTLY
    assert stats["errors"] == []
    for name in pipeline.NIGHTLY:
        assert stats[name] == {"n": 1}
    assert "apply" not in stats
    assert spy.commits == len(pipeline.NIGHTLY)


def test_unconfigured_sync_stages_skipped(conn, tmp_path, monkeypatch):
    calls = fake_stages(monkeypatch)
    bare = C._build({"core": {"data_dir": str(tmp_path)}})
    stats = pipeline.run_recipe(conn, bare, "nightly")

    for stage in ("sync_arr", "sync_jellyfin", "sync_lastfm", "sync_listenbrainz"):
        assert stats[stage] == "skipped"
    assert [c[0] for c in calls] == ["enrich", "candidates", "embed", "train", "rank"]
    assert stats["errors"] == []


def test_lastfm_api_key_only_skips_sync_stage(conn, tmp_path, monkeypatch):
    # api_key without user serves enrich/candidates but cannot sync —
    # the stage must skip, not run (and error) every night
    calls = fake_stages(monkeypatch)
    partial = C._build({"core": {"data_dir": str(tmp_path)},
                        "lastfm": {"api_key": "lk"}})
    stats = pipeline.run_recipe(conn, partial, "nightly")

    assert stats["sync_lastfm"] == "skipped"
    assert "sync_lastfm" not in [c[0] for c in calls]
    assert stats["errors"] == []


def test_stage_error_is_isolated_and_still_committed(conn, cfg, monkeypatch):
    calls = fake_stages(monkeypatch, fail={"embed"})
    spy = CommitSpy(conn)
    stats = pipeline.run_recipe(spy, cfg, "nightly")

    assert stats["embed"] == {"error": "embed blew up"}
    assert stats["errors"] == ["embed"]
    ran = [c[0] for c in calls]
    assert ran.index("train") > ran.index("embed")  # run continues past the failure
    assert stats["train"] == {"n": 1} and stats["rank"] == {"n": 1}
    assert spy.commits == len(pipeline.NIGHTLY)  # commit happens even for the failed stage


def test_unimportable_stage_module_breaks_only_its_stage(conn, cfg, monkeypatch):
    calls = fake_stages(monkeypatch)
    # None in sys.modules makes import_module raise ImportError — same shape
    # as a missing optional dep at stage-import time.
    monkeypatch.setitem(sys.modules, "gustarr.ml.embed", None)
    stats = pipeline.run_recipe(conn, cfg, "nightly")

    assert stats["errors"] == ["embed"]
    assert "error" in stats["embed"]
    assert stats["train"] == {"n": 1} and stats["rank"] == {"n": 1}
    assert "embed" not in [c[0] for c in calls]


def test_weekly_appends_apply_and_threads_dry_run(conn, cfg, monkeypatch):
    calls = fake_stages(monkeypatch)
    stats = pipeline.run_recipe(conn, cfg, "weekly", dry_run=True)

    assert [c[0] for c in calls] == pipeline.WEEKLY
    assert calls[-1] == ("apply", {"dry_run": True})
    assert all(kw == {} for _, kw in calls[:-1])  # dry_run only reaches apply
    assert stats["apply"] == {"n": 1}


def test_unknown_recipe_raises(conn, cfg):
    with pytest.raises(ValueError, match="unknown recipe"):
        pipeline.run_recipe(conn, cfg, "hourly")
