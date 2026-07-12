"""Offline tests for the approval queue and the recipe pipeline."""

from __future__ import annotations

import json
import sys
import types

import pytest

from gustarr import config as C
from gustarr import db, pipeline, queue, settings
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


def add_rec(conn, ident, domain="movie", title=None, year=None, score=0.5,
            why=None, status="proposed", genres=None, meta=None, profile="default"):
    """ident is 'ns:key' — resolved through THE write path, so the same
    spelling in two calls lands on the same item, as in production."""
    ns, key = ident.split(":", 1)
    item_meta = dict(meta or {})
    if genres:
        item_meta["genres"] = genres
    item_id = db.resolve_item(conn, domain, ns, key, title=title, year=year,
                              meta=item_meta or None)
    cur = conn.execute(
        "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score, why, status)"
        " VALUES (?, 'r1', ?, ?, ?, ?, ?, ?)",
        (profile, db.now(), domain, item_id, score, json.dumps(why or {}), status))
    return cur.lastrowid


def item_of(conn, rec_id):
    return conn.execute(
        "SELECT item_id FROM recommendations WHERE id=?", (rec_id,)).fetchone()["item_id"]


# ── queue: list ──────────────────────────────────────────────────────


def test_list_recs_default_open_queue_score_desc(conn):
    low = add_rec(conn, "tmdb:1", title="Low", year=2001, score=0.2, genres=["Drama"])
    high = add_rec(conn, "tmdb:2", title="High", year=2002, score=0.9,
                   why={"sources": ["tmdb_similar"]})
    add_rec(conn, "tvdb:3", domain="series", title="Done", score=0.7, status="added")

    rows = queue.list_recs(conn)
    assert [r["id"] for r in rows] == [high, low]
    assert rows[0]["title"] == "High" and rows[0]["year"] == 2002
    assert rows[0]["why"] == {"sources": ["tmdb_similar"]}  # parsed, not a json string
    assert rows[1]["genres"] == ["Drama"]
    assert all(r["status"] == "proposed" for r in rows)


def test_list_recs_surfaces_poster_and_truncated_overview(conn):
    rich = add_rec(conn, "tmdb:10", title="Rich", score=0.9,
                   meta={"poster_path": "/p10.jpg", "overview": "o" * 500})
    bare = add_rec(conn, "tmdb:11", title="Bare", score=0.1)

    rows = {r["id"]: r for r in queue.list_recs(conn)}
    assert rows[rich]["poster_path"] == "/p10.jpg"
    assert rows[rich]["overview"] == "o" * 220
    # pre-poster items must list cleanly, not KeyError
    assert rows[bare]["poster_path"] is None
    assert rows[bare]["overview"] is None


def test_list_recs_surfaces_trailer_and_tolerates_absence(conn):
    with_clip = add_rec(conn, "tmdb:12", title="Clip", score=0.8,
                        meta={"trailer": "yt-abc123"})
    without = add_rec(conn, "tmdb:13", title="NoClip", score=0.2)

    rows = {r["id"]: r for r in queue.list_recs(conn)}
    assert rows[with_clip]["trailer"] == "yt-abc123"
    # pre-trailer items (enriched before videos were fetched) must list cleanly
    assert rows[without]["trailer"] is None


def test_list_recs_filters(conn):
    add_rec(conn, "tmdb:1", title="M", score=0.4)
    add_rec(conn, "tvdb:2", domain="series", title="S", score=0.6)
    add_rec(conn, "tmdb:3", title="A", score=0.5, status="approved")

    assert [r["title"] for r in queue.list_recs(conn, domain="movie")] == ["M"]
    assert len(queue.list_recs(conn, status="all")) == 3
    assert [r["title"] for r in queue.list_recs(conn, status="approved")] == ["A"]


def test_list_recs_exposes_identities_under_ids(conn):
    rid = add_rec(conn, "tmdb:603", title="The Matrix", score=0.9)
    db.attach_identity(conn, item_of(conn, rid), "imdb", "tt0133093")

    row = queue.list_recs(conn)[0]
    # the response key stays 'ids' for web/CLI compat (they read mbid/tvdb
    # out of it); the content now comes from the identities table
    assert row["ids"] == {"tmdb": "603", "imdb": "tt0133093"}
    assert row["item_id"] == item_of(conn, rid)


def test_list_recs_surfaces_image_artist_and_type(conn):
    album = add_rec(conn, "mbid:rg1", domain="album", title="In Rainbows", year=2007,
                    score=0.8, meta={"image": "https://img.example/ir.jpg",
                                     "artist": "Radiohead", "type": "Album"})
    bare = add_rec(conn, "mbid:aa", domain="artist", title="Radiohead", score=0.3)

    rows = {r["id"]: r for r in queue.list_recs(conn)}
    assert rows[album]["image"] == "https://img.example/ir.jpg"
    assert rows[album]["artist"] == "Radiohead"
    assert rows[album]["type"] == "Album"
    # pre-enrichment items must list cleanly, not KeyError
    assert rows[bare]["image"] is None
    assert rows[bare]["artist"] is None
    assert rows[bare]["type"] is None


def test_list_recs_music_domain_alias(conn):
    artist = add_rec(conn, "mbid:aa", domain="artist", title="Artist", score=0.9)
    album = add_rec(conn, "mbid:rg1", domain="album", title="Album", score=0.5)
    add_rec(conn, "tmdb:1", title="Movie", score=0.7)

    # 'music' spans both audio domains; score ordering still applies
    assert [r["id"] for r in queue.list_recs(conn, domain="music")] == [artist, album]
    # the alias never leaks into exact-domain filters
    assert [r["id"] for r in queue.list_recs(conn, domain="album")] == [album]
    assert [r["id"] for r in queue.list_recs(conn, domain="artist")] == [artist]


def test_list_recs_scoped_to_profile(conn):
    alice = add_rec(conn, "tmdb:1", title="Hers", score=0.9, profile="alice")
    add_rec(conn, "tmdb:2", title="His", score=0.8, profile="bob")
    default = add_rec(conn, "tmdb:3", title="Legacy", score=0.7)

    # no profile argument = the synthesized single-user profile, so
    # pre-profile callers (web, scripts) keep seeing exactly their queue
    assert [r["id"] for r in queue.list_recs(conn)] == [default]
    rows = queue.list_recs(conn, profile="alice")
    assert [r["id"] for r in rows] == [alice]
    assert rows[0]["profile"] == "alice"


# ── queue: approve / reject ──────────────────────────────────────────


def test_approve_flips_status_and_writes_event(conn):
    rid = add_rec(conn, "tmdb:603", title="The Matrix", year=1999, score=0.9)
    stats = queue.set_status(conn, rid, "approved")
    assert stats == {"updated": 1, "events": 1}

    rec = conn.execute("SELECT * FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "approved"
    assert rec["acted_at"] is not None

    ev = conn.execute("SELECT * FROM events").fetchone()
    assert ev["item_id"] == item_of(conn, rid)
    assert ev["kind"] == "approve"
    assert ev["source"] == "user"
    assert ev["weight"] == WEIGHTS["approve"]
    assert ev["ts"] == rec["acted_at"]
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_reject_writes_negative_event(conn):
    rid = add_rec(conn, "mbid:aa", domain="artist", title="Nickelback")
    queue.set_status(conn, rid, "rejected")

    rec = conn.execute("SELECT status FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "rejected"
    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["item_id"] == item_of(conn, rid)
    assert ev["weight"] == WEIGHTS["reject"] == -1.0
    assert ev["source"] == "user"


def test_exploration_reject_is_soft(conn):
    rid = add_rec(conn, "tmdb:8", title="Wildcard", why={"exploration": True})
    queue.set_status(conn, rid, "rejected")

    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["weight"] == pytest.approx(WEIGHTS["reject"] * 0.3)
    assert ev["weight"] == pytest.approx(-0.3)
    assert json.loads(ev["meta"]) == {"rec_id": rid, "exploration": True}
    # the stored weight is what training sees — no special-casing downstream
    assert aggregate_label([(ev["ts"], ev["kind"], ev["weight"])]) == pytest.approx(-0.3, abs=1e-3)


def test_reject_with_falsy_exploration_stays_full_weight(conn):
    rid = add_rec(conn, "tmdb:8", title="Safe Bet", why={"exploration": False})
    queue.set_status(conn, rid, "rejected")

    ev = conn.execute("SELECT * FROM events WHERE kind='reject'").fetchone()
    assert ev["weight"] == WEIGHTS["reject"] == -1.0
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_exploration_approve_keeps_full_weight(conn):
    rid = add_rec(conn, "tmdb:9", title="Wildcard", why={"exploration": True})
    queue.set_status(conn, rid, "approved")

    ev = conn.execute("SELECT * FROM events WHERE kind='approve'").fetchone()
    assert ev["weight"] == WEIGHTS["approve"] == 1.0
    assert json.loads(ev["meta"]) == {"rec_id": rid}


def test_set_status_profile_guard_and_event_attribution(conn):
    rid = add_rec(conn, "tmdb:40", title="Hers", profile="alice")
    # the guard only bites when a profile is supplied and wrong
    with pytest.raises(ValueError, match="belongs to profile 'alice'"):
        queue.set_status(conn, rid, "approved", profile="bob")
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0

    # no profile given: the rec id is trusted — and the taste event must
    # land on the rec owner's profile so it trains *her* model
    queue.set_status(conn, rid, "approved")
    ev = conn.execute("SELECT profile, kind FROM events").fetchone()
    assert (ev["profile"], ev["kind"]) == ("alice", "approve")


def test_forgive_and_explain_profile_guard(conn):
    rid = add_rec(conn, "tmdb:41", title="Hers", profile="alice")
    with pytest.raises(ValueError, match="belongs to profile 'alice'"):
        queue.explain(conn, rid, profile="bob")
    assert "Hers" in queue.explain(conn, rid, profile="alice")
    queue.set_status(conn, rid, "rejected", profile="alice")
    with pytest.raises(ValueError, match="belongs to profile 'alice'"):
        queue.forgive(conn, rid, profile="bob")
    assert queue.forgive(conn, rid, profile="alice") == {"deleted_events": 1}


def test_double_approve_raises(conn):
    rid = add_rec(conn, "tmdb:1", title="M")
    queue.set_status(conn, rid, "approved")
    with pytest.raises(ValueError, match="already approved"):
        queue.set_status(conn, rid, "approved")
    # only the first verdict produced an event
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_set_status_guards(conn):
    with pytest.raises(ValueError, match="no recommendation"):
        queue.set_status(conn, 999, "approved")

    terminal = add_rec(conn, "tmdb:1", title="M", status="added")
    with pytest.raises(ValueError, match="added"):
        queue.set_status(conn, terminal, "rejected")

    auto = add_rec(conn, "tmdb:2", title="N", status="auto_added")
    with pytest.raises(ValueError, match="auto_added"):
        queue.set_status(conn, auto, "approved")

    open_rec = add_rec(conn, "tmdb:3", title="O")
    with pytest.raises(ValueError):
        queue.set_status(conn, open_rec, "added")  # apply's business, not the queue's
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


# ── queue: snooze ────────────────────────────────────────────────────


def test_snooze_sets_acted_at_and_writes_no_event(conn):
    rid = add_rec(conn, "tmdb:20", title="Later", score=0.7)
    stats = queue.set_status(conn, rid, "snoozed")
    assert stats == {"updated": 1, "events": 0}

    rec = conn.execute("SELECT * FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "snoozed"
    assert rec["acted_at"] is not None
    # not a verdict — the model must learn nothing from a snooze
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_double_snooze_raises(conn):
    rid = add_rec(conn, "tmdb:21", title="Later")
    queue.set_status(conn, rid, "snoozed")
    with pytest.raises(ValueError, match="already snoozed"):
        queue.set_status(conn, rid, "snoozed")


def test_snooze_only_from_proposed(conn):
    approved = add_rec(conn, "tmdb:22", title="Yes", status="approved")
    with pytest.raises(ValueError, match="approved"):
        queue.set_status(conn, approved, "snoozed")
    rejected = add_rec(conn, "tmdb:23", title="No", status="rejected")
    with pytest.raises(ValueError, match="rejected"):
        queue.set_status(conn, rejected, "snoozed")
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_snoozed_rec_can_still_be_approved(conn):
    rid = add_rec(conn, "tmdb:24", title="Actually Yes")
    queue.set_status(conn, rid, "snoozed")
    queue.set_status(conn, rid, "approved")
    rec = conn.execute("SELECT status FROM recommendations WHERE id=?", (rid,)).fetchone()
    assert rec["status"] == "approved"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='approve'").fetchone()[0] == 1


def test_list_recs_filters_snoozed(conn):
    add_rec(conn, "tmdb:25", title="Open")
    snoozed = add_rec(conn, "tmdb:26", title="Napping")
    queue.set_status(conn, snoozed, "snoozed")

    assert [r["title"] for r in queue.list_recs(conn, status="snoozed")] == ["Napping"]
    assert [r["title"] for r in queue.list_recs(conn)] == ["Open"]


# ── queue: forgive ───────────────────────────────────────────────────


def test_forgive_deletes_only_that_recs_reject_and_expires_it(conn):
    # two rejected recs for the SAME item (legal: the partial unique index
    # only covers open statuses) — forgiving one must not touch the other's
    # reject event
    first = add_rec(conn, "tmdb:30", title="Redeemed", score=0.5)
    queue.set_status(conn, first, "rejected")
    # backdate the first verdict: same-second rejects of one item would
    # collapse into a single row under the events uniqueness key
    conn.execute("UPDATE events SET ts='2026-01-01T00:00:00Z' WHERE kind='reject'")
    second = add_rec(conn, "tmdb:30", title="Redeemed", score=0.6)
    queue.set_status(conn, second, "rejected")
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind='reject'").fetchone()[0] == 2

    stats = queue.forgive(conn, first)
    assert stats == {"deleted_events": 1}

    rec = conn.execute("SELECT status FROM recommendations WHERE id=?", (first,)).fetchone()
    assert rec["status"] == "expired"
    survivors = conn.execute("SELECT meta FROM events WHERE kind='reject'").fetchall()
    assert [json.loads(r["meta"])["rec_id"] for r in survivors] == [second]
    other = conn.execute("SELECT status FROM recommendations WHERE id=?", (second,)).fetchone()
    assert other["status"] == "rejected"


def test_forgive_spares_other_event_kinds(conn):
    rid = add_rec(conn, "tmdb:31", title="Watched Anyway")
    queue.set_status(conn, rid, "rejected")
    db.add_event(conn, "2026-01-01T00:00:00Z", item_of(conn, rid), "complete",
                 WEIGHTS["complete"], "jellyfin")

    assert queue.forgive(conn, rid) == {"deleted_events": 1}
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events")]
    assert kinds == ["complete"]


def test_forgive_requires_rejected(conn):
    with pytest.raises(ValueError, match="no recommendation"):
        queue.forgive(conn, 999)
    proposed = add_rec(conn, "tmdb:32", title="Open")
    with pytest.raises(ValueError, match="proposed"):
        queue.forgive(conn, proposed)
    added = add_rec(conn, "tmdb:33", title="Done", status="added")
    with pytest.raises(ValueError, match="added"):
        queue.forgive(conn, added)


# ── queue: explain ───────────────────────────────────────────────────


def test_explain_renders_all_sections(conn):
    # why-json references items by their INT ids now
    heat = db.resolve_item(conn, "movie", "tmdb", "604", title="Heat", year=1995)
    rid = add_rec(
        conn, "tmdb:603", title="The Matrix", year=1999, score=0.87,
        genres=["Action", "Sci-Fi"],
        why={
            "sources": ["tmdb_similar", "tmdb_discover"],
            "seeds": [heat],
            "neighbors": [
                {"item_id": heat, "sim": 0.82},
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


def test_explain_resolves_bare_int_refs(conn):
    # a bare int is looked up in items; a titleless item degrades to its
    # id, and a numeric STRING stays text (it may be a title like "1984")
    seed = db.resolve_item(conn, "movie", "tmdb", "604", title="Heat")
    ghost = db.resolve_item(conn, "movie", "tmdb", "605")  # never enriched
    rid = add_rec(conn, "tmdb:606", title="Ronin",
                  why={"neighbors": [seed, ghost], "seeds": ["1984"]})
    text = queue.explain(conn, rid)
    assert "because you liked Heat" in text
    assert f"because you liked item #{ghost}" in text
    assert "seeded by: 1984" in text


def test_explain_serendipity_marker(conn):
    ser = add_rec(conn, "tmdb:70", title="Left Field", year=2011,
                  why={"sources": ["serendipity"], "serendipity": True})
    plain = add_rec(conn, "tmdb:71", title="Safe Bet", year=2012,
                    why={"sources": ["tmdb_similar"]})

    assert "(serendipity)" in queue.explain(conn, ser)
    assert "(serendipity)" not in queue.explain(conn, plain)


def test_explain_unknown_rec_raises(conn):
    with pytest.raises(ValueError):
        queue.explain(conn, 42)


# ── queue: store stats ───────────────────────────────────────────────


def test_store_stats(conn):
    m1 = add_rec(conn, "tmdb:1", title="M", score=0.5)
    m2 = add_rec(conn, "tmdb:2", title="N", score=0.6, status="added")
    add_rec(conn, "tmdb:4", title="Z", score=0.4, status="snoozed")
    settings.set(conn, "paused", True)
    artist = db.resolve_item(conn, "artist", "mbid", "aa", title="Radiohead")
    db.add_event(conn, "2024-01-01T00:00:00Z", artist, "scrobble",
                 WEIGHTS["scrobble"], "lastfm")
    db.add_event(conn, "2024-01-02T00:00:00Z", artist, "scrobble",
                 WEIGHTS["scrobble"], "lastfm")
    db.add_event(conn, "2024-01-03T00:00:00Z", item_of(conn, m1), "complete",
                 WEIGHTS["complete"], "jellyfin")
    conn.execute(
        "INSERT INTO candidates (item_id, source, first_seen, last_seen)"
        " VALUES (?, 'tmdb_similar', ?, ?)", (item_of(conn, m2), db.now(), db.now()))
    db.put_embedding(conn, artist, "m", b"\x00\x00", 1)
    # cursors and models live in the profile namespace now; arr inventory
    # stays global
    db.pset_state(conn, "default", "lastfm:last_uts", "1700000000")
    db.set_state(conn, "arr:known:radarr", json.dumps({"1": 11}))
    db.pset_state(conn, "default", "model:movie",
                  json.dumps({"trained_at": "2026-07-01T00:00:00Z"}))

    s = queue.store_stats(conn)
    assert s["profile"] == "default"
    assert s["tables"]["items"] == 4
    assert s["tables"]["identities"] == 4  # one spelling each, v3's new table
    assert s["tables"]["events"] == 3
    assert s["tables"]["recommendations"] == 3
    assert s["tables"]["candidates"] == 1
    assert s["tables"]["embeddings"] == 1
    assert s["events_by_kind"] == {"scrobble": 2, "complete": 1}
    # snoozed shows up like any other status — no special-casing
    assert s["recs_by_status"] == {"proposed": 1, "added": 1, "snoozed": 1}
    assert s["sync"]["lastfm:last_uts"] == "1700000000"
    assert s["sync"]["arr:known:radarr"] is True
    assert s["models"]["movie"] == "2026-07-01T00:00:00Z"
    assert s["settings_overridden"] == 1  # only setting:% keys count
    # 3 items carry events/candidates; only the artist has a vector
    assert s["embedding_coverage"] == {"needed": 3, "embedded": 1}


def test_store_stats_scoped_to_profile(conn):
    add_rec(conn, "tmdb:1", title="Hers", profile="alice")
    add_rec(conn, "tmdb:2", title="His", status="approved", profile="bob")
    db.pset_state(conn, "alice", "model:movie",
                  json.dumps({"trained_at": "2026-07-01T00:00:00Z"}))
    db.pset_state(conn, "alice", "lastfm:last_uts", "1700000000")

    s = queue.store_stats(conn, profile="alice")
    assert s["profile"] == "alice"
    assert s["recs_by_status"] == {"proposed": 1}  # bob's approval invisible
    assert s["models"] == {"movie": "2026-07-01T00:00:00Z"}
    assert s["sync"] == {"lastfm:last_uts": "1700000000"}
    assert s["tables"]["recommendations"] == 2  # table totals stay global

    s = queue.store_stats(conn, profile="bob")
    assert s["recs_by_status"] == {"approved": 1}
    assert s["models"] == {}  # bob hasn't trained yet — alice's model isn't his
    assert s["sync"] == {}


def test_store_stats_diversity_entropy_and_share(conn):
    # recs: Drama x2, Comedy x2 — an even 2-genre split is exactly 1 bit;
    # the genre-less rec must be skipped, not counted as a genre
    add_rec(conn, "tmdb:1", title="A", genres=["Drama"])
    add_rec(conn, "tmdb:2", title="B", genres=["Comedy"])
    add_rec(conn, "tmdb:3", title="C", genres=["Drama", "Comedy"])
    add_rec(conn, "tmdb:4", title="D", why={"exploration": True})

    # library: Drama x3, Sci-Fi x1 → -(0.75*log2(0.75) + 0.25*log2(0.25)) = 0.811
    for i, genre in enumerate(["Drama", "Drama", "Drama", "Sci-Fi"]):
        item = db.resolve_item(conn, "movie", "tmdb", str(100 + i), title=f"L{i}",
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
        add_rec(conn, f"tmdb:{i}", title=f"T{i}", status=status,
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
    add_rec(conn, "tmdb:1000", title="Old", genres=["Western"],
            why={"exploration": True})
    for i in range(100):
        add_rec(conn, f"tmdb:{i}", title=f"T{i}",
                genres=["Drama"] if i % 2 else ["Comedy"])

    d = queue.store_stats(conn)["diversity"]
    assert d["genre_entropy_recs"] == 1.0  # 50/50 Drama-Comedy; Western aged out
    assert d["exploration_share"] == 0.0


# ── settings ─────────────────────────────────────────────────────────


def test_settings_defaults_come_from_config(conn, cfg):
    assert settings.get(conn, cfg, "paused") is False
    assert settings.get(conn, cfg, "music_mode") == "auto"
    assert settings.get(conn, cfg, "music_max_artists_per_week") == 3
    assert settings.get(conn, cfg, "music_max_albums_per_week") == 10
    assert settings.get(conn, cfg, "video_queue_max_pending") == 20
    assert settings.get(conn, cfg, "exploration_frac") == 0.15


def test_settings_override_wins_and_clear_restores(conn, cfg):
    assert settings.set(conn, "music_mode", "queue") == "queue"
    assert settings.get(conn, cfg, "music_mode") == "queue"
    # persisted json-encoded under the documented state key
    assert db.get_state(conn, "setting:music_mode") == '"queue"'

    settings.clear(conn, "music_mode")
    assert settings.get(conn, cfg, "music_mode") == "auto"
    assert db.get_state(conn, "setting:music_mode") is None
    settings.clear(conn, "music_mode")  # clearing a non-override is a no-op


def test_settings_get_all_reports_overrides(conn, cfg):
    settings.set(conn, "paused", "true")
    allv = settings.get_all(conn, cfg)
    assert set(allv) == set(settings.RUNTIME_KEYS)
    assert allv["paused"] == {"value": True, "overridden": True, "default": False}
    assert allv["exploration_frac"] == {"value": 0.15, "overridden": False, "default": 0.15}


def test_settings_coercion(conn):
    assert settings.set(conn, "music_max_artists_per_week", "3") == 3
    assert settings.set(conn, "paused", "true") is True
    assert settings.set(conn, "paused", "false") is False
    assert settings.set(conn, "exploration_frac", "0.5") == 0.5
    assert settings.set(conn, "video_queue_max_pending", 1) == 1
    # coerced types (not the raw strings) survive the json round-trip
    assert json.loads(db.get_state(conn, "setting:music_max_artists_per_week")) == 3
    assert json.loads(db.get_state(conn, "setting:paused")) is False
    assert json.loads(db.get_state(conn, "setting:exploration_frac")) == 0.5


def test_settings_unknown_key_raises(conn, cfg):
    with pytest.raises(ValueError, match="unknown setting"):
        settings.set(conn, "nope", 1)
    with pytest.raises(ValueError, match="unknown setting"):
        settings.get(conn, cfg, "nope")
    with pytest.raises(ValueError, match="unknown setting"):
        settings.clear(conn, "nope")


def test_settings_invalid_values_raise(conn):
    for key, bad in [
        ("paused", "maybe"),
        ("music_mode", "chaos"),
        ("music_max_artists_per_week", -1),
        ("music_max_artists_per_week", "lots"),
        ("music_max_artists_per_week", 2.5),
        ("music_max_artists_per_week", True),  # bool is not an int here
        ("video_queue_max_pending", 0),
        ("exploration_frac", 0.95),
        ("exploration_frac", -0.1),
        ("exploration_frac", "wide"),
    ]:
        with pytest.raises(ValueError):
            settings.set(conn, key, bad)
    # a failed set must never leave a partial override behind
    n = conn.execute("SELECT COUNT(*) FROM state WHERE key LIKE 'setting:%'").fetchone()[0]
    assert n == 0


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
    assert stats["apply"] == {"n": 1}  # approvals land nightly now
    assert spy.commits == len(pipeline.NIGHTLY)


def test_unconfigured_sync_stages_skipped(conn, tmp_path, monkeypatch):
    calls = fake_stages(monkeypatch)
    bare = C._build({"core": {"data_dir": str(tmp_path)}})
    stats = pipeline.run_recipe(conn, bare, "nightly")

    for stage in ("sync_arr", "sync_jellyfin", "sync_lastfm", "sync_listenbrainz"):
        assert stats[stage] == "skipped"
    assert [c[0] for c in calls] == ["enrich", "candidates", "embed", "train", "rank", "apply"]
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
    # dry_run only reaches apply; enrich alone carries its batch bound
    assert all(
        kw == ({"limit": pipeline.ENRICH_BATCH_LIMIT} if name == "enrich" else {})
        for name, kw in calls[:-1]
    )
    assert stats["apply"] == {"n": 1}


def test_unknown_recipe_raises(conn, cfg):
    with pytest.raises(ValueError, match="unknown recipe"):
        pipeline.run_recipe(conn, cfg, "hourly")
