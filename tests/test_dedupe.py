"""Offline tests for the dedupe passes: all HTTP mocked at gustarr.http.request_json."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from gustarr import config as C
from gustarr import db, http, ids
from gustarr.cli import main as cli_main
from gustarr.dedupe import run

MBID = "53f7a13b-0f75-4b31-9c30-2ed40de6dc35"
OTHER_MBID = "b0b2c1ff-1111-4222-8333-444455556666"
MBID_ITEM = f"artist:mbid:{MBID}"
KANA = "ヨルシカ"
ROMAJI = "Yorushika"
ROMAJI_FID = ids.make("artist", "lastfm", ROMAJI)
KANA_FID = ids.make("artist", "lastfm", KANA)

# An id a pre-normalization gustarr minted verbatim from a full-width
# Last.fm spelling with a stray trailing space. Built by hand because the
# current ids.make can no longer produce it — that is the point.
RAW_OLD_ID = "artist:lastfm:ＹＯＡＳＯＢＩ "
NORM_ID = ids.make("artist", "lastfm", "ＹＯＡＳＯＢＩ ")

ZERO_STATS = {"normalized": 0, "alias_registered": 0, "alias_merged": 0,
              "alias_conflicts": 0, "fetched": 0, "errors": 0}


class FakeApi:
    """Dispatches on URL substring, in registration order (specific first)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, *, params=None, **kw):
        self.calls.append((method, url, dict(params or {})))
        for frag, payload in self.routes:
            if frag in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected request: {url} {params}")


@pytest.fixture(autouse=True)
def _no_politeness(monkeypatch):
    monkeypatch.setattr(http, "HOST_DELAYS", {})


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def make_cfg(tmp_path, **sections):
    return C._build({"core": {"data_dir": str(tmp_path)}, **sections})


def mock_api(monkeypatch, routes):
    api = FakeApi(routes)
    monkeypatch.setattr(http, "request_json", api)
    return api


def event_items(conn):
    return sorted(r["item_id"] for r in conn.execute("SELECT item_id FROM events"))


# ── pass 1: normalize ────────────────────────────────────────────────


def test_normalize_merges_prenorm_item_with_events_intact(conn, tmp_path):
    assert NORM_ID == "artist:lastfm:yoasobi"  # the contract the pass repairs toward
    db.upsert_item(conn, RAW_OLD_ID, "artist", title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", RAW_OLD_ID, "scrobble", 0.15, "lastfm", dedup="t1")
    db.add_event(conn, "2026-01-02T00:00:00Z", RAW_OLD_ID, "scrobble", 0.15, "lastfm", dedup="t2")
    # a normalized twin already exists with one identical synced scrobble
    db.upsert_item(conn, NORM_ID, "artist", title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", NORM_ID, "scrobble", 0.15, "lastfm", dedup="t1")

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "normalized": 1}
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (RAW_OLD_ID,)).fetchone() is None
    # duplicate scrobble collapsed on the uniqueness key, the rest survived
    assert event_items(conn) == [NORM_ID, NORM_ID]
    assert db.canonical_id(conn, RAW_OLD_ID) == NORM_ID
    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_normalize_covers_multipart_track_keys(conn, tmp_path):
    old = f"track:lastfm:ＹＯＡＳＯＢＩ{ids._SEP}夜に駆ける"
    db.upsert_item(conn, old, "track", title="夜に駆ける")

    stats = run(conn, make_cfg(tmp_path))

    new = ids.make("track", "lastfm", "ＹＯＡＳＯＢＩ", "夜に駆ける")
    assert stats["normalized"] == 1
    assert new == f"track:lastfm:yoasobi{ids._SEP}夜に駆ける"
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (new,)).fetchone() is not None
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (old,)).fetchone() is None


def test_normalize_follows_existing_alias_redirect_to_mbid_item(conn, tmp_path):
    """When the normalized form was already merged into an mbid item, the
    pre-norm row must merge straight to the live row, never onto an alias."""
    db.upsert_item(conn, MBID_ITEM, "artist", title="YOASOBI", ids={"mbid": MBID})
    conn.execute("INSERT INTO item_aliases (alias_id, canonical_id) VALUES (?,?)",
                 (NORM_ID, MBID_ITEM))
    db.upsert_item(conn, RAW_OLD_ID, "artist", title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", RAW_OLD_ID, "scrobble", 0.15, "lastfm")

    stats = run(conn, make_cfg(tmp_path))

    assert stats["normalized"] == 1
    assert event_items(conn) == [MBID_ITEM]
    assert db.canonical_id(conn, RAW_OLD_ID) == MBID_ITEM


def test_normalize_skips_names_that_fold_to_nothing(conn, tmp_path):
    stranded = "artist:lastfm:　 "  # ideographic space: NFKC+strip leaves an empty key
    db.upsert_item(conn, stranded, "artist")

    stats = run(conn, make_cfg(tmp_path))

    assert stats == ZERO_STATS  # a stranded row beats a wrong merge
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (stranded,)).fetchone() is not None


# ── pass 2: alias-register ───────────────────────────────────────────


def test_alias_pass_merges_romaji_fallback_into_kana_mbid_artist(conn, tmp_path):
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA,
                   ids={"mbid": MBID}, meta={"aliases": [ROMAJI]})
    db.upsert_item(conn, ROMAJI_FID, "artist", title=ROMAJI)
    db.add_event(conn, "2026-01-01T00:00:00Z", ROMAJI_FID, "scrobble", 0.15, "lastfm", dedup="a")
    db.add_event(conn, "2026-01-02T00:00:00Z", ROMAJI_FID, "scrobble", 0.15, "lastfm", dedup="b")

    stats = run(conn, make_cfg(tmp_path))

    # romaji fallback merged; the kana primary title registered as an alias too
    assert stats == {**ZERO_STATS, "alias_merged": 1, "alias_registered": 1}
    assert conn.execute("SELECT 1 FROM items WHERE id=?", (ROMAJI_FID,)).fetchone() is None
    assert event_items(conn) == [MBID_ITEM, MBID_ITEM]
    assert db.canonical_id(conn, ROMAJI_FID) == MBID_ITEM
    assert db.canonical_id(conn, KANA_FID) == MBID_ITEM

    # the next sync minting the romaji id lands on the mbid item directly
    assert db.add_event(conn, "2026-02-01T00:00:00Z", ROMAJI_FID, "scrobble", 0.15, "lastfm")
    assert event_items(conn) == [MBID_ITEM] * 3

    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_alias_conflict_is_counted_and_first_writer_wins(conn, tmp_path):
    other = f"artist:mbid:{OTHER_MBID}"
    conn.execute("INSERT INTO item_aliases (alias_id, canonical_id) VALUES (?,?)",
                 (ROMAJI_FID, other))
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA,
                   ids={"mbid": MBID}, meta={"aliases": [ROMAJI]})

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "alias_conflicts": 1, "alias_registered": 1}
    assert db.canonical_id(conn, ROMAJI_FID) == other  # mapping untouched


def test_alias_pass_ignores_junk_alias_entries(conn, tmp_path):
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA,
                   ids={"mbid": MBID}, meta={"aliases": ["", None, 7, ROMAJI]})

    stats = run(conn, make_cfg(tmp_path))

    assert stats["alias_registered"] == 2  # kana title + romaji, junk skipped
    assert stats["errors"] == 0


# ── pass 3: fetch ────────────────────────────────────────────────────


def _mb_artist(mbid, name, aliases):
    return {"id": mbid, "name": name,
            "aliases": [{"name": a, "locale": "ja"} for a in aliases] + [{"no": "name"}]}


def test_fetch_stores_aliases_then_merges_and_never_refetches(conn, tmp_path, monkeypatch):
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA, ids={"mbid": MBID})
    db.add_event(conn, "2026-01-01T00:00:00Z", MBID_ITEM, "scrobble", 0.15, "lastfm")
    db.upsert_item(conn, ROMAJI_FID, "artist", title=ROMAJI)
    db.add_event(conn, "2026-01-02T00:00:00Z", ROMAJI_FID, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [(f"/artist/{MBID}", _mb_artist(MBID, KANA, [ROMAJI]))])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == {**ZERO_STATS, "fetched": 1, "alias_merged": 1, "alias_registered": 1}
    assert [c[1] for c in api.calls] == [f"https://musicbrainz.org/ws/2/artist/{MBID}"]
    assert api.calls[0][2] == {"inc": "aliases", "fmt": "json"}
    meta = json.loads(conn.execute(
        "SELECT meta FROM items WHERE id=?", (MBID_ITEM,)).fetchone()["meta"])
    assert meta["aliases"] == [ROMAJI]  # raw spellings; normalization happens at compare time
    assert event_items(conn) == [MBID_ITEM, MBID_ITEM]

    assert run(conn, make_cfg(tmp_path), fetch=True) == ZERO_STATS
    assert len(api.calls) == 1  # aliases stored: no second MB round-trip


def test_fetch_empty_alias_list_still_marks_fetched(conn, tmp_path, monkeypatch):
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA, ids={"mbid": MBID})
    db.add_event(conn, "2026-01-01T00:00:00Z", MBID_ITEM, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [(f"/artist/{MBID}", {"id": MBID, "name": KANA})])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats["fetched"] == 1 and stats["alias_registered"] == 1  # primary title only
    meta = json.loads(conn.execute(
        "SELECT meta FROM items WHERE id=?", (MBID_ITEM,)).fetchone()["meta"])
    assert meta["aliases"] == []  # key presence is the fetched marker

    assert run(conn, make_cfg(tmp_path), fetch=True) == ZERO_STATS
    assert len(api.calls) == 1


def test_fetch_only_touches_played_artists(conn, tmp_path, monkeypatch):
    db.upsert_item(conn, MBID_ITEM, "artist", title=KANA, ids={"mbid": MBID})
    api = mock_api(monkeypatch, [])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == ZERO_STATS
    assert api.calls == []


def test_fetch_error_is_per_item_and_limit_caps_attempts(conn, tmp_path, monkeypatch):
    dead = f"artist:mbid:{OTHER_MBID}"
    for iid, title in ((MBID_ITEM, KANA), (dead, "gone")):
        db.upsert_item(conn, iid, "artist", title=title)
        db.add_event(conn, "2026-01-01T00:00:00Z", iid, "scrobble", 0.15, "lastfm")
    api = mock_api(monkeypatch, [
        (f"/artist/{MBID}", _mb_artist(MBID, KANA, [ROMAJI])),
        (f"/artist/{OTHER_MBID}", http.ApiError(f"x/{OTHER_MBID}", 404, "gone")),
    ])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats["fetched"] == 1 and stats["errors"] == 1  # one dead mbid never aborts the run
    assert len(api.calls) == 2

    # errored artist stays unfetched; limit budgets its retry attempt too
    stats = run(conn, make_cfg(tmp_path), fetch=True, limit=1)
    assert len(api.calls) == 3
    assert stats["errors"] == 1 and stats["fetched"] == 0


# ── cli ──────────────────────────────────────────────────────────────


def test_cli_dedupe_prints_stats_json(tmp_path):
    cfg_path = tmp_path / "gustarr.toml"
    cfg_path.write_text(f'[core]\ndata_dir = "{tmp_path}"\n')
    conn = db.connect(tmp_path / "gustarr.db")
    db.upsert_item(conn, RAW_OLD_ID, "artist", title="YOASOBI")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(cli_main, ["--config", str(cfg_path), "dedupe"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {**ZERO_STATS, "normalized": 1}


def test_cli_dedupe_help_says_when_to_run(tmp_path):
    result = CliRunner().invoke(cli_main, ["dedupe", "--help"])

    assert result.exit_code == 0
    assert "after upgrading" in result.output
    assert "--fetch" in result.output and "--limit" in result.output
