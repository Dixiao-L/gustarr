"""Offline tests for `gustarr identify`: the pending list, the
MusicBrainz search proxy, and the human mbid assertion."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from gustarr import db, http, identify
from gustarr.cli import main as cli_main

MBID = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
OTHER_MBID = "0b8a3e2b-6f1d-4c58-9e2a-7f0f2b1c9d44"

MB_SEARCH = {"artists": [
    {"id": MBID, "name": "Kinoko Teikoku", "score": 100, "type": "Group", "country": "JP"},
    {"id": OTHER_MBID, "name": "Kinoko", "score": 55,
     "disambiguation": "solo project"},
    {"name": "idless noise entry"},  # malformed hit: no id, must be dropped
]}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def listen(conn, item_id, ts, profile="default"):
    db.add_event(conn, ts, item_id, "scrobble", 1.0, "lastfm", profile=profile)


def mock_mb(monkeypatch, payload=MB_SEARCH):
    calls = []

    def fake(method, url, *, params=None, **kw):
        calls.append((method, url, dict(params or {})))
        return payload

    monkeypatch.setattr(http, "request_json", fake)
    return calls


# ── pending ──────────────────────────────────────────────────────────


def test_pending_lists_mbidless_artists_ranked_by_history(conn):
    quiet = db.resolve_item(conn, "artist", "name", "Quiet One", title="Quiet One")
    loud = db.resolve_item(conn, "artist", "name", "Loud One", title="Loud One")
    identified = db.resolve_item(conn, "artist", "mbid", MBID, title="Radiohead")
    movie = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    listen(conn, quiet, "2026-01-01T00:00:00Z")
    for day in (1, 2, 3):
        listen(conn, loud, f"2026-01-0{day}T00:00:00Z")
    listen(conn, identified, "2026-01-04T00:00:00Z")

    rows = identify.pending(conn)

    # mbid holders and non-artists never show, however much history they hold
    assert [r["item_id"] for r in rows] == [loud, quiet]
    assert movie not in [r["item_id"] for r in rows]
    assert rows[0] == {"item_id": loud, "title": "Loud One",
                       "spellings": ["loud one"], "events": 3}
    assert rows[1]["events"] == 1


def test_pending_counts_history_across_all_profiles(conn):
    # identity is global — an assert changes the shared store, so the
    # ranking weighs everyone's listening, not one profile's
    artist = db.resolve_item(conn, "artist", "name", "Shared", title="Shared")
    listen(conn, artist, "2026-01-01T00:00:00Z", profile="alice")
    listen(conn, artist, "2026-01-02T00:00:00Z", profile="bob")

    assert identify.pending(conn)[0]["events"] == 2


def test_pending_lists_every_spelling(conn):
    artist = db.resolve_item(conn, "artist", "name", "KinokoTeikoku",
                             title="KinokoTeikoku")
    db.attach_identity(conn, artist, "name", "きのこ帝国")

    (row,) = identify.pending(conn)
    # all name identities, not identities_of's first-per-ns — the human
    # matches on the odd romanizations
    assert row["spellings"] == ["kinokoteikoku", "きのこ帝国"]


def test_pending_respects_limit(conn):
    for i in range(5):
        db.resolve_item(conn, "artist", "name", f"A{i}", title=f"A{i}")
    assert len(identify.pending(conn, limit=3)) == 3
    assert len(identify.pending(conn)) == 5


class QueryLog:
    """conn wrapper recording every SQL statement (whitespace-collapsed)."""

    def __init__(self, conn):
        self._conn = conn
        self.queries = []

    def execute(self, sql, *args, **kw):
        self.queries.append(" ".join(sql.split()))
        return self._conn.execute(sql, *args, **kw)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_pending_counts_events_in_one_aggregated_query(conn):
    """The ranking runs as ONE LEFT JOIN + GROUP BY statement — the old
    correlated (SELECT COUNT(*) FROM events ...) subquery re-scanned
    events once per artist row in the store."""
    for i in range(3):
        artist = db.resolve_item(conn, "artist", "name", f"A{i}", title=f"A{i}")
        listen(conn, artist, f"2026-01-0{i + 1}T00:00:00Z")

    log = QueryLog(conn)
    rows = identify.pending(log)

    assert [r["events"] for r in rows] == [1, 1, 1]  # LEFT JOIN counts stay right
    (ranking,) = [q for q in log.queries if "FROM items" in q]
    assert "LEFT JOIN events" in ranking and "GROUP BY" in ranking
    assert "SELECT COUNT(*) FROM events" not in ranking  # no per-artist subquery
    # EXPLAIN shape: events participates as a join scanned by the outer
    # loop — never inside a subquery node SQLite re-runs per artist row
    # (the cheap NOT EXISTS mbid filter over identities may stay one)
    plan = conn.execute("EXPLAIN QUERY PLAN " + ranking, (25,)).fetchall()
    subquery_nodes = {r[0] for r in plan if "SUBQUERY" in str(r[3]).upper()}
    events_scans = [r for r in plan if str(r[3]).startswith(("SCAN e", "SEARCH e"))]
    assert events_scans
    assert all(r[1] not in subquery_nodes for r in events_scans)


# ── search ───────────────────────────────────────────────────────────


def test_search_maps_musicbrainz_hits(monkeypatch):
    calls = mock_mb(monkeypatch)

    hits = identify.search('Kinoko "Teikoku"')

    method, url, params = calls[0]
    assert method == "GET"
    assert url == "https://musicbrainz.org/ws/2/artist"
    # quotes are stripped so the name can't break out of the phrase
    # query — the same pattern as enrich's automated search
    assert params["query"] == 'artist:"Kinoko Teikoku"'
    assert params["fmt"] == "json"
    assert [h["id"] for h in hits] == [MBID, OTHER_MBID]  # the idless hit is dropped
    assert hits[0]["name"] == "Kinoko Teikoku"
    assert hits[0]["score"] == 100
    assert (hits[0]["type"], hits[0]["country"]) == ("Group", "JP")
    assert hits[0]["disambiguation"] == ""  # empty string, never None
    assert hits[1]["disambiguation"] == "solo project"


def test_search_tolerates_empty_payload(monkeypatch):
    mock_mb(monkeypatch, payload=None)
    assert identify.search("Nobody") == []


# ── assert ───────────────────────────────────────────────────────────


def test_assert_attaches_and_reopens_enrichment(conn):
    artist = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    db.upsert_item_fields(conn, artist, enriched=True)  # enriched matchless last run
    listen(conn, artist, "2026-01-01T00:00:00Z")

    assert identify.assert_mbid(conn, artist, MBID) == artist

    assert db.lookup_item(conn, "artist", "mbid", MBID) == artist
    # the next enrich must re-fetch: MB's canonical name, tags and the
    # alias list that bridges future spellings
    row = conn.execute("SELECT enriched_at FROM items WHERE id=?", (artist,)).fetchone()
    assert row["enriched_at"] is None
    assert identify.pending(conn) == []  # identified: off the human's list


def test_assert_merges_split_history_into_mbid_holder(conn):
    # the headline use: a scrobble spelling holding real history merges
    # into the properly-identified artist under the existing rules
    twin = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    listen(conn, twin, "2026-01-01T00:00:00Z")
    real = db.resolve_item(conn, "artist", "mbid", MBID, title="Kinoko Teikoku")
    db.upsert_item_fields(conn, real, enriched=True)

    assert identify.assert_mbid(conn, twin, MBID) == real

    assert conn.execute("SELECT 1 FROM items WHERE id=?", (twin,)).fetchone() is None
    assert [r["item_id"] for r in conn.execute("SELECT item_id FROM events")] == [real]
    # the spelling lands on the survivor on arrival from now on
    assert db.lookup_item(conn, "artist", "name", "KinokoTeikoku") == real
    # the survivor re-enriches too — the assert changed what MB would say
    assert conn.execute("SELECT enriched_at FROM items WHERE id=?",
                        (real,)).fetchone()["enriched_at"] is None


def test_assert_refuses_artist_that_already_has_an_mbid(conn):
    # the cross-entity refusal rule, held on the human path too: two
    # artists each holding their own authoritative id are different
    # entities, and a typo'd assert must never force-merge them
    real = db.resolve_item(conn, "artist", "mbid", MBID, title="Radiohead")
    db.upsert_item_fields(conn, real, enriched=True)
    other = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="Yorushika")
    listen(conn, other, "2026-01-01T00:00:00Z")

    with pytest.raises(ValueError, match="already has MusicBrainz id"):
        identify.assert_mbid(conn, real, OTHER_MBID)

    # refused loudly AND nothing written: both artists intact, history
    # unmoved, enrichment not reopened
    assert db.lookup_item(conn, "artist", "mbid", MBID) == real
    assert db.lookup_item(conn, "artist", "mbid", OTHER_MBID) == other
    assert [r["item_id"] for r in conn.execute("SELECT item_id FROM events")] == [other]
    assert conn.execute("SELECT enriched_at FROM items WHERE id=?",
                        (real,)).fetchone()["enriched_at"] is not None


def test_assert_refuses_unknown_and_nonartist_items(conn):
    with pytest.raises(ValueError, match="no item #999"):
        identify.assert_mbid(conn, 999, MBID)
    movie = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    with pytest.raises(ValueError, match="movie, not an artist"):
        identify.assert_mbid(conn, movie, MBID)


def test_assert_refuses_malformed_mbid(conn):
    # 'banana', a pasted artist name, and a truncated uuid are typos to
    # refuse loudly — an mbid is a UUID (8-4-4-4-12 hex), and a garbage
    # identity row would poison merges forever
    artist = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    for bad in ("banana", "Kinoko Teikoku", MBID[:-4]):
        with pytest.raises(ValueError, match="not a MusicBrainz id"):
            identify.assert_mbid(conn, artist, bad)
    # refused loudly AND nothing written
    assert db.identities_of(conn, artist) == {"name": "yorushika"}


def test_assert_accepts_uppercase_mbid(conn):
    # the UUID shape check is case-insensitive; normalize_key lowercases
    # the stored key as for every identity
    artist = db.resolve_item(conn, "artist", "name", "Yorushika", title="Yorushika")
    assert identify.assert_mbid(conn, artist, MBID.upper()) == artist
    assert db.lookup_item(conn, "artist", "mbid", MBID) == artist


# ── cli ──────────────────────────────────────────────────────────────


def cli_store(tmp_path):
    cfg_path = tmp_path / "gustarr.toml"
    cfg_path.write_text(f'[core]\ndata_dir = "{tmp_path}"\n')
    return cfg_path, db.connect(tmp_path / "gustarr.db")


def test_cli_identify_bare_lists_pending(tmp_path):
    cfg_path, conn = cli_store(tmp_path)
    artist = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    listen(conn, artist, "2026-01-01T00:00:00Z")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(cli_main, ["--config", str(cfg_path), "identify"])

    assert result.exit_code == 0, result.output
    assert f"#{artist}" in result.output
    assert "KinokoTeikoku" in result.output
    assert "1 events" in result.output


def test_cli_identify_name_searches_musicbrainz(tmp_path, monkeypatch):
    cfg_path, conn = cli_store(tmp_path)
    conn.close()
    mock_mb(monkeypatch)

    result = CliRunner().invoke(
        cli_main, ["--config", str(cfg_path), "identify", "Kinoko Teikoku"])

    assert result.exit_code == 0, result.output
    assert MBID in result.output
    assert "Kinoko Teikoku" in result.output
    assert "solo project" in result.output  # disambiguation shown when MB has one


def test_cli_identify_asserts_mbid(tmp_path):
    cfg_path, conn = cli_store(tmp_path)
    artist = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli_main, ["--config", str(cfg_path), "identify", "KinokoTeikoku", "--mbid", MBID])

    assert result.exit_code == 0, result.output
    assert MBID in result.output
    conn = db.connect(tmp_path / "gustarr.db")
    try:
        assert db.lookup_item(conn, "artist", "mbid", MBID) == artist  # committed
    finally:
        conn.close()


def test_cli_identify_unknown_name_errors_clearly(tmp_path):
    cfg_path, conn = cli_store(tmp_path)
    conn.close()

    result = CliRunner().invoke(
        cli_main, ["--config", str(cfg_path), "identify", "Nobody", "--mbid", MBID])

    assert result.exit_code != 0
    assert "no artist named 'Nobody'" in result.output


def test_cli_identify_refusal_is_a_clean_error(tmp_path):
    # the cross-entity refusal surfaces as a message, not a traceback
    cfg_path, conn = cli_store(tmp_path)
    real = db.resolve_item(conn, "artist", "mbid", MBID, title="Radiohead")
    db.attach_identity(conn, real, "name", "Radiohead")  # so NAME resolves to it
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli_main, ["--config", str(cfg_path), "identify", "Radiohead", "--mbid", OTHER_MBID])

    assert result.exit_code != 0
    assert "already has MusicBrainz id" in result.output
    assert "Traceback" not in result.output


def test_cli_identify_malformed_mbid_is_a_clean_error(tmp_path):
    cfg_path, conn = cli_store(tmp_path)
    db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli_main,
        ["--config", str(cfg_path), "identify", "KinokoTeikoku", "--mbid", "banana"])

    assert result.exit_code != 0
    assert "not a MusicBrainz id" in result.output
    assert "Traceback" not in result.output


def test_cli_identify_mbid_without_name_is_a_usage_error(tmp_path):
    cfg_path, conn = cli_store(tmp_path)
    conn.close()

    result = CliRunner().invoke(
        cli_main, ["--config", str(cfg_path), "identify", "--mbid", MBID])

    assert result.exit_code != 0
    assert "name the artist" in result.output
