"""Offline tests for the dedupe passes: all HTTP mocked at gustarr.http.request_json."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from gustarr import config as C
from gustarr import db, http
from gustarr.cli import main as cli_main
from gustarr.dedupe import run

MBID = "53f7a13b-0f75-4b31-9c30-2ed40de6dc35"
OTHER_MBID = "b0b2c1ff-1111-4222-8333-444455556666"
KANA = "ヨルシカ"
ROMAJI = "Yorushika"

# A key a pre-normalization gustarr stored verbatim from a full-width
# Last.fm spelling with a stray trailing space. Inserted by hand because
# the v3 write path can no longer produce it — that is the point.
RAW_KEY = "ＹＯＡＳＯＢＩ "

ZERO_STATS = {"normalized": 0, "merged": 0, "alias_attached": 0, "alias_conflicts": 0,
              "fetched": 0, "errors": 0}


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


def mint_raw(conn, domain, ns, key, title=None):
    """An identity written verbatim, bypassing resolve-time normalization —
    the shape only a store predating a normalize_key tightening can hold."""
    ts = db.now()
    cur = conn.execute(
        "INSERT INTO items (domain, title, meta, created_at, updated_at) VALUES (?,?,?,?,?)",
        (domain, title, "{}", ts, ts))
    conn.execute("INSERT INTO identities (domain, ns, key, item_id) VALUES (?,?,?,?)",
                 (domain, ns, key, cur.lastrowid))
    return cur.lastrowid


def event_items(conn):
    return sorted(r["item_id"] for r in conn.execute("SELECT item_id FROM events"))


def identity_keys(conn, domain, ns):
    return sorted(r["key"] for r in conn.execute(
        "SELECT key FROM identities WHERE domain=? AND ns=?", (domain, ns)))


def mbid_keys(conn, item_id):
    return {r["key"] for r in conn.execute(
        "SELECT key FROM identities WHERE item_id=? AND ns='mbid'", (item_id,))}


def gone(conn, item_id):
    return conn.execute("SELECT 1 FROM items WHERE id=?", (item_id,)).fetchone() is None


# ── pass 1: normalize ────────────────────────────────────────────────


def test_normalize_merges_prenorm_twin_with_events_intact(conn, tmp_path):
    raw = mint_raw(conn, "artist", "name", RAW_KEY, title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", raw, "scrobble", 1.0, "lastfm", dedup="t1")
    db.add_event(conn, "2026-01-02T00:00:00Z", raw, "scrobble", 1.0, "lastfm", dedup="t2")
    # a normalized twin already exists with one identical synced scrobble
    norm = db.resolve_item(conn, "artist", "name", "YOASOBI", title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", norm, "scrobble", 1.0, "lastfm", dedup="t1")

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "normalized": 1, "merged": 1}
    # equal authority: the older item survives
    assert gone(conn, norm) and not gone(conn, raw)
    # duplicate scrobble collapsed on the uniqueness key, the rest survived
    assert event_items(conn) == [raw, raw]
    # one identity row remains, normalized, so any spelling lands there
    assert identity_keys(conn, "artist", "name") == ["yoasobi"]
    assert db.lookup_item(conn, "artist", "name", RAW_KEY) == raw
    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_normalize_rewrites_stale_key_without_twin(conn, tmp_path):
    iid = mint_raw(conn, "track", "name", "ＹＯＡＳＯＢＩ 夜に駆ける", title="夜に駆ける")

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "normalized": 1}
    assert identity_keys(conn, "track", "name") == ["yoasobi 夜に駆ける"]
    assert db.lookup_item(conn, "track", "name", "ＹＯＡＳＯＢＩ 夜に駆ける") == iid
    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_normalize_collision_merges_into_authoritative_item(conn, tmp_path):
    """When the normalized spelling already names an mbid item, the
    pre-norm twin must fold into it — external references stay stable."""
    mb_item = db.resolve_item(conn, "artist", "mbid", MBID, title="YOASOBI")
    db.attach_identity(conn, mb_item, "name", "YOASOBI")
    raw = mint_raw(conn, "artist", "name", RAW_KEY, title="YOASOBI")
    db.add_event(conn, "2026-01-01T00:00:00Z", raw, "scrobble", 1.0, "lastfm")

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "normalized": 1, "merged": 1}
    assert gone(conn, raw)
    assert event_items(conn) == [mb_item]
    assert db.lookup_item(conn, "artist", "name", RAW_KEY) == mb_item
    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_normalize_skips_keys_that_fold_to_nothing(conn, tmp_path):
    stranded = mint_raw(conn, "artist", "name", "　 ")  # ideographic space: NFKC leaves nothing

    stats = run(conn, make_cfg(tmp_path))

    assert stats == ZERO_STATS  # a stranded row beats a wrong merge
    assert not gone(conn, stranded)
    assert identity_keys(conn, "artist", "name") == ["　 "]


# ── pass 2: alias-register ───────────────────────────────────────────


def test_alias_pass_merges_romaji_twin_into_kana_mbid_artist(conn, tmp_path):
    kana_item = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA,
                                meta={"aliases": [ROMAJI]})
    romaji_item = db.resolve_item(conn, "artist", "name", ROMAJI, title=ROMAJI)
    db.add_event(conn, "2026-01-01T00:00:00Z", romaji_item, "scrobble", 1.0, "lastfm", dedup="a")
    db.add_event(conn, "2026-01-02T00:00:00Z", romaji_item, "scrobble", 1.0, "lastfm", dedup="b")

    stats = run(conn, make_cfg(tmp_path))

    # romaji twin merged; the kana primary title attached as an alias too
    assert stats == {**ZERO_STATS, "merged": 1, "alias_attached": 1}
    assert gone(conn, romaji_item)
    assert event_items(conn) == [kana_item, kana_item]
    assert db.lookup_item(conn, "artist", "name", ROMAJI) == kana_item
    assert db.lookup_item(conn, "artist", "name", KANA) == kana_item

    # the next sync resolving the romaji spelling lands on the mbid item
    assert db.resolve_item(conn, "artist", "name", ROMAJI) == kana_item
    db.add_event(conn, "2026-02-01T00:00:00Z", kana_item, "scrobble", 1.0, "lastfm")
    assert event_items(conn) == [kana_item] * 3

    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_alias_shared_spelling_between_mbid_artists_is_refused(conn, tmp_path):
    """MB alias lists carry OTHER entities' names — The Kinks list "The
    Ravens", personas list their performer — so a spelling shared by two
    artists that each hold their own mbid proves they are different, not
    the same. The attach is refused and counted; without this rule the
    alias pass once merged Michael Jackson into Mozart on live data."""
    other = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title=ROMAJI)
    db.attach_identity(conn, other, "name", ROMAJI)
    kana_item = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA,
                                meta={"aliases": [ROMAJI]})

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "alias_attached": 1, "alias_conflicts": 1}
    # both artists survive; each keeps its own spelling
    assert not gone(conn, kana_item) and not gone(conn, other)
    assert db.lookup_item(conn, "artist", "name", ROMAJI) == other
    assert db.lookup_item(conn, "artist", "name", KANA) == kana_item
    assert db.identities_of(conn, kana_item)["mbid"] == MBID
    # a standing conflict is still standing on the rerun — counted, not merged
    assert run(conn, make_cfg(tmp_path)) == {**ZERO_STATS, "alias_conflicts": 1}


def test_alias_pass_absorbs_spaceless_scrobble_twin(conn, tmp_path):
    """The 0.5.0 whitespace fold, offline: MB's alias "Kinoko Teikoku"
    absorbs the pre-existing scrobble spelling that dropped the space —
    the same exact string after a deterministic fold, never fuzzy."""
    mb_item = db.resolve_item(conn, "artist", "mbid", MBID, title="きのこ帝国",
                              meta={"aliases": ["Kinoko Teikoku"]})
    twin = db.resolve_item(conn, "artist", "name", "KinokoTeikoku", title="KinokoTeikoku")
    db.add_event(conn, "2026-01-01T00:00:00Z", twin, "scrobble", 1.0, "lastfm", dedup="a")
    db.add_event(conn, "2026-01-02T00:00:00Z", twin, "scrobble", 1.0, "lastfm", dedup="b")

    stats = run(conn, make_cfg(tmp_path))

    # kana title + spaced alias attach; the spaceless twin merges in
    assert stats == {**ZERO_STATS, "alias_attached": 2, "merged": 1}
    assert gone(conn, twin)
    assert event_items(conn) == [mb_item, mb_item]
    # each spelling keeps its own normalize_key'd row — spaceless is a
    # lookup fold, never a storage key
    assert identity_keys(conn, "artist", "name") == \
        ["kinoko teikoku", "kinokoteikoku", "きのこ帝国"]
    assert db.lookup_item(conn, "artist", "name", "KinokoTeikoku") == mb_item
    assert db.lookup_item(conn, "artist", "name", "Kinoko Teikoku") == mb_item
    assert run(conn, make_cfg(tmp_path)) == ZERO_STATS


def test_alias_pass_spaceless_twin_with_own_mbid_is_refused(conn, tmp_path):
    """The cross-entity refusal rule is untouched by the whitespace fold:
    a spaceless twin holding its OWN mbid is a different entity, so the
    hunt counts a conflict and merges nothing."""
    other = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="KinokoTeikoku")
    db.attach_identity(conn, other, "name", "KinokoTeikoku")
    db.add_event(conn, "2026-01-01T00:00:00Z", other, "scrobble", 1.0, "lastfm")
    mb_item = db.resolve_item(conn, "artist", "mbid", MBID, title="きのこ帝国",
                              meta={"aliases": ["Kinoko Teikoku"]})

    stats = run(conn, make_cfg(tmp_path))

    assert stats == {**ZERO_STATS, "alias_attached": 2, "alias_conflicts": 1}
    # both artists survive; the twin keeps its spelling and its history
    assert not gone(conn, mb_item) and not gone(conn, other)
    assert event_items(conn) == [other]
    assert db.lookup_item(conn, "artist", "name", "KinokoTeikoku") == other
    assert db.lookup_item(conn, "artist", "name", "Kinoko Teikoku") == mb_item
    # a standing conflict is still standing on the rerun — counted, not merged
    assert run(conn, make_cfg(tmp_path)) == {**ZERO_STATS, "alias_conflicts": 1}


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


FOLD_SCAN = "SELECT key FROM identities WHERE domain='artist' AND ns='name' ORDER BY rowid"


def test_alias_pass_scans_identities_once_not_per_alias(conn, tmp_path):
    """The whitespace-fold twin hunt runs off ONE fold index built at the
    top of the pass — an O(1) dict lookup per alias, never a fresh
    identities scan each (that was quadratic in store size)."""
    for i in range(3):
        db.resolve_item(
            conn, "artist", "mbid", f"{i}{MBID[1:]}", title=f"Artist {i}",
            meta={"aliases": [f"Artist {i} Alias {j}" for j in range(4)]})
    first_log = QueryLog(conn)
    first = run(first_log, make_cfg(tmp_path))
    assert first == {**ZERO_STATS, "alias_attached": 15}  # 3 titles + 12 aliases

    rerun_log = QueryLog(conn)
    stats = run(rerun_log, make_cfg(tmp_path))

    assert stats == ZERO_STATS  # rerun on the settled store: same stats, no work
    for log in (first_log, rerun_log):
        scans = [q for q in log.queries if q == FOLD_SCAN]
        assert len(scans) == 1  # once per pass, not once per alias hunted


def test_alias_pass_ignores_junk_alias_entries(conn, tmp_path):
    db.resolve_item(conn, "artist", "mbid", MBID, title=KANA,
                    meta={"aliases": ["", None, 7, ROMAJI]})

    stats = run(conn, make_cfg(tmp_path))

    assert stats["alias_attached"] == 2  # kana title + romaji, junk skipped
    assert stats["errors"] == 0


# ── pass 3: fetch ────────────────────────────────────────────────────


def _mb_artist(mbid, name, aliases):
    return {"id": mbid, "name": name,
            "aliases": [{"name": a, "locale": "ja"} for a in aliases] + [{"no": "name"}]}


def test_fetch_stores_aliases_then_merges_and_never_refetches(conn, tmp_path, monkeypatch):
    kana_item = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)
    db.add_event(conn, "2026-01-01T00:00:00Z", kana_item, "scrobble", 1.0, "lastfm")
    romaji_item = db.resolve_item(conn, "artist", "name", ROMAJI, title=ROMAJI)
    db.add_event(conn, "2026-01-02T00:00:00Z", romaji_item, "scrobble", 1.0, "lastfm")
    api = mock_api(monkeypatch, [(f"/artist/{MBID}", _mb_artist(MBID, KANA, [ROMAJI]))])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == {**ZERO_STATS, "fetched": 1, "merged": 1, "alias_attached": 1}
    assert [c[1] for c in api.calls] == [f"https://musicbrainz.org/ws/2/artist/{MBID}"]
    assert api.calls[0][2] == {"inc": "aliases", "fmt": "json"}
    meta = json.loads(conn.execute(
        "SELECT meta FROM items WHERE id=?", (kana_item,)).fetchone()["meta"])
    assert meta["aliases"] == [ROMAJI]  # raw spellings; normalization happens at write time
    assert event_items(conn) == [kana_item, kana_item]

    assert run(conn, make_cfg(tmp_path), fetch=True) == ZERO_STATS
    assert len(api.calls) == 1  # aliases stored: no second MB round-trip


def test_fetch_empty_alias_list_still_marks_fetched(conn, tmp_path, monkeypatch):
    kana_item = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)
    db.add_event(conn, "2026-01-01T00:00:00Z", kana_item, "scrobble", 1.0, "lastfm")
    api = mock_api(monkeypatch, [(f"/artist/{MBID}", {"id": MBID, "name": KANA})])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == {**ZERO_STATS, "fetched": 1, "alias_attached": 1}  # primary title only
    meta = json.loads(conn.execute(
        "SELECT meta FROM items WHERE id=?", (kana_item,)).fetchone()["meta"])
    assert meta["aliases"] == []  # key presence is the fetched marker

    assert run(conn, make_cfg(tmp_path), fetch=True) == ZERO_STATS
    assert len(api.calls) == 1


def test_fetch_only_touches_played_artists(conn, tmp_path, monkeypatch):
    db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)
    api = mock_api(monkeypatch, [])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == ZERO_STATS
    assert api.calls == []


def test_fetch_skips_name_only_artists(conn, tmp_path, monkeypatch):
    """Played but mbid-less artists wait for enrich — there is no MB url
    to fetch aliases from yet."""
    iid = db.resolve_item(conn, "artist", "name", ROMAJI, title=ROMAJI)
    db.add_event(conn, "2026-01-01T00:00:00Z", iid, "scrobble", 1.0, "lastfm")
    api = mock_api(monkeypatch, [])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    assert stats == ZERO_STATS
    assert api.calls == []


def test_fetch_error_is_per_item_and_limit_caps_attempts(conn, tmp_path, monkeypatch):
    good = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)
    dead = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="gone")
    for iid in (good, dead):
        db.add_event(conn, "2026-01-01T00:00:00Z", iid, "scrobble", 1.0, "lastfm")
    api = mock_api(monkeypatch, [
        (f"/artist/{MBID}", _mb_artist(MBID, KANA, [ROMAJI])),
        (f"/artist/{OTHER_MBID}", http.ApiError(f"x/{OTHER_MBID}", 404, "gone")),
    ])

    stats = run(conn, make_cfg(tmp_path), fetch=True)

    # one dead mbid never aborts the run
    assert stats == {**ZERO_STATS, "fetched": 1, "errors": 1, "alias_attached": 2}
    assert len(api.calls) == 2

    # errored artist stays unfetched; limit budgets its retry attempt too
    stats = run(conn, make_cfg(tmp_path), fetch=True, limit=1)
    assert len(api.calls) == 3
    assert stats == {**ZERO_STATS, "errors": 1}


# ── cli ──────────────────────────────────────────────────────────────


def test_cli_dedupe_prints_stats_json(tmp_path):
    cfg_path = tmp_path / "gustarr.toml"
    cfg_path.write_text(f'[core]\ndata_dir = "{tmp_path}"\n')
    conn = db.connect(tmp_path / "gustarr.db")
    mint_raw(conn, "artist", "name", RAW_KEY, title="YOASOBI")
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


# ── attach_identity merge/refuse rule (db-level pins) ─────────────────


def test_attach_name_between_two_authoritative_items_is_refused(conn):
    """The db-level backstop for the alias-conflict rule: a name collision
    between two items that each hold their own authoritative id must not
    merge them, whatever the caller."""
    a = db.resolve_item(conn, "artist", "mbid", MBID, title="The Kinks")
    b = db.resolve_item(conn, "artist", "mbid", OTHER_MBID, title="The Ravens")
    db.attach_identity(conn, b, "name", "The Ravens")

    assert db.attach_identity(conn, a, "name", "The Ravens") == a

    assert not gone(conn, a) and not gone(conn, b)
    assert db.lookup_item(conn, "artist", "name", "The Ravens") == b


def test_attach_name_absorbs_name_only_twin(conn):
    """The healing case stays: a name-only twin (no authoritative id) is
    the same entity by construction and merges into the mbid holder."""
    twin = db.resolve_item(conn, "artist", "name", ROMAJI, title=ROMAJI)
    db.add_event(conn, "2026-01-01T00:00:00Z", twin, "scrobble", 1.0, "lastfm")
    a = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)

    assert db.attach_identity(conn, a, "name", ROMAJI) == a

    assert gone(conn, twin)
    assert event_items(conn) == [a]
    assert db.lookup_item(conn, "artist", "name", ROMAJI) == a


def test_attach_authoritative_key_still_merges(conn):
    """An authoritative-key collision is same-entity proof and merges even
    when both sides already hold ids: a movie known by imdb on one row
    learning its tmdb id folds into the row that owns that tmdb id."""
    by_tmdb = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    by_imdb = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix")

    winner = db.attach_identity(conn, by_imdb, "tmdb", "603")

    assert winner == by_tmdb
    assert gone(conn, by_imdb)
    assert db.identities_of(conn, by_tmdb)["imdb"] == "tt0133093"


def add_rec(conn, item_id, status, run_id):
    conn.execute(
        "INSERT INTO recommendations (profile, run_id, ts, domain, item_id, score, status)"
        " VALUES ('default', ?, ?, 'movie', ?, 0.5, ?)",
        (run_id, db.now(), item_id, status))


def recs(conn, item_id):
    return [(r["status"], r["run_id"]) for r in conn.execute(
        "SELECT status, run_id FROM recommendations WHERE item_id=? ORDER BY id", (item_id,))]


def item_year(conn, item_id):
    return conn.execute("SELECT year FROM items WHERE id=?", (item_id,)).fetchone()["year"]


def test_empty_key_is_refused_at_both_write_paths(conn):
    """A key that folds to nothing must raise, never resolve: a shared ''
    identity row would fuse every blank-named item into one."""
    iid = db.resolve_item(conn, "artist", "mbid", MBID, title=KANA)
    for key in ("", "  ", "　"):  # empty, spaces, ideographic space
        with pytest.raises(ValueError):
            db.resolve_item(conn, "artist", "name", key)
        with pytest.raises(ValueError):
            db.attach_identity(conn, iid, "name", key)


def test_attach_jellyfin_between_authoritative_items_moves_the_pointer(conn):
    """A jellyfin key is a library pointer, not an immutable identity: when
    the entry is retagged from movie A to movie B (each holding its own
    tmdb id), the pointer MOVES — no merge, A survives, and the events it
    earned under the old identification stay put."""
    a = db.resolve_item(conn, "movie", "tmdb", "1", title="A")
    db.attach_identity(conn, a, "jellyfin", "jf-1")
    db.add_event(conn, "2026-01-01T00:00:00Z", a, "play", 1.0, "jellyfin")
    b = db.resolve_item(conn, "movie", "tmdb", "2", title="B")

    assert db.attach_identity(conn, b, "jellyfin", "jf-1") == b

    assert db.lookup_item(conn, "movie", "jellyfin", "jf-1") == b
    assert not gone(conn, a) and not gone(conn, b)
    assert event_items(conn) == [a]
    assert db.identities_of(conn, a)["tmdb"] == "1"


def test_attach_jellyfin_still_merges_pending_item(conn):
    """The enrichment path stays a merge: a jellyfin-only item (no
    authoritative id yet) holding the key IS the same entry, so the tmdb
    holder attaching that key absorbs it, events included."""
    a = db.resolve_item(conn, "movie", "jellyfin", "jf-1", title="A")
    db.add_event(conn, "2026-01-01T00:00:00Z", a, "play", 1.0, "jellyfin")
    b = db.resolve_item(conn, "movie", "tmdb", "2", title="B")

    assert db.attach_identity(conn, b, "jellyfin", "jf-1") == b

    assert gone(conn, a)
    assert event_items(conn) == [b]
    assert db.lookup_item(conn, "movie", "jellyfin", "jf-1") == b


def test_merge_keeps_losers_approved_rec_over_winners_proposal(conn):
    """When a merge folds two open recommendations onto one item, the
    user's verdict outranks the standing proposal: the winner's proposed
    row must yield so the loser's approved one is not IGNOREd away."""
    winner = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    loser = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix")
    add_rec(conn, winner, "proposed", "run-w")
    add_rec(conn, loser, "approved", "run-l")

    db.merge_items(conn, loser, winner)

    assert recs(conn, winner) == [("approved", "run-l")]
    assert recs(conn, loser) == []


def test_merge_keeps_winners_approved_rec_over_losers_proposal(conn):
    """Same rule, verdict on the other side: the winner's approved row
    stands and the loser's proposal is the one that dies."""
    winner = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    loser = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix")
    add_rec(conn, winner, "approved", "run-w")
    add_rec(conn, loser, "proposed", "run-l")

    db.merge_items(conn, loser, winner)

    assert recs(conn, winner) == [("approved", "run-w")]
    assert recs(conn, loser) == []


def test_merge_keeps_winners_year(conn):
    winner = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix", year=1999)
    loser = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix", year=2001)

    db.merge_items(conn, loser, winner)

    assert item_year(conn, winner) == 1999


def test_merge_fills_winners_null_year_from_loser(conn):
    winner = db.resolve_item(conn, "movie", "tmdb", "603", title="The Matrix")
    loser = db.resolve_item(conn, "movie", "imdb", "tt0133093", title="The Matrix", year=2001)

    db.merge_items(conn, loser, winner)

    assert item_year(conn, winner) == 2001
