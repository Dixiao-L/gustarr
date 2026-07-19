"""Duplicate repair: one artist, one item, whatever the spelling.

The same CJK artist arrives romanized from one source and in kana/kanji
from another; each spelling used to mint its own item, splitting one
person's listening history across twins. In the v3 identity model every
spelling is a row in ``identities``, so repair means repairing that
table. Three idempotent passes:

  1. normalize — re-normalize every identity key with the current
     ids.normalize_key (NFKC + casefold + whitespace collapse). Keys that
     collide once normalized name the same entity, so their items merge
     first (authoritative holder wins, mirroring db.attach_identity);
     then the stale spellings are rewritten in place. Heals stores whose
     keys predate a normalization tightening.
  2. alias-register — a MusicBrainz artist's alias names are exactly the
     spellings collectors resolve name identities from; attaching each
     via db.attach_identity folds existing twins into the mbid item and
     makes all future encounters land on it on arrival. Cross-script
     variants (kana/kanji vs romaji) collapse here, and so do spaceless
     spelling twins ("KinokoTeikoku" for the alias "Kinoko Teikoku"),
     hunted by a deterministic whitespace fold — never fuzzy. The fold's
     honest cost: two genuinely distinct name-only artists whose
     spellings differ only by whitespace lose that distinction to it —
     gustarr identify is the recovery path for the mbid side, and a
     wrongly absorbed name-only twin has no automated undo.
  3. fetch (opt-in) — pull alias lists from MusicBrainz for played
     artists that lack them, then alias-register those artists.

Not a pipeline stage: run `gustarr dedupe` after upgrading gustarr or
after importing history from a new source. Failures are per-item.
"""

from __future__ import annotations

import json
import sqlite3

from . import db, http, ids
from .config import Config

MB = "https://musicbrainz.org/ws/2"


def run(
    conn: sqlite3.Connection,
    cfg: Config,  # unused today; kept so every stage shares the run(conn, cfg, ...) shape
    fetch: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    stats = {"normalized": 0, "merged": 0, "alias_attached": 0, "alias_conflicts": 0,
             "fetched": 0, "errors": 0}
    _normalize_pass(conn, stats)
    _alias_pass(conn, stats)
    if fetch:
        _fetch_pass(conn, stats, limit)
    return stats


def _authority(conn: sqlite3.Connection, item_id: int) -> int:
    # the same rule db.attach_identity merges by: external-id holders
    # beat name/jellyfin-only items, so references stay stable
    return conn.execute(
        "SELECT COUNT(*) FROM identities WHERE item_id=? AND ns != 'name'"
        " AND ns != 'jellyfin'", (item_id,)).fetchone()[0]


def _normalize_pass(conn: sqlite3.Connection, stats: dict[str, int]) -> None:
    # fetchall up front: merges rewrite rows under the cursor otherwise
    rows = conn.execute("SELECT domain, ns, key, item_id FROM identities").fetchall()
    groups: dict[tuple[str, str, str], list[str]] = {}
    for r in rows:
        norm = ids.normalize_key(r["key"])
        if not norm:
            continue  # name folds to nothing; a stranded row beats a wrong merge
        groups.setdefault((r["domain"], r["ns"], norm), []).append(r["key"])

    # Colliding keys mean same-entity: two spellings of one name (or one
    # id) that only normalization can tell apart. Merge before rewriting
    # so the rewrite never has to move events itself.
    for (domain, ns, _norm), keys in sorted(groups.items()):
        if len(keys) < 2:
            continue
        marks = ",".join("?" * len(keys))
        # re-query owners live: an earlier group's merge may have already
        # repointed some of these rows
        members = sorted({r["item_id"] for r in conn.execute(
            f"SELECT item_id FROM identities WHERE domain=? AND ns=? AND key IN ({marks})",
            (domain, ns, *keys))})
        if len(members) < 2:
            continue
        if ns == "name":
            # A shared spelling only proves sameness when at most one
            # side has an authoritative identity; two mbid holders whose
            # names fold together are different entities (see
            # attach_identity), and with several strong members there is
            # no defensible owner for the weak ones either — skip.
            strong = [m for m in members if _authority(conn, m) > 0]
            if len(strong) > 1:
                stats["alias_conflicts"] += 1
                continue
        # ties go to the oldest item, approximating first-writer-wins
        winner = max(members, key=lambda i: (_authority(conn, i), -i))
        for loser in members:
            if loser != winner:
                db.merge_items(conn, loser, winner)
                stats["merged"] += 1

    for r in rows:
        norm = ids.normalize_key(r["key"])
        if not norm or norm == r["key"]:
            continue
        cur = conn.execute(
            "UPDATE OR IGNORE identities SET key=? WHERE domain=? AND ns=? AND key=?",
            (norm, r["domain"], r["ns"], r["key"]))
        if cur.rowcount == 0:
            # the normalized row already exists — the stale spelling is
            # redundant only when the same item owns both; a skipped
            # conflict group above keeps its raw keys (deleting one would
            # strip a different item's identity)
            conn.execute(
                "DELETE FROM identities WHERE domain=? AND ns=? AND key=? AND item_id ="
                " (SELECT item_id FROM identities WHERE domain=? AND ns=? AND key=?)",
                (r["domain"], r["ns"], r["key"], r["domain"], r["ns"], norm))
        stats["normalized"] += 1


def _alias_pass(conn: sqlite3.Connection, stats: dict[str, int]) -> None:
    rows = conn.execute(
        "SELECT id, title, meta FROM items WHERE domain='artist' AND EXISTS("
        " SELECT 1 FROM identities x WHERE x.item_id = items.id AND x.ns='mbid')").fetchall()
    fold = _fold_index(conn)
    for row in rows:
        if conn.execute("SELECT 1 FROM items WHERE id=?", (row["id"],)).fetchone() is None:
            continue  # merged away by an earlier row's registration
        meta = json.loads(row["meta"])
        # key-present is the marker, so an empty list (artist fetched,
        # MB knows no aliases) still registers the primary title and is
        # never re-fetched.
        if "aliases" in meta:
            _register(conn, row["id"], row["title"], meta["aliases"], stats, fold)


def _register(conn: sqlite3.Connection, item_id: int, title: str | None,
              aliases: list, stats: dict[str, int], fold: dict[str, list[str]]) -> int:
    """attach_identity per spelling, with the dedupe bookkeeping: a
    spelling held by a name-only twin merges the two (the authoritative
    holder survives and registration continues on it); one held by
    another artist with its own mbid is refused and counted — MB alias
    lists carry other entities' names, so that collision proves the two
    are different. Each spelling is also hunted for spaceless twins —
    the scrobble spelling that dropped the alias's spaces — whose stored
    keys go through the very same attach, so the merge/refuse rule is
    untouched."""
    for name in [title, *aliases]:
        # aliases are stored raw; attach_identity normalizes on write, so
        # this is exactly the identity a collector would resolve for that
        # spelling. Names folding to nothing can never be looked up.
        if not name or not isinstance(name, str) or not ids.normalize_key(name):
            continue
        item_id = _register_one(conn, item_id, name, stats, fold)
        for twin in _spaceless_twins(fold, name):
            item_id = _register_one(conn, item_id, twin, stats, fold)
    return item_id


def _register_one(conn: sqlite3.Connection, item_id: int, name: str,
                  stats: dict[str, int], fold: dict[str, list[str]]) -> int:
    holder = db.lookup_item(conn, "artist", "name", name)
    if holder == item_id:
        return item_id  # already registered: reruns stay silent
    item_id = db.attach_identity(conn, item_id, "name", name)
    _fold_add(fold, name)  # newly attached spellings join the huntable set
    if holder is None:
        stats["alias_attached"] += 1
    elif item_id != holder and db.lookup_item(conn, "artist", "name", name) == holder:
        stats["alias_conflicts"] += 1
    else:
        stats["merged"] += 1
    return item_id


def _fold_index(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """The whitespace-fold lookup index behind alias twin-hunting:
    {ids.spaceless(key): [stored keys]} over every artist 'name'
    identity, built ONCE per pass so each hunt is an O(1) lookup instead
    of a fresh table scan per alias. Keys only — item ownership is
    resolved live at attach time, so an earlier merge can never stale a
    hunt. SQL's REPLACE(key, ' ', '') cannot reproduce ids.spaceless (a
    store predating a normalize_key tightening holds full-width and
    ideographic spaces REPLACE can't see), so the fold runs in python."""
    fold: dict[str, list[str]] = {}
    for r in conn.execute(
            "SELECT key FROM identities WHERE domain='artist' AND ns='name' ORDER BY rowid"):
        fold.setdefault(ids.spaceless(r["key"]), []).append(r["key"])
    return fold


def _fold_add(fold: dict[str, list[str]], name: str) -> None:
    """Register a just-attached spelling in the fold index (stored form:
    attach_identity writes normalize_key'd keys) so later aliases in the
    same pass can hunt it too."""
    norm = ids.normalize_key(name)
    keys = fold.setdefault(ids.spaceless(norm), [])
    if norm not in keys:
        keys.append(norm)


def _spaceless_twins(fold: dict[str, list[str]], alias: str) -> list[str]:
    """Existing artist 'name' spellings that equal this alias once spaces
    are folded out but differ as stored keys — the scrobble-spelling twins
    ("KinokoTeikoku" for MB's "Kinoko Teikoku"). Still an exact match
    after a deterministic fold, never fuzzy: one dict lookup in the
    pass-wide fold index."""
    norm = ids.normalize_key(alias)
    return [k for k in fold.get(ids.spaceless(alias), []) if k != norm]


def _fetch_pass(conn: sqlite3.Connection, stats: dict[str, int], limit: int | None) -> None:
    rows = conn.execute(
        "SELECT id, title, meta,"
        # first mbid by rowid — the same one identities_of would report
        " (SELECT key FROM identities x WHERE x.item_id = items.id AND x.ns='mbid'"
        "  ORDER BY x.rowid LIMIT 1) AS mbid"
        " FROM items WHERE domain='artist'"
        # played artists only: each lookup costs 1.1s of MB politeness,
        # and only items with history have split history worth healing
        " AND EXISTS(SELECT 1 FROM events e WHERE e.item_id = items.id)").fetchall()
    fold = _fold_index(conn)
    attempts = 0
    for row in rows:
        if row["mbid"] is None:
            continue  # name-only artists wait until enrich reveals an mbid
        if conn.execute("SELECT 1 FROM items WHERE id=?", (row["id"],)).fetchone() is None:
            continue  # merged away by an earlier row's registration
        meta = json.loads(row["meta"])
        if "aliases" in meta:
            continue
        # limit is a request budget, so failed lookups spend it too — a
        # store full of erroring artists must not hammer MB uncapped
        if limit is not None and attempts >= limit:
            break
        attempts += 1
        try:
            data = http.get_json(f"{MB}/artist/{row['mbid']}",
                                 params={"inc": "aliases", "fmt": "json"})
        except Exception:
            stats["errors"] += 1  # per-item: one dead mbid never aborts the run
            continue
        aliases = [a["name"] for a in (data or {}).get("aliases") or []
                   if isinstance(a, dict) and a.get("name")]
        db.upsert_item_fields(conn, row["id"], meta={"aliases": aliases})
        stats["fetched"] += 1
        _register(conn, row["id"], row["title"], aliases, stats, fold)
        # MB politeness makes big runs span minutes; commit as we go so
        # an abort keeps the aliases already paid for.
        conn.commit()
