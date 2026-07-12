"""Duplicate repair: one artist, one item, whatever the spelling.

The same CJK artist arrives romanized from one source and in kana/kanji
from another; each spelling minted its own name-keyed fallback item, so
listening history split across twins (a live store held 468 fallback
artists, 47 of them CJK-named with events). Three idempotent passes:

  1. normalize — re-mint every name-keyed id with the current ids.make
     (NFKC + casefold + whitespace collapse) and merge rows whose id
     changes, healing items minted before key normalization existed.
     Width/case/composition variants of one spelling collapse here.
  2. alias-register — a MusicBrainz artist's alias names are exactly the
     spellings collectors mint fallback ids from; recording each in
     item_aliases folds existing twins into the mbid item and redirects
     all future writes (db.canonical_id). Cross-script variants
     (kana/kanji vs romaji) collapse here.
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
    stats = {"normalized": 0, "alias_registered": 0, "alias_merged": 0,
             "alias_conflicts": 0, "fetched": 0, "errors": 0}
    _normalize_pass(conn, stats)
    _alias_pass(conn, stats)
    if fetch:
        _fetch_pass(conn, stats, limit)
    return stats


def _normalize_pass(conn: sqlite3.Connection, stats: dict[str, int]) -> None:
    # fetchall up front: merge_item rewrites rows under the cursor otherwise
    for row in conn.execute("SELECT id FROM items").fetchall():
        domain, ns, key = ids.parse(row["id"])
        if ns != "lastfm":
            continue  # numeric/uuid keys normalize to themselves
        try:
            new_id = ids.make(domain, ns, *key.split(ids._SEP))
        except ValueError:
            continue  # name normalizes to nothing; a stranded row beats a wrong merge
        if new_id == row["id"]:
            continue
        # An earlier run may have redirected the normalized form onward
        # (e.g. onto an mbid item); merge straight to the live row so
        # events never point at an alias with no items row behind it.
        target = db.canonical_id(conn, new_id)
        if target == row["id"]:
            continue
        db.merge_item(conn, row["id"], target)
        stats["normalized"] += 1


def _alias_pass(conn: sqlite3.Connection, stats: dict[str, int]) -> None:
    rows = conn.execute(
        "SELECT id, title, meta FROM items"
        " WHERE domain='artist' AND id LIKE 'artist:mbid:%'").fetchall()
    for row in rows:
        meta = json.loads(row["meta"])
        # key-present is the marker, so an empty list (artist fetched,
        # MB knows no aliases) still registers the primary title and is
        # never re-fetched.
        if "aliases" in meta:
            _register(conn, row["id"], row["title"], meta["aliases"], stats)


def _register(conn: sqlite3.Connection, item_id: str, title: str | None,
              aliases: list[str], stats: dict[str, int]) -> None:
    fallback_ids = set()
    for name in [title, *aliases]:
        if not name or not isinstance(name, str):
            continue
        try:
            # aliases are stored raw; ids.make is the normalizer, so this
            # is exactly the id a collector would mint for that spelling
            fallback_ids.add(ids.make("artist", "lastfm", name))
        except ValueError:
            continue
    for fid in sorted(fallback_ids):  # deterministic conflict attribution
        row = conn.execute(
            "SELECT canonical_id FROM item_aliases WHERE alias_id=?", (fid,)).fetchone()
        if row is not None:
            if row["canonical_id"] != item_id:
                # first-writer-wins: two artists genuinely sharing a
                # spelling would ping-pong the mapping forever otherwise
                stats["alias_conflicts"] += 1
            continue
        if conn.execute("SELECT 1 FROM items WHERE id=?", (fid,)).fetchone():
            db.merge_item(conn, fid, item_id)  # records the alias row itself
            stats["alias_merged"] += 1
        else:
            conn.execute(
                "INSERT INTO item_aliases (alias_id, canonical_id) VALUES (?,?)",
                (fid, item_id))
            stats["alias_registered"] += 1


def _fetch_pass(conn: sqlite3.Connection, stats: dict[str, int], limit: int | None) -> None:
    rows = conn.execute(
        "SELECT id, title, meta FROM items"
        " WHERE domain='artist' AND id LIKE 'artist:mbid:%'"
        # played artists only: each lookup costs 1.1s of MB politeness,
        # and only items with history have split history worth healing
        " AND EXISTS(SELECT 1 FROM events e WHERE e.item_id = items.id)").fetchall()
    attempts = 0
    for row in rows:
        meta = json.loads(row["meta"])
        if "aliases" in meta:
            continue
        # limit is a request budget, so failed lookups spend it too — a
        # store full of erroring artists must not hammer MB uncapped
        if limit is not None and attempts >= limit:
            break
        attempts += 1
        mbid = ids.parse(row["id"])[2]
        try:
            data = http.get_json(f"{MB}/artist/{mbid}",
                                 params={"inc": "aliases", "fmt": "json"})
        except Exception:
            stats["errors"] += 1  # per-item: one dead mbid never aborts the run
            continue
        aliases = [a["name"] for a in (data or {}).get("aliases") or []
                   if isinstance(a, dict) and a.get("name")]
        db.upsert_item(conn, row["id"], "artist", meta={"aliases": aliases})
        stats["fetched"] += 1
        _register(conn, row["id"], row["title"], aliases, stats)
        # MB politeness makes big runs span minutes; commit as we go so
        # an abort keeps the aliases already paid for.
        conn.commit()
