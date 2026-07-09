"""Last.fm collector: scrobbles + loved tracks → taste events.

Each scrobble/loved lands on BOTH the track item and its artist item —
they are separate domains, so artist events drive artist recommendations
while track events stay available for album/track granularity later.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from .. import db, ids
from ..http import get_json
from ..signals import WEIGHTS

API_ROOT = "https://ws.audioscrobbler.com/2.0/"
CURSOR_KEY = "lastfm:last_uts"
PAGE_LIMIT = 200
# 500 pages × 200 = 100k scrobbles per walk; beyond that we truncate and
# flag it in stats rather than hammer the API for hours.
MAX_PAGES = 500


def _iso(uts: int) -> str:
    return datetime.fromtimestamp(uts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unbox(payload: Any, box_key: str) -> tuple[list[dict], int]:
    box = (payload or {}).get(box_key) or {}
    tracks = box.get("track") or []
    if isinstance(tracks, dict):  # last.fm collapses single-element lists
        tracks = [tracks]
    total_pages = int((box.get("@attr") or {}).get("totalPages", 0) or 0)
    return tracks, total_pages


def _walk(
    params: dict[str, Any],
    box_key: str,
    stats: dict[str, Any],
    transport: httpx.BaseTransport | None,
) -> Iterator[dict]:
    page, total_pages = 1, 1
    while page <= total_pages:
        payload = get_json(API_ROOT, params={**params, "page": page}, transport=transport)
        tracks, total_pages = _unbox(payload, box_key)
        stats["pages"] += 1
        if total_pages > MAX_PAGES:
            total_pages = MAX_PAGES
            stats["warning"] = f"{box_key} walk capped at {MAX_PAGES} pages"
        yield from tracks
        page += 1


def _merge_name_twin(conn: sqlite3.Connection, artist_name: str, mbid_id: str) -> None:
    """Fold a previously minted artist:lastfm:<name> item into the mbid
    item the moment Last.fm itself pairs the name with an mbid — its own
    assertion beats enrich's fuzzy MusicBrainz name search."""
    try:
        fallback_id = ids.make("artist", "lastfm", artist_name)
    except ValueError:
        return
    if fallback_id != mbid_id and conn.execute(
            "SELECT 1 FROM items WHERE id=?", (fallback_id,)).fetchone():
        db.merge_item(conn, fallback_id, mbid_id)


def _upsert_pair(conn: sqlite3.Connection, t: dict) -> tuple[str, str]:
    """Mint/refresh artist + track items for one API row; returns their
    effective ids (merge-resolved, so never a merged-away fallback).

    Raises ValueError (from ids.make) when the row has no usable names/mbids.
    """
    artist = t.get("artist") or {}
    # extended=1 gives {"name": ...}; unextended fallback is {"#text": ...}
    artist_name = artist.get("name") or artist.get("#text") or ""
    artist_mbid = artist.get("mbid") or ""
    if artist_mbid:
        artist_id = ids.make("artist", "mbid", artist_mbid)
    else:
        artist_id = ids.make("artist", "lastfm", artist_name)
    db.upsert_item(conn, artist_id, "artist", title=artist_name,
                   ids={"mbid": artist_mbid} if artist_mbid else None)
    if artist_mbid:
        _merge_name_twin(conn, artist_name, artist_id)
    # follow-up writes must land on the live row, not a merged fallback
    artist_id = db.canonical_id(conn, artist_id)

    track_name = t.get("name") or ""
    track_mbid = t.get("mbid") or ""
    if track_mbid:
        track_id = ids.make("track", "mbid", track_mbid)
    else:
        track_id = ids.make("track", "lastfm", artist_name, track_name)
    meta: dict[str, Any] = {"artist": artist_name, "artist_id": artist_id}
    album_name = (t.get("album") or {}).get("#text") or ""
    if album_name:  # never merge an empty album over a known one
        meta["album"] = album_name
    db.upsert_item(conn, track_id, "track", title=track_name,
                   ids={"mbid": track_mbid} if track_mbid else None, meta=meta)
    return db.canonical_id(conn, track_id), artist_id


def sync(
    conn: sqlite3.Connection,
    cfg: Any,
    full: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    # api_key alone is a legitimate config (enrich/candidates use it
    # without a user), so skip instead of KeyError-ing the pipeline.
    api_key, user = cfg.lastfm.get("api_key"), cfg.lastfm.get("user")
    if not (api_key and user):
        return {"skipped": "lastfm not fully configured"}
    stats: dict[str, Any] = {"scrobbles": 0, "loved": 0, "items": 0, "pages": 0}
    seen_items: set[str] = set()
    base = {"api_key": api_key, "user": user, "format": "json", "limit": PAGE_LIMIT}

    cursor = None if full else db.get_state(conn, CURSOR_KEY)
    recent = {**base, "method": "user.getrecenttracks", "extended": 1}
    if cursor:
        recent["from"] = int(cursor) + 1
    max_uts = int(cursor) if cursor else 0

    for t in _walk(recent, "recenttracks", stats, transport):
        if (t.get("@attr") or {}).get("nowplaying"):
            continue
        uts = int((t.get("date") or {}).get("uts", 0) or 0)
        if not uts:
            continue
        try:
            track_id, artist_id = _upsert_pair(conn, t)
        except ValueError:
            continue
        ts = _iso(uts)
        if db.add_event(conn, ts, track_id, "scrobble", WEIGHTS["scrobble"], "lastfm"):
            stats["scrobbles"] += 1
        # dedup=track_id: two different tracks scrobbled the same second
        # must both count on the artist item; re-syncs still collide.
        db.add_event(conn, ts, artist_id, "scrobble", WEIGHTS["scrobble"], "lastfm",
                     dedup=track_id)
        seen_items.update((track_id, artist_id))
        max_uts = max(max_uts, uts)

    for t in _walk({**base, "method": "user.getlovedtracks"}, "lovedtracks", stats, transport):
        uts = int((t.get("date") or {}).get("uts", 0) or 0)
        if not uts:
            continue
        try:
            track_id, artist_id = _upsert_pair(conn, t)
        except ValueError:
            continue
        ts = _iso(uts)
        if db.add_event(conn, ts, track_id, "loved", WEIGHTS["loved"], "lastfm"):
            stats["loved"] += 1
        db.add_event(conn, ts, artist_id, "loved", WEIGHTS["loved"], "lastfm",
                     dedup=track_id)
        seen_items.update((track_id, artist_id))

    if max_uts:
        db.set_state(conn, CURSOR_KEY, str(max_uts))
    stats["items"] = len(seen_items)
    return stats
