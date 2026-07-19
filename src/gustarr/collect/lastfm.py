"""Last.fm collector: scrobbles + loved tracks → taste events.

Each scrobble/loved lands on BOTH the track item and its artist item —
they are separate domains, so artist events drive artist recommendations
while track events stay available for album/track granularity later.
Rows resolve through db.resolve_item: mbid when Last.fm supplies one,
the spelling otherwise, so re-syncs and spelling variants converge on
one item instead of minting twins.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from .. import db, ids
from ..http import get_json

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


def _resolve_pair(conn: sqlite3.Connection, t: dict) -> tuple[int, int]:
    """Resolve one API row to its (track, artist) item ids, post-merge.

    Raises ValueError when the row carries no usable names/mbids.
    """
    artist = t.get("artist") or {}
    # extended=1 gives {"name": ...}; unextended fallback is {"#text": ...}
    artist_name = (artist.get("name") or artist.get("#text") or "").strip()
    artist_mbid = artist.get("mbid") or ""
    if artist_mbid:
        artist_id = db.resolve_item(conn, "artist", "mbid", artist_mbid,
                                    title=artist_name or None)
        if artist_name:
            # Last.fm pairing the name with an mbid folds any name-minted
            # twin into the mbid item — its own assertion beats enrich's
            # fuzzy MusicBrainz name search; continue with the survivor.
            artist_id = db.attach_identity(conn, artist_id, "name", artist_name)
    elif artist_name:
        artist_id = db.resolve_item(conn, "artist", "name", artist_name, title=artist_name)
    else:
        raise ValueError("row has no artist mbid or name")

    track_name = (t.get("name") or "").strip()
    track_mbid = t.get("mbid") or ""
    meta: dict[str, Any] = {"artist": artist_name, "artist_id": artist_id}
    album_name = (t.get("album") or {}).get("#text") or ""
    if album_name:  # never merge an empty album over a known one
        meta["album"] = album_name
    if track_mbid:
        track_id = db.resolve_item(conn, "track", "mbid", track_mbid,
                                   title=track_name or None, meta=meta)
    elif artist_name and track_name:
        # the artist name disambiguates covers; the unit separator keeps
        # the artist/title boundary so shifted splits can't collide, and
        # matches the keys v2 stores migrated in with
        track_id = db.resolve_item(conn, "track", "name",
                                   f"{artist_name}{ids.SEP}{track_name}",
                                   title=track_name, meta=meta)
    else:
        raise ValueError("row has no track mbid or name")
    return track_id, artist_id


def sync(
    conn: sqlite3.Connection,
    cfg: Any,
    full: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    # api_key alone is a legitimate config (enrich/candidates use it
    # without a user), so skip instead of KeyError-ing the pipeline.
    api_key = cfg.lastfm.get("api_key")
    users = {name: p.lastfm_user for name, p in cfg.profiles.items() if p.lastfm_user}
    if not (api_key and users):
        return {"skipped": "lastfm not fully configured"}
    # flat totals across profiles: the per-profile split isn't actionable
    # in a pipeline log line, and single-user setups keep the old shape
    stats: dict[str, Any] = {"scrobbles": 0, "loved": 0, "items": 0, "pages": 0,
                             "profiles": 0,
                             "profiles_skipped": len(cfg.profiles) - len(users)}
    for profile, user in users.items():
        stats["profiles"] += 1
        _sync_profile(conn, profile, user, api_key, full, transport, stats)
    return stats


def _sync_profile(
    conn: sqlite3.Connection,
    profile: str,
    user: str,
    api_key: str,
    full: bool,
    transport: httpx.BaseTransport | None,
    stats: dict[str, Any],
) -> None:
    """One profile's walk: its own uts cursor, events under its profile."""
    seen_items: set[int] = set()
    base = {"api_key": api_key, "user": user, "format": "json", "limit": PAGE_LIMIT}

    cursor = None if full else db.pget_state(conn, profile, CURSOR_KEY)
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
            track_id, artist_id = _resolve_pair(conn, t)
        except ValueError:
            continue
        ts = _iso(uts)
        if db.add_event(conn, ts, track_id, "scrobble", 1.0, "lastfm",
                        profile=profile):
            stats["scrobbles"] += 1
        # dedup=track id: two different tracks scrobbled the same second
        # must both count on the artist item; re-syncs still collide.
        db.add_event(conn, ts, artist_id, "scrobble", 1.0, "lastfm",
                     dedup=str(track_id), profile=profile)
        seen_items.update((track_id, artist_id))
        max_uts = max(max_uts, uts)

    for t in _walk({**base, "method": "user.getlovedtracks"}, "lovedtracks", stats, transport):
        uts = int((t.get("date") or {}).get("uts", 0) or 0)
        if not uts:
            continue
        try:
            track_id, artist_id = _resolve_pair(conn, t)
        except ValueError:
            continue
        ts = _iso(uts)
        if db.add_event(conn, ts, track_id, "loved", 1.0, "lastfm",
                        profile=profile):
            stats["loved"] += 1
        db.add_event(conn, ts, artist_id, "loved", 1.0, "lastfm",
                     dedup=str(track_id), profile=profile)
        seen_items.update((track_id, artist_id))

    if max_uts:
        db.pset_state(conn, profile, CURSOR_KEY, str(max_uts))
    stats["items"] += len(seen_items)
