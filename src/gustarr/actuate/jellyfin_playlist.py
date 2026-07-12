"""Mirror each profile's ListenBrainz Weekly Exploration into a playable
Jellyfin playlist, so exploration picks land where the listening happens.

One playlist per profile, owned by that profile's Jellyfin user and
tracked by a state key — gustarr only ever touches the playlist it
created. Every run reconciles contents against what the library holds
NOW: Weekly Exploration is music the user mostly doesn't have yet, and
gustarr itself feeds those tracks to Lidarr, so matches keep arriving
during the week and the playlist grows with them. Jellyfin 10.x exposes
no workable in-place playlist edit to an API key (the /Playlists/{id}
/Items surface needs a user context the token doesn't carry), so a
changed track set deletes and recreates the playlist — LB's running
order restored, at the cost of a fresh playlist id (kept in state).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db, http, ids
from ..collect.listenbrainz import _headers as lb_headers
from ..collect.listenbrainz import _recording_mbid, fetch_weekly
from ..config import Config

PLAYLIST_NAME = "Weekly Exploration"

_ID_KEY = "jellyfin:weekly_playlist_id"


def _user_id(base: str, headers: dict[str, str], username: str) -> str | None:
    users = http.get_json(f"{base}/Users", headers=headers) or []
    for u in users:
        if (u.get("Name") or "").casefold() == username.casefold():
            return u.get("Id")
    return None


def _find_track(base: str, headers: dict[str, str],
                mbid: str, title: str | None, creator: str | None) -> str | None:
    """The library's copy of one LB recording. Jellyfin 10.x has no
    provider-id query (the Emby-era AnyProviderIdEquals is silently
    ignored), so this searches by title and verifies each hit client-side:
    the MusicBrainz track id first (Lidarr-tagged libraries carry it),
    then an exact normalized title whose artist credit matches — a fuzzy
    title hit on the wrong artist is worse than a missing track."""
    if not title:
        return None
    found = http.get_json(
        f"{base}/Items",
        params={"Recursive": "true", "IncludeItemTypes": "Audio",
                "SearchTerm": title, "Fields": "ProviderIds", "Limit": 25},
        headers=headers)
    items = (found or {}).get("Items") or []
    want_mbid = mbid.casefold()
    for item in items:
        pids = {k.casefold(): str(v).casefold()
                for k, v in (item.get("ProviderIds") or {}).items() if v}
        if pids.get("musicbrainztrack") == want_mbid:
            return item["Id"]
    if not creator:
        return None
    want_artist = ids.normalize_key(creator)
    want_title = ids.normalize_key(title)
    for item in items:
        # Jellyfin fills Artists for audio DTOs unconditionally
        artists = [ids.normalize_key(a) for a in item.get("Artists") or []]
        if ids.normalize_key(item.get("Name") or "") == want_title and want_artist in artists:
            return item["Id"]
    return None


def _children(base: str, headers: dict[str, str], playlist_id: str) -> set[str] | None:
    """Member ids of the tracked playlist, or None when it no longer
    exists (the user deleted it). Read via /Items?ParentId — unlike the
    /Playlists/{id}/Items surface, it works for an API key."""
    got = http.get_json(f"{base}/Items",
                        params={"Ids": playlist_id, "Recursive": "true"}, headers=headers)
    if not (got or {}).get("Items"):
        return None
    members = http.get_json(f"{base}/Items",
                            params={"ParentId": playlist_id}, headers=headers)
    return {i["Id"] for i in (members or {}).get("Items") or []}


def sync_playlists(conn: sqlite3.Connection, cfg: Config,
                   dry_run: bool = False) -> dict[str, Any]:
    base = (cfg.jellyfin.get("url") or "").rstrip("/")
    token = cfg.jellyfin.get("api_key")
    if not base or not token or not cfg.listenbrainz.get("weekly_playlist", True):
        return {"skipped": True}
    lb_hdrs = lb_headers(cfg)
    headers = {"X-Emby-Token": token}
    stats: dict[str, Any] = {"profiles": 0, "matched": 0, "missing": 0,
                             "created": 0, "rebuilt": 0, "unchanged": 0}

    for profile, p in cfg.profiles.items():
        if not p.listenbrainz_user or not p.jellyfin_user:
            continue
        fetched = fetch_weekly(p.listenbrainz_user, lb_hdrs)
        if fetched is None:
            continue  # LB hasn't generated a week yet — next run
        _week_mbid, tracks = fetched
        stats["profiles"] += 1
        if dry_run:
            stats["would_sync"] = stats.get("would_sync", 0) + 1
            continue

        jf_ids: list[str] = []
        for track in tracks:
            mbid = _recording_mbid(track)
            if not mbid:
                continue
            jf_id = _find_track(base, headers, mbid,
                                track.get("title"), track.get("creator"))
            if jf_id is None:
                stats["missing"] += 1
            elif jf_id not in jf_ids:
                stats["matched"] += 1
                jf_ids.append(jf_id)
        if not jf_ids:
            continue  # nothing in the library yet; never wipe, try next run

        # Only the playlist gustarr created is ever touched: a stored id
        # that vanished (user deleted it) means create a fresh one, and a
        # same-named playlist the user made themselves is never adopted.
        playlist_id = db.pget_state(conn, profile, _ID_KEY)
        current = _children(base, headers, playlist_id) if playlist_id else None
        if current is not None and current == set(jf_ids):
            stats["unchanged"] += 1
            continue
        user_id = _user_id(base, headers, p.jellyfin_user)
        if user_id is None:
            continue  # profile's Jellyfin user unknown to the server
        if current is not None:
            # no in-place edit for an API key: rebuild with LB's order
            http.request_json("DELETE", f"{base}/Items/{playlist_id}", headers=headers)
        made = http.post_json(
            f"{base}/Playlists",
            json_body={"Name": PLAYLIST_NAME, "Ids": jf_ids,
                       "UserId": user_id, "MediaType": "Audio"},
            headers=headers) or {}
        new_id = made.get("Id")
        if not new_id:
            continue
        db.pset_state(conn, profile, _ID_KEY, new_id)
        # the playlist exists in Jellyfin now — the record of it must
        # survive a crash later in the run (same rule as _record_add)
        conn.commit()
        stats["rebuilt" if current is not None else "created"] += 1
    return stats
