"""Weekly Exploration album acquisition: buy the playlist some substance.

ListenBrainz's Weekly Exploration is by definition music the user does
not own, so the Jellyfin playlist mirror starts every week empty and
only fills as tracks arrive. This stage closes that loop: each ISO week
a small budgeted number of the playlist's albums is fed to Lidarr, so
downloads land during the week and the playlist becomes playable.

The recording→release-group resolution costs one MusicBrainz lookup per
track at the mandatory 1.1s spacing, so the resolved list is cached in
state per playlist mbid — a nightly re-run of an already-seen week does
zero MusicBrainz traffic, and older weeks' caches are dropped when a new
playlist appears. Adds go through the exact same LidarrClient album path
and record-keeping (recommendation row → queue's transition owner →
inert audit event, committed immediately) as apply's ordinary auto-adds.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .. import candidates, db, http, settings
from ..collect.listenbrainz import _headers as lb_headers
from ..collect.listenbrainz import _recording_mbid, fetch_weekly
from ..config import Config
from ..enrich import MB, _credit_artist_mbid, _credit_names
from .arr_client import ArrError, LidarrClient

# One cache row per playlist mbid; run() deletes siblings when a new
# week's playlist shows up, so state never accumulates dead weeks.
_RG_PREFIX = "weekly_exploration:rgs:"


def run(conn: sqlite3.Connection, cfg: Config, dry_run: bool = False) -> dict[str, Any]:
    # call-time import: apply imports this module at load, and the week
    # window + add bookkeeping must stay apply's single implementation
    from .apply import _record_add, _week_start

    limit = settings.get(conn, cfg, "weekly_exploration_albums")
    if limit <= 0:
        return {"skipped": "disabled"}  # 0 disables: zero HTTP, zero writes
    users = {name: p.listenbrainz_user
             for name, p in cfg.profiles.items() if p.listenbrainz_user}
    if not users:
        return {"skipped": "listenbrainz not configured"}
    if cfg.lidarr is None:
        return {"skipped": "lidarr not configured"}
    if dry_run:
        # resolution alone is a minute of MusicBrainz grind and a cache
        # write — too much side effect for a preview, so dry runs skip
        # the stage entirely rather than half-simulate it.
        return {"skipped": "dry_run"}

    # The budget is global per ISO week like every other budget (one
    # disk, one Lidarr): what any profile's playlist spent is spent.
    # Only this stage's own adds count against it — they carry the why
    # marker — so a busy regular album week never starves exploration.
    spent = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE status='auto_added'"
        " AND acted_at>=? AND json_extract(why, '$.weekly_exploration')",
        (_week_start(),)).fetchone()[0]
    budget = max(0, limit - spent)
    stats: dict[str, Any] = {"resolved": 0, "cached": False, "added": 0,
                             "budget": budget, "skipped": 0, "errors": []}
    if budget == 0:
        return stats  # spent for the week: not even the LB fetch runs

    headers = lb_headers(cfg)
    client = LidarrClient(cfg.lidarr)
    ts = db.now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for profile, user in users.items():
        if budget <= 0:
            break
        fetched = fetch_weekly(user, headers)
        if fetched is None:
            continue  # LB hasn't generated a week yet — next run
        playlist_mbid, tracks = fetched
        rgs, cached = _release_groups(conn, playlist_mbid, tracks)
        stats["resolved"] += len(rgs)
        stats["cached"] = stats["cached"] or cached

        excluded = candidates._excluded_ids(conn, profile)
        rejected_artists = _rejected_artists(conn, profile)
        # top of the playlist first: LB orders the week by its own
        # confidence, and the budget should buy the best of it
        for entry in rgs:
            if budget <= 0:
                break
            meta = {k: entry[k] for k in ("artist", "artist_mbid") if entry.get(k)}
            item_id = db.resolve_item(conn, "album", "mbid", entry["rg"],
                                      title=entry.get("title"), meta=meta or None)
            if item_id in excluded:
                stats["skipped"] += 1
                continue
            artist_mbid = entry.get("artist_mbid")
            if artist_mbid is not None:
                artist_item = db.lookup_item(conn, "artist", "mbid", artist_mbid)
                if artist_item is not None and artist_item in rejected_artists:
                    # a rejected artist's albums must not sneak back in
                    # through the playlist side door
                    stats["skipped"] += 1
                    continue
            try:
                client.add_album(entry["rg"])
            except (http.ApiError, ArrError) as exc:
                # one album failing (outage or verdict) must not spend
                # the budget or stop the rest of the playlist
                stats["errors"].append(f"lidarr add {entry.get('title') or entry['rg']}:"
                                       f" {exc}")
                continue
            # Record exactly like apply's auto-adds: a recommendation row
            # moved by the transition owner, the inert audit event, and
            # an immediate commit (the arr add is irreversible). The why
            # marker is both provenance for the UI and the budget key.
            cur = conn.execute(
                "INSERT INTO recommendations"
                " (profile, run_id, ts, domain, item_id, score, why, status)"
                " VALUES (?,?,?,?,?,?,?,'proposed')",
                (profile, run_id, ts, "album", item_id, 0.0,
                 json.dumps({"weekly_exploration": True, "playlist": playlist_mbid})))
            row = conn.execute(
                "SELECT id, item_id, profile, status, why FROM recommendations"
                " WHERE id=?", (cur.lastrowid,)).fetchone()
            _record_add(conn, row, ts)
            budget -= 1
            stats["added"] += 1
    stats["budget"] = budget
    return stats


def _release_groups(
    conn: sqlite3.Connection, playlist_mbid: str, tracks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """The playlist's albums, resolved once per playlist: each recording
    mbid → the first release's release-group (id, title, primary artist
    credit), deduped in playlist order. Cached in state so the 6 other
    nightly runs of the same week cost zero MusicBrainz round-trips."""
    key = _RG_PREFIX + playlist_mbid
    raw = db.get_state(conn, key)
    if raw is not None:
        return json.loads(raw), True
    rgs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for track in tracks:
        rec_mbid = _recording_mbid(track)
        if not rec_mbid:
            continue
        try:
            data = http.get_json(
                f"{MB}/recording/{rec_mbid}",
                params={"inc": "artist-credits+releases+release-groups", "fmt": "json"})
        except http.ApiError:
            continue  # one bad recording skips the track, not the stage
        releases = (data or {}).get("releases") or []
        rg = releases[0].get("release-group") if releases else None
        if not isinstance(rg, dict) or not rg.get("id") or rg["id"] in seen:
            continue
        seen.add(rg["id"])
        entry: dict[str, Any] = {"rg": rg["id"], "title": rg.get("title")}
        names = _credit_names(data)
        if names:
            entry["artist"] = names[0]
        artist_mbid = _credit_artist_mbid(data)
        if artist_mbid:
            entry["artist_mbid"] = artist_mbid
        rgs.append(entry)
    if rgs:
        # an all-failures round must NOT cache empty and poison the week
        # — leave it uncached so the next nightly retries the lookups.
        # A new playlist retires the older weeks' caches in the same
        # transaction, and the commit makes the expensive MB round
        # survive a crash later in the run.
        conn.execute("DELETE FROM state WHERE key LIKE ? AND key != ?",
                     (_RG_PREFIX + "%", key))
        db.set_state(conn, key, json.dumps(rgs))
        conn.commit()
    return rgs, False


def _rejected_artists(conn: sqlite3.Connection, profile: str) -> set[int]:
    """Artist items this profile ever rejected — reject events are the
    lasting verdict (candidates uses the same source for items)."""
    return {r["item_id"] for r in conn.execute(
        "SELECT DISTINCT e.item_id FROM events e JOIN items i ON i.id = e.item_id"
        " WHERE e.profile=? AND e.kind='reject' AND i.domain='artist'", (profile,))}
