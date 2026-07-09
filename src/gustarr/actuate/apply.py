"""Actuation: the only pipeline stage that changes the outside world.

Music flows automatically inside weekly caps; video moves only after an
explicit user approve. The caps, TTL expiry and dry_run all live here so
every irreversible step is inspectable in one place.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db, http, signals
from ..config import Config
from . import jellyfin_collections
from .arr_client import ArrError, LidarrClient, RadarrClient, SonarrClient
from .jellyfin_collections import external_id


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_start() -> str:
    ref = datetime.now(timezone.utc)
    start = (ref - timedelta(days=ref.isoweekday() - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return _iso(start)


def _mark(conn: sqlite3.Connection, rec_id: int, status: str, ts: str) -> None:
    conn.execute(
        "UPDATE recommendations SET status=?, acted_at=? WHERE id=?", (status, ts, rec_id))


def run(conn: sqlite3.Connection, cfg: Config, dry_run: bool = False) -> dict[str, Any]:
    ts = db.now()
    stats: dict[str, Any] = {"errors": []}
    if dry_run:
        stats["would_add"] = []

    # rank owns TTL expiry too; repeated here so apply never acts on a
    # proposal that outlived autonomy.proposal_ttl_days between runs.
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(days=cfg.autonomy.proposal_ttl_days))
    if dry_run:
        stats["would_expire"] = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE status='proposed' AND ts<?",
            (cutoff,)).fetchone()[0]
    else:
        stats["expired"] = conn.execute(
            "UPDATE recommendations SET status='expired', acted_at=?"
            " WHERE status='proposed' AND ts<?", (ts, cutoff)).rowcount

    _apply_music(conn, cfg, ts, dry_run, stats)
    _apply_video(conn, cfg, ts, dry_run, stats)
    _expire_overflow(conn, cfg, ts, dry_run, stats)

    try:
        stats["jellyfin"] = jellyfin_collections.sync_collections(conn, cfg, dry_run=dry_run)
    except Exception as exc:  # collections are cosmetic — never fail the run
        stats["jellyfin_error"] = str(exc)
    return stats


def _apply_music(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any]
) -> None:
    stats["music_added"] = 0
    if cfg.autonomy.music_mode != "auto":
        return
    acted = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE domain='artist'"
        " AND status IN ('auto_added','added') AND acted_at>=?",
        (_week_start(),)).fetchone()[0]
    budget = max(0, cfg.autonomy.music_max_artists_per_week - acted)
    stats["music_budget"] = budget
    if budget == 0:
        return
    rows = conn.execute(
        "SELECT r.id, r.item_id, r.why, i.title, i.ids FROM recommendations r"
        " JOIN items i ON i.id = r.item_id"
        " WHERE r.status='proposed' AND r.domain='artist' ORDER BY r.score DESC").fetchall()
    picks: list[tuple[sqlite3.Row, str]] = []
    for row in rows:
        mbid = external_id(row, "mbid")
        if mbid:
            picks.append((row, mbid))
        if len(picks) == budget:
            break
    if not picks:
        return
    if dry_run:
        stats["would_add"].extend(row["title"] or row["item_id"] for row, _ in picks)
        return
    if cfg.lidarr is None:
        stats["errors"].append("music proposals ready but lidarr is not configured")
        return
    client = LidarrClient(cfg.lidarr)
    for row, mbid in picks:
        try:
            client.add_artist(mbid)
        except (http.ApiError, ArrError) as exc:
            stats["errors"].append(f"lidarr add {row['title'] or mbid}: {exc}")
            why = json.loads(row["why"] or "{}")
            why["attempts"] = int(why.get("attempts", 0)) + 1
            conn.execute(
                "UPDATE recommendations SET why=? WHERE id=?", (json.dumps(why), row["id"]))
            continue
        # library_add, not approve: an auto-add is gustarr's own action and
        # must not feed back into training as user praise.
        db.add_event(conn, ts, row["item_id"], "library_add",
                     signals.WEIGHTS["library_add"], "gustarr", {"rec_id": row["id"]})
        _mark(conn, row["id"], "auto_added", ts)
        stats["music_added"] += 1


def _apply_video(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any]
) -> None:
    stats["video_added"] = 0
    stats["video_failed"] = 0
    rows = conn.execute(
        "SELECT r.id, r.item_id, r.domain, i.title, i.ids FROM recommendations r"
        " JOIN items i ON i.id = r.item_id"
        " WHERE r.status='approved' AND r.domain IN ('movie','series')"
        " ORDER BY r.score DESC").fetchall()
    if not rows:
        return
    if dry_run:
        stats["would_add"].extend(row["title"] or row["item_id"] for row in rows)
        return
    clients: dict[str, RadarrClient | SonarrClient] = {}
    for row in rows:
        domain, title = row["domain"], row["title"] or row["item_id"]
        arr_name = "radarr" if domain == "movie" else "sonarr"
        arr_cfg = cfg.radarr if domain == "movie" else cfg.sonarr
        if arr_cfg is None:
            # stays approved: retried once the arr gets configured
            stats["errors"].append(f"{title}: approved but {arr_name} is not configured")
            continue
        ns_key = "tmdb" if domain == "movie" else "tvdb"
        ext = external_id(row, ns_key)
        if ext is None:
            stats["errors"].append(f"{title}: no {ns_key} id, cannot add to {arr_name}")
            _mark(conn, row["id"], "failed", ts)
            stats["video_failed"] += 1
            continue
        if arr_name not in clients:
            clients[arr_name] = (
                RadarrClient(arr_cfg) if domain == "movie" else SonarrClient(arr_cfg))
        try:
            clients[arr_name].add(ext)
        except (http.ApiError, ArrError) as exc:
            stats["errors"].append(f"{arr_name} add {title}: {exc}")
            _mark(conn, row["id"], "failed", ts)
            stats["video_failed"] += 1
            continue
        # approve event was already written when the user approved.
        _mark(conn, row["id"], "added", ts)
        stats["video_added"] += 1


def _expire_overflow(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any]
) -> None:
    pending = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE status='proposed'"
        " AND domain IN ('movie','series')").fetchone()[0]
    surplus = max(0, pending - cfg.autonomy.video_queue_max_pending)
    if dry_run:
        stats["would_expire_overflow"] = surplus
        return
    stats["overflow_expired"] = 0
    if surplus == 0:
        return
    stats["overflow_expired"] = conn.execute(
        "UPDATE recommendations SET status='expired', acted_at=? WHERE id IN ("
        " SELECT id FROM recommendations WHERE status='proposed'"
        " AND domain IN ('movie','series') ORDER BY score ASC, id ASC LIMIT ?)",
        (ts, surplus)).rowcount
