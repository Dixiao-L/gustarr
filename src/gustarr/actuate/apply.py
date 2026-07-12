"""Actuation: the only pipeline stage that changes the outside world.

Explicitly approved recommendations are actuated in every mode — an
approve is consent. On top of that, 'auto' mode adds proposed recs on
its own: music inside a weekly cap, video capped per run. The caps, TTL
expiry and dry_run all live here so every irreversible step is
inspectable in one place. Modes and caps go through settings.get so a
runtime override (state) beats the TOML; a 'paused' override stops the
whole stage before any HTTP or store write.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .. import db, http, settings
from ..config import Config
from . import jellyfin_collections
from .arr_client import ArrConfigError, ArrError, LidarrClient, RadarrClient, SonarrClient


def _ext_id(conn: sqlite3.Connection, item_id: int, ns: str) -> str | None:
    """The external id actuation addresses an arr with, straight from the
    item's identities — the only place external names live in v3."""
    return db.identities_of(conn, item_id).get(ns)


def _music_mbid(conn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
    """Albums prefer the release-group mbid enrich stashed in meta —
    Lidarr's album lookup keys on release groups, while the identities
    mbid may be any MB id a collector first saw. Artists (and albums
    enrich hasn't reached) fall back to the identities mbid."""
    rg = json.loads(row["meta"] or "{}").get("release_group_mbid")
    return str(rg) if rg else _ext_id(conn, row["item_id"], "mbid")


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


def _transient(exc: Exception) -> bool:
    """Transport failures (status None), 5xx/429 and 401/403 are outages
    of the arr, not verdicts on the item — credentials are service-level
    (enrich's taxonomy: an api-key rotation window must not burn
    approvals). ArrConfigError is likewise operator-fixable config, so
    the rec keeps its status and the next apply retries. Other 4xx and
    the remaining ArrErrors are deterministic failures."""
    if isinstance(exc, ArrConfigError):
        return True
    return isinstance(exc, http.ApiError) and (
        exc.status is None or exc.status in (401, 403, 429) or exc.status >= 500)


def _bump_attempts(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    why = json.loads(row["why"] or "{}")
    why["attempts"] = int(why.get("attempts", 0)) + 1
    conn.execute("UPDATE recommendations SET why=? WHERE id=?", (json.dumps(why), row["id"]))


def _record_add(conn: sqlite3.Connection, row: sqlite3.Row, ts: str) -> None:
    """Store writes for one successful arr add, committed immediately:
    the add is irreversible, so the record of it must survive a crash
    later in the run."""
    if row["status"] == "approved":
        # approve event was already written when the user approved.
        _mark(conn, row["id"], "added", ts)
    else:
        # audit trail only: 'auto_add' is not in signals.WEIGHTS, so
        # gustarr's own adds never feed back into training as praise.
        # Attributed to the rec's own profile: budgets are shared, but
        # the record of whose queue an add came from must not be.
        db.add_event(conn, ts, row["item_id"], "auto_add", 0.0, "gustarr",
                     {"rec_id": row["id"]}, profile=row["profile"])
        _mark(conn, row["id"], "auto_added", ts)
    conn.commit()


def run(conn: sqlite3.Connection, cfg: Config, dry_run: bool = False) -> dict[str, Any]:
    if settings.get(conn, cfg, "paused"):
        return {"paused": True}
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
    # One client for both music domains so the tag/profile/root lookups
    # are fetched once per run (they're cached on the instance).
    client = LidarrClient(cfg.lidarr) if cfg.lidarr is not None else None
    _apply_music_domain(
        conn, cfg, ts, dry_run, stats, client, domain="artist",
        budget_setting="music_max_artists_per_week",
        added_key="music_added", budget_key="music_budget", add=LidarrClient.add_artist)
    _apply_music_domain(
        conn, cfg, ts, dry_run, stats, client, domain="album",
        budget_setting="music_max_albums_per_week",
        added_key="albums_added", budget_key="albums_budget", add=LidarrClient.add_album)


def _apply_music_domain(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any],
    client: LidarrClient | None, *, domain: str, budget_setting: str,
    added_key: str, budget_key: str, add: Callable[[LidarrClient, str], Any],
) -> None:
    # Artists and albums share one flow: same mode switch, same approve
    # semantics, same error split — only the domain, the weekly budget
    # and the LidarrClient method differ.
    stats[added_key] = 0
    # The acted count deliberately ignores profile: one disk, one Lidarr,
    # so alice's auto-adds spend bob's remaining weekly allowance too.
    acted = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE domain=?"
        " AND status IN ('auto_added','added') AND acted_at>=?",
        (domain, _week_start())).fetchone()[0]
    budget = max(0, settings.get(conn, cfg, budget_setting) - acted)
    stats[budget_key] = budget
    # Approved rows are actuated in every mode (an explicit approve is
    # consent) and don't consume the weekly budget; proposed rows are
    # auto-picked only in auto mode, inside the budget. Every profile's
    # queue competes in one score-ordered pass — the budget is household
    # property, not a per-person allowance.
    auto = settings.get(conn, cfg, "music_mode") == "auto"
    rows = conn.execute(
        "SELECT r.id, r.item_id, r.profile, r.status, r.why, i.title, i.meta"
        " FROM recommendations r JOIN items i ON i.id = r.item_id"
        " WHERE r.domain=? AND r.status IN ('approved','proposed')"
        " ORDER BY r.status='approved' DESC, r.score DESC", (domain,)).fetchall()
    picks: list[tuple[sqlite3.Row, str]] = []
    unaddressable: list[sqlite3.Row] = []
    auto_picked = 0
    for row in rows:
        approved = row["status"] == "approved"
        if not approved and (not auto or auto_picked >= budget):
            continue
        mbid = _music_mbid(conn, row)
        if mbid is None:
            # an approved rec we cannot address fails loudly instead
            # of being stranded forever; a proposed one is just skipped.
            if approved:
                unaddressable.append(row)
            continue
        picks.append((row, mbid))
        auto_picked += 0 if approved else 1
    if not picks and not unaddressable:
        return
    if dry_run:
        stats["would_add"].extend(row["title"] or row["item_id"] for row, _ in picks)
        return
    if client is None:
        # stays approved/proposed: retried once lidarr gets configured
        stats["errors"].append(
            f"music {domain} recommendations ready but lidarr is not configured")
        return
    for row in unaddressable:
        stats["errors"].append(f"{row['title'] or row['item_id']}: no mbid,"
                               " cannot add to lidarr")
        _mark(conn, row["id"], "failed", ts)
    for row, mbid in picks:
        try:
            add(client, mbid)
        except (http.ApiError, ArrError) as exc:
            stats["errors"].append(f"lidarr add {row['title'] or mbid}: {exc}")
            if row["status"] == "approved" and not _transient(exc):
                _mark(conn, row["id"], "failed", ts)
            else:
                _bump_attempts(conn, row)
            continue
        _record_add(conn, row, ts)
        stats[added_key] += 1


def _apply_video(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any]
) -> None:
    stats["video_added"] = 0
    stats["video_failed"] = 0
    # No profile filter: an approve is consent whoever gave it, and auto
    # mode's per-run cap bounds writes to the shared arrs, so the top
    # proposals of every profile compete for it on score.
    sql = (
        "SELECT r.id, r.item_id, r.profile, r.domain, r.status, r.why, i.title"
        " FROM recommendations r JOIN items i ON i.id = r.item_id"
        " WHERE r.status=? AND r.domain IN ('movie','series') ORDER BY r.score DESC")
    rows = conn.execute(sql, ("approved",)).fetchall()
    if cfg.autonomy.video_mode == "auto":
        # auto mode: top proposed video recs are added without waiting
        # for an approve, capped at video_queue_max_pending per run
        # (video has no weekly budget; the pending cap bounds the blast
        # radius instead). Rows missing the arr's id don't burn a slot.
        cap = settings.get(conn, cfg, "video_queue_max_pending")
        for row in conn.execute(sql, ("proposed",)):
            if cap <= 0:
                break
            ns = "tmdb" if row["domain"] == "movie" else "tvdb"
            if _ext_id(conn, row["item_id"], ns) is not None:
                rows.append(row)
                cap -= 1
    if not rows:
        return
    if dry_run:
        stats["would_add"].extend(row["title"] or row["item_id"] for row in rows)
        return
    clients: dict[str, RadarrClient | SonarrClient] = {}
    for row in rows:
        domain, title = row["domain"], row["title"] or row["item_id"]
        approved = row["status"] == "approved"
        arr_name = "radarr" if domain == "movie" else "sonarr"
        arr_cfg = cfg.radarr if domain == "movie" else cfg.sonarr
        if arr_cfg is None:
            # stays approved/proposed: retried once the arr gets configured
            stats["errors"].append(f"{title}: ready but {arr_name} is not configured")
            continue
        ns_key = "tmdb" if domain == "movie" else "tvdb"
        ext = _ext_id(conn, row["item_id"], ns_key)
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
            if approved and not _transient(exc):
                _mark(conn, row["id"], "failed", ts)
                stats["video_failed"] += 1
            else:
                # arr outage or unactuated proposal: keep the current
                # status so the next apply retries.
                _bump_attempts(conn, row)
            continue
        _record_add(conn, row, ts)
        stats["video_added"] += 1


def _expire_overflow(
    conn: sqlite3.Connection, cfg: Config, ts: str, dry_run: bool, stats: dict[str, Any]
) -> None:
    # The pending cap bounds each profile's own approval queue (rank
    # fills per profile up to it), so the trim is per profile too — a
    # global trim would let one profile's backlog expire another's picks.
    cap = settings.get(conn, cfg, "video_queue_max_pending")
    expired = 0
    for profile in cfg.profiles:
        pending = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE profile=? AND status='proposed'"
            " AND domain IN ('movie','series')", (profile,)).fetchone()[0]
        surplus = max(0, pending - cap)
        if dry_run or surplus == 0:
            expired += surplus
            continue
        expired += conn.execute(
            "UPDATE recommendations SET status='expired', acted_at=? WHERE id IN ("
            " SELECT id FROM recommendations WHERE profile=? AND status='proposed'"
            " AND domain IN ('movie','series') ORDER BY score ASC, id ASC LIMIT ?)",
            (ts, profile, surplus)).rowcount
    stats["would_expire_overflow" if dry_run else "overflow_expired"] = expired
