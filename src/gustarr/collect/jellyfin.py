"""Jellyfin collector: library items, watch/listen history, favorites.

Four passes — library state, played episodes rolled up to series, played
audio rolled up to artists, and (best-effort) the Playback Reporting
plugin. Progress signals go through state cursors so re-running the sync
adds zero events until something new is watched or played.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterator

import httpx

from .. import db, http, ids
from ..config import Config
from ..signals import WEIGHTS

SOURCE = "jellyfin"
PAGE_SIZE = 500
IDS_CHUNK = 50
# Specials/extras keep RecursiveItemCount above what anyone watches, so
# "finished the show" triggers below 100%.
SERIES_COMPLETE_FRAC = 0.8
# 40 replays of one track shouldn't drown out a completed movie.
SCROBBLE_DELTA_CAP = 5

_VIDEO_NS = {
    "Movie": ("movie", ("tmdb", "imdb")),
    "Series": ("series", ("tvdb", "tmdb", "imdb")),
}


def sync(conn: sqlite3.Connection, cfg: Config,
         transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Pull Jellyfin state into the store; returns per-signal counts."""
    jf = cfg.jellyfin
    for field in ("url", "api_key", "user"):
        if not jf.get(field):
            raise ValueError(f"jellyfin config missing {field!r}")
    base = jf["url"].rstrip("/")
    headers = {"X-Emby-Token": jf["api_key"]}
    uid = _resolve_user(base, jf["user"], headers, transport)
    stats: dict[str, Any] = {"items": 0, "skipped": 0, "favorites": 0, "completes": 0,
                             "series_plays": 0, "series_completes": 0, "scrobbles": 0,
                             "playback_reporting": 0, "pbr_scrobbles": 0, "pbr_plays": 0,
                             "pbr_completes": 0}
    _sync_library(conn, base, uid, headers, transport, stats)
    # Playback Reporting rows carry real per-play timestamps and durations;
    # when the plugin answers, they are the listening history and the
    # count-delta paths below only maintain their cursors (so removing the
    # plugin later can't burst-emit the whole backlog as one fake listen).
    pbr_active = _sync_playback_reporting(conn, base, uid, headers, transport, stats)
    _sync_series(conn, base, uid, headers, transport, stats, emit_plays=not pbr_active)
    _sync_audio(conn, base, uid, headers, transport, stats, emit=not pbr_active)
    return stats


def _resolve_user(base: str, user: str, headers: dict[str, str],
                  transport: httpx.BaseTransport | None) -> str:
    users = http.get_json(f"{base}/Users", headers=headers, transport=transport) or []
    for u in users:
        if (u.get("Name") or "").lower() == user.lower():
            return u["Id"]
    names = ", ".join(sorted(u.get("Name") or "?" for u in users)) or "none"
    raise ValueError(f"jellyfin user {user!r} not found on {base} (users: {names})")


def _paged(url: str, params: dict[str, Any], headers: dict[str, str],
           transport: httpx.BaseTransport | None) -> Iterator[dict[str, Any]]:
    start = 0
    while True:
        page = http.get_json(
            url, params={**params, "StartIndex": start, "Limit": PAGE_SIZE},
            headers=headers, transport=transport) or {}
        items = page.get("Items") or []
        yield from items
        start += len(items)
        if not items or start >= int(page.get("TotalRecordCount") or start):
            return


def _fetch_by_ids(base: str, uid: str, jf_ids: list[str], headers: dict[str, str],
                  transport: httpx.BaseTransport | None,
                  fields: str = "ProviderIds,ProductionYear") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for i in range(0, len(jf_ids), IDS_CHUNK):
        chunk = jf_ids[i:i + IDS_CHUNK]
        page = http.get_json(
            f"{base}/Users/{uid}/Items",
            params={"Ids": ",".join(chunk), "Fields": fields, "EnableImages": "false"},
            headers=headers, transport=transport) or {}
        found.extend(page.get("Items") or [])
    return found


def _canonical(raw: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    """(canonical id | None, domain, provider ids worth keeping)."""
    pids = {k.lower(): str(v).strip() for k, v in (raw.get("ProviderIds") or {}).items()
            if v and str(v).strip()}
    name = (raw.get("Name") or "").strip()
    jf_type = raw.get("Type") or ""
    if jf_type == "MusicArtist":
        if mbid := pids.get("musicbrainzartist"):
            return ids.make("artist", "mbid", mbid), "artist", {"mbid": mbid}
        if name:
            return ids.make("artist", "lastfm", name), "artist", {}
        return None, "artist", {}
    if jf_type in _VIDEO_NS:
        domain, order = _VIDEO_NS[jf_type]
        known = {ns: (int(pids[ns]) if pids[ns].isdigit() else pids[ns])
                 for ns in order if ns in pids}
        if known:
            ns, val = next(iter(known.items()))
            return ids.make(domain, ns, str(val)), domain, known
        return None, domain, known
    return None, "", {}


def _norm_ts(raw: str | None) -> str:
    # Jellyfin emits 7-digit fractional seconds ('...T20:00:00.0000000Z');
    # clamp to whole seconds so the same play always yields the same ts.
    if raw and len(raw) >= 19:
        return raw[:19] + "Z"
    return db.now()


def _merged_state(conn: sqlite3.Connection, key: str, pre_merge_key: str) -> str | None:
    """State lookup that survives an id merge: a cursor written before enrich
    merged the minted id away lives under the old key; reading it there keeps
    the first post-merge sync from re-emitting already-counted events."""
    val = db.get_state(conn, key)
    if val is None and pre_merge_key != key:
        val = db.get_state(conn, pre_merge_key)
    return val


def _flag_event(conn: sqlite3.Connection, item_id: str, kind: str, ts: str) -> bool:
    """Favorite/complete are persistent flags, not occurrences: one event per
    (item, kind) ever, so a drifting fallback ts can't break idempotency.
    Checked against the canonical id — after enrich merges the minted id away
    the flag event lives under the canonical item, and missing it here would
    re-append the flag with a fresh ts every sync."""
    item_id = db.canonical_id(conn, item_id)
    row = conn.execute(
        "SELECT 1 FROM events WHERE item_id=? AND kind=? AND source=? LIMIT 1",
        (item_id, kind, SOURCE)).fetchone()
    if row:
        return False
    return db.add_event(conn, ts, item_id, kind, WEIGHTS[kind], SOURCE)


def _sync_library(conn: sqlite3.Connection, base: str, uid: str, headers: dict[str, str],
                  transport: httpx.BaseTransport | None, stats: dict[str, Any]) -> None:
    params = {"Recursive": "true", "IncludeItemTypes": "Movie,Series,MusicArtist",
              "Fields": "ProviderIds,UserData,ProductionYear", "EnableImages": "false"}
    for raw in _paged(f"{base}/Users/{uid}/Items", params, headers, transport):
        item_id, domain, known = _canonical(raw)
        if item_id is None:
            stats["skipped"] += 1
            continue
        db.upsert_item(conn, item_id, domain, title=raw.get("Name"),
                       year=raw.get("ProductionYear"), ids=known,
                       meta={"jellyfin_id": raw.get("Id")})
        stats["items"] += 1
        ud = raw.get("UserData") or {}
        ts = _norm_ts(ud.get("LastPlayedDate"))
        if ud.get("IsFavorite") and _flag_event(conn, item_id, "favorite", ts):
            stats["favorites"] += 1
        # A Series reads Played when every *downloaded* episode is watched,
        # which over-reports partly-synced shows — episode counts decide.
        if domain == "movie" and ud.get("Played") and _flag_event(conn, item_id, "complete", ts):
            stats["completes"] += 1


def _sync_series(conn: sqlite3.Connection, base: str, uid: str, headers: dict[str, str],
                 transport: httpx.BaseTransport | None, stats: dict[str, Any],
                 emit_plays: bool = True) -> None:
    params = {"IncludeItemTypes": "Episode", "Filters": "IsPlayed", "Recursive": "true",
              "Fields": "SeriesId,ProviderIds", "EnableImages": "false"}
    played: dict[str, int] = {}
    for ep in _paged(f"{base}/Users/{uid}/Items", params, headers, transport):
        if sid := ep.get("SeriesId"):
            played[sid] = played.get(sid, 0) + 1
    for sid, count in played.items():
        fetched = _fetch_by_ids(base, uid, [sid], headers, transport,
                                fields="ProviderIds,ProductionYear,RecursiveItemCount")
        if not fetched:
            continue
        series = fetched[0]
        minted, domain, known = _canonical({**series, "Type": series.get("Type") or "Series"})
        if minted is None:
            stats["skipped"] += 1
            continue
        db.upsert_item(conn, minted, domain, title=series.get("Name"),
                       year=series.get("ProductionYear"), ids=known, meta={"jellyfin_id": sid})
        # enrich may have merged the minted id away; cursors must follow the
        # canonical id or a merge would reset them and duplicate the events
        item_id = db.canonical_id(conn, minted)
        pkey = f"jellyfin:series_played:{item_id}"
        prev = _merged_state(conn, pkey, f"jellyfin:series_played:{minted}")
        if count > int(prev or 0):
            # First-ever sight of the series is pre-plugin backlog that
            # Playback Reporting can't know about — emit it regardless.
            if (emit_plays or prev is None) and db.add_event(
                    conn, db.now(), item_id, "play", WEIGHTS["play"],
                    SOURCE, {"episodes_played": count}):
                stats["series_plays"] += 1
            db.set_state(conn, pkey, str(count))
        total = int(series.get("RecursiveItemCount") or 0)
        ckey = f"jellyfin:series_complete:{item_id}"
        if total and count >= SERIES_COMPLETE_FRAC * total \
                and _merged_state(conn, ckey, f"jellyfin:series_complete:{minted}") is None:
            if db.add_event(conn, db.now(), item_id, "complete", WEIGHTS["complete"], SOURCE,
                            {"episodes_played": count, "episodes_total": total}):
                stats["series_completes"] += 1
            db.set_state(conn, ckey, db.now())


def _sync_audio(conn: sqlite3.Connection, base: str, uid: str, headers: dict[str, str],
                transport: httpx.BaseTransport | None, stats: dict[str, Any],
                emit: bool = True) -> None:
    params = {"IncludeItemTypes": "Audio", "Filters": "IsPlayed", "Recursive": "true",
              "Fields": "ProviderIds,ArtistItems,Album,AlbumId", "EnableImages": "false"}
    tracks = list(_paged(f"{base}/Users/{uid}/Items", params, headers, transport))
    order = list(dict.fromkeys(
        t["ArtistItems"][0]["Id"] for t in tracks
        if t.get("ArtistItems") and t["ArtistItems"][0].get("Id")))
    jf_artist: dict[str, str] = {}
    for art in _fetch_by_ids(base, uid, order, headers, transport):
        art_id, domain, known = _canonical({**art, "Type": art.get("Type") or "MusicArtist"})
        if art_id is None:
            continue
        db.upsert_item(conn, art_id, domain, title=art.get("Name"),
                       year=art.get("ProductionYear"), ids=known,
                       meta={"jellyfin_id": art.get("Id")})
        jf_artist[art["Id"]] = db.canonical_id(conn, art_id)
    for t in tracks:
        arts = t.get("ArtistItems") or []
        a0 = arts[0] if arts else {}
        art_id = jf_artist.get(a0.get("Id") or "")
        if not art_id:
            name = (a0.get("Name") or "").strip()
            if not name:
                stats["skipped"] += 1
                continue
            art_id = ids.make("artist", "lastfm", name)
            db.upsert_item(conn, art_id, "artist", title=name)
            art_id = db.canonical_id(conn, art_id)
        ud = t.get("UserData") or {}
        # IsPlayed-filtered rows can still carry PlayCount 0 on some servers
        plays = max(int(ud.get("PlayCount") or 0), 1)
        key = f"jellyfin:track_plays:{t.get('Id')}"
        prev = db.get_state(conn, key)
        delta = plays - int(prev or 0)
        if delta <= 0:
            continue
        # A track with no cursor yet is pre-plugin backlog Playback
        # Reporting can't cover — bootstrap it even when PBR owns events.
        if not emit and prev is not None:
            db.set_state(conn, key, str(plays))
            continue
        weight = WEIGHTS["scrobble"] * min(delta, SCROBBLE_DELTA_CAP)
        # dedup=track id: two tracks by one artist played the same second are
        # distinct listens, not one re-synced event. A False return is then a
        # genuine replay of an already-stored row, so the cursor must hold —
        # advancing it would silently discard the still-pending plays.
        if db.add_event(conn, _norm_ts(ud.get("LastPlayedDate")), art_id, "scrobble", weight,
                        SOURCE, {"delta": delta, "track": t.get("Name"), "album": t.get("Album")},
                        dedup=str(t.get("Id") or "")):
            stats["scrobbles"] += 1
            db.set_state(conn, key, str(plays))


def _pbr_ts(raw: str) -> str:
    """PBR's DateCreated is a naive server-local string; treat it as UTC.
    The skew (one timezone) is noise against the 1-year recency half-life,
    and per-play ordering — the part that matters — is preserved."""
    return raw.strip().replace(" ", "T")[:19] + "Z"


# Watching at least this share of a movie's runtime counts as a completion.
PBR_COMPLETE_FRAC = 0.85


def _sync_playback_reporting(conn: sqlite3.Connection, base: str, uid: str,
                             headers: dict[str, str],
                             transport: httpx.BaseTransport | None,
                             stats: dict[str, Any]) -> bool:
    """Emit per-play events from the Playback Reporting plugin (the precise
    listening history: real timestamps + play durations). Returns True when
    the plugin answered, which demotes the count-delta passes to cursor
    maintenance. Rows from other Jellyfin users are ignored."""
    cursor = int(db.get_state(conn, "jellyfin:pbr_rowid", "0") or 0)
    seen = 0
    uid_norm = uid.replace("-", "").lower()
    try:
        while True:
            query = ("SELECT rowid, DateCreated, UserId, ItemId, ItemType, PlayDuration"
                     f" FROM PlaybackActivity WHERE rowid > {cursor} ORDER BY rowid LIMIT 1000")
            resp = http.post_json(
                f"{base}/user_usage_stats/submit_custom_query",
                json_body={"CustomQueryString": query, "ReplaceUserId": False},
                headers=headers, transport=transport) or {}
            # response header key is 'colums' — upstream misspelling, unused here
            rows = resp.get("results") or []
            if not rows:
                break
            seen += len(rows)
            mine = [r for r in rows
                    if str(r[2] or "").replace("-", "").lower() == uid_norm]
            resolved = {i["Id"]: i for i in _fetch_by_ids(
                base, uid, list({str(r[3]) for r in mine}), headers, transport,
                fields="ProviderIds,ArtistItems,SeriesId,Type,RunTimeTicks,Name,Album")}
            for r in mine:
                _pbr_row(conn, base, uid, headers, transport, r,
                         resolved.get(str(r[3])), stats)
            top = max(int(r[0]) for r in rows)
            if top <= cursor:  # server ignored the WHERE; don't loop forever
                break
            cursor = top
            db.set_state(conn, "jellyfin:pbr_rowid", str(cursor))
            if len(rows) < 1000:
                break
    except (http.ApiError, TypeError, ValueError):
        stats["playback_reporting"] = "unavailable"
        return False
    stats["playback_reporting"] = seen
    return True


def _pbr_row(conn: sqlite3.Connection, base: str, uid: str, headers: dict[str, str],
             transport: httpx.BaseTransport | None, row: list,
             item: dict[str, Any] | None, stats: dict[str, Any]) -> None:
    if item is None:  # deleted since it was played; nothing to attribute
        return
    rowid, ts, kind = str(row[0]), _pbr_ts(str(row[1])), str(row[4])
    duration = int(row[5] or 0)
    if kind == "Audio":
        arts = item.get("ArtistItems") or []
        a0 = arts[0] if arts else {}
        if a0.get("Id"):
            fetched = _fetch_by_ids(base, uid, [a0["Id"]], headers, transport)
            art = fetched[0] if fetched else {}
            art_id, domain, known = _canonical(
                {**art, "Type": art.get("Type") or "MusicArtist"})
            if art_id:
                db.upsert_item(conn, art_id, domain, title=art.get("Name"), ids=known,
                               meta={"jellyfin_id": a0["Id"]})
        elif name := (a0.get("Name") or "").strip():
            art_id = ids.make("artist", "lastfm", name)
            db.upsert_item(conn, art_id, "artist", title=name)
        else:
            return
        art_id = db.canonical_id(conn, art_id)
        if db.add_event(conn, ts, art_id, "scrobble", WEIGHTS["scrobble"], SOURCE,
                        {"track": item.get("Name"), "album": item.get("Album"),
                         "seconds": duration}, dedup=f"pbr{rowid}"):
            stats["pbr_scrobbles"] += 1
        return
    if kind == "Episode":
        sid = item.get("SeriesId")
        fetched = _fetch_by_ids(base, uid, [sid], headers, transport) if sid else []
        if not fetched:
            return
        series = fetched[0]
        minted, domain, known = _canonical(
            {**series, "Type": series.get("Type") or "Series"})
        if minted is None:
            return
        db.upsert_item(conn, minted, domain, title=series.get("Name"), ids=known,
                       meta={"jellyfin_id": sid})
        target = db.canonical_id(conn, minted)
    elif kind == "Movie":
        minted, domain, known = _canonical({**item, "Type": "Movie"})
        if minted is None:
            return
        db.upsert_item(conn, minted, domain, title=item.get("Name"), ids=known,
                       meta={"jellyfin_id": item.get("Id")})
        target = db.canonical_id(conn, minted)
        runtime = int(item.get("RunTimeTicks") or 0) // 10_000_000
        if runtime and duration >= PBR_COMPLETE_FRAC * runtime \
                and _flag_event(conn, target, "complete", ts):
            stats["pbr_completes"] += 1
    else:
        return
    if db.add_event(conn, ts, target, "play", WEIGHTS["play"], SOURCE,
                    {"episode": item.get("Name")} if kind == "Episode"
                    else {"seconds": duration}, dedup=f"pbr{rowid}"):
        stats["pbr_plays"] += 1
