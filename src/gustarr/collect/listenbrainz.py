"""ListenBrainz collector: CF recommendations + created-for playlists.

Unlike the history collectors this one writes no events — ListenBrainz's
collaborative filtering is an external recommender, so its output feeds
the candidate pool (with provenance) rather than the taste-signal store.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db, ids
from ..config import Config
from ..http import get_json, post_json

API = "https://api.listenbrainz.org"
METADATA_CHUNK = 100  # /1/metadata/recording/ caps the batch size


def _headers(cfg: Config) -> dict[str, str]:
    # Public reads work anonymously; the token only lifts rate limits.
    token = cfg.listenbrainz.get("token")
    return {"Authorization": f"Token {token}"} if token else {}


def _upsert_candidate(
    conn: sqlite3.Connection, profile: str, item_id: str, source: str,
    score: float | None, ts: str
) -> None:
    conn.execute(
        "INSERT INTO candidates (profile, item_id, source, external_score,"
        " first_seen, last_seen) VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(profile, item_id, source) DO UPDATE SET"
        " last_seen=excluded.last_seen, external_score=excluded.external_score",
        (profile, item_id, source, score, ts, ts),
    )


def _artist_ref(artist: dict[str, Any]) -> tuple[str | None, str | None]:
    """(item_id, name) for the first credited artist; lastfm fallback when no MBID."""
    credited = artist.get("artists") or []
    first = credited[0] if credited else {}
    name = artist.get("name") or first.get("name")
    mbid = first.get("artist_mbid")
    if mbid:
        return ids.make("artist", "mbid", mbid), name
    if name:
        return ids.make("artist", "lastfm", name), name
    return None, None


def _upsert_artist(conn: sqlite3.Connection, artist_id: str, name: str | None) -> None:
    _, ns, key = ids.parse(artist_id)
    db.upsert_item(conn, artist_id, "artist", title=name,
                   ids={"mbid": key} if ns == "mbid" else {})


def _fetch_metadata(mbids: list[str], headers: dict[str, str]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for i in range(0, len(mbids), METADATA_CHUNK):
        chunk = mbids[i : i + METADATA_CHUNK]
        resp = post_json(
            f"{API}/1/metadata/recording/",
            json_body={"recording_mbids": chunk, "inc": "artist release"},
            headers=headers,
        )
        if isinstance(resp, dict):
            resolved.update(resp)
    return resolved


def _sync_cf(conn: sqlite3.Connection, profile: str, user: str, headers: dict[str, str],
             stats: dict[str, Any]) -> None:
    data = get_json(f"{API}/1/cf/recommendation/user/{user}/recording",
                    params={"count": 200}, headers=headers)
    scores: dict[str, float | None] = {}
    for entry in ((data or {}).get("payload") or {}).get("mbids") or []:
        mbid = entry.get("recording_mbid")
        if mbid:
            scores[mbid] = entry.get("score")
    if not scores:
        # New accounts get 204 until LB's CF pipeline has seen enough listens.
        stats["cf"] = "not_ready"
        return

    metadata = _fetch_metadata(list(scores), headers)
    ts = db.now()
    artist_best: dict[str, tuple[str | None, float]] = {}
    # album id → (title, artist_name, artist_id, best track score): several
    # CF tracks off one record are a stronger album signal than any single
    # track, so the album inherits the max.
    album_best: dict[str, tuple[str, str | None, str | None, float]] = {}
    for mbid, score in scores.items():
        info = metadata.get(mbid) or {}
        recording = info.get("recording") or {}
        artist = info.get("artist") or {}
        release = info.get("release") or {}
        artist_id, artist_name = _artist_ref(artist)
        meta: dict[str, Any] = {}
        if artist_name:
            meta["artist"] = artist_name
        if release.get("name"):
            meta["album"] = release["name"]
        if release.get("mbid"):
            meta["release_mbid"] = release["mbid"]
        track_id = ids.make("track", "mbid", mbid)
        db.upsert_item(conn, track_id, "track", title=recording.get("name"),
                       ids={"mbid": mbid}, meta=meta)
        _upsert_candidate(conn, profile, track_id, "listenbrainz_cf", score, ts)
        stats["cf_tracks"] += 1
        numeric = float(score) if isinstance(score, (int, float)) else 0.0
        if artist_id:
            _upsert_artist(conn, artist_id, artist_name)
            prev = artist_best.get(artist_id)
            if prev is None or numeric > prev[1]:
                artist_best[artist_id] = (artist_name, numeric)
        if release.get("mbid") and release.get("name"):
            # LB hands back RELEASE mbids, not release-groups; store as-is,
            # enrich owns upgrading them to the release-group Lidarr wants.
            album_id = ids.make("album", "mbid", release["mbid"])
            aprev = album_best.get(album_id)
            if aprev is None or numeric > aprev[3]:
                album_best[album_id] = (release["name"], artist_name, artist_id, numeric)

    for artist_id, (_, best) in artist_best.items():
        _upsert_candidate(conn, profile, artist_id, "listenbrainz_cf_artist", best, ts)
        stats["cf_artists"] += 1

    for album_id, (title, artist_name, artist_id, best) in album_best.items():
        ids_d: dict[str, Any] = {"mbid": ids.parse(album_id)[2]}
        meta = {"artist": artist_name} if artist_name else {}
        if artist_id:
            _, ns, key = ids.parse(artist_id)
            if ns == "mbid":
                # ids for the actuate/Lidarr contract, meta for the UI/enrich.
                ids_d["artist_mbid"] = key
                meta["artist_mbid"] = key
        db.upsert_item(conn, album_id, "album", title=title, ids=ids_d, meta=meta)
        _upsert_candidate(conn, profile, album_id, "listenbrainz_cf_album", best, ts)
        stats["cf_albums"] += 1


def _recording_mbid(track: dict[str, Any]) -> str | None:
    ident = track.get("identifier") or ""
    if isinstance(ident, list):  # JSPF allows one identifier or a list
        ident = ident[0] if ident else ""
    mbid = str(ident).rstrip("/").rsplit("/", 1)[-1]
    return mbid or None


def _sync_weekly(conn: sqlite3.Connection, profile: str, user: str, headers: dict[str, str],
                 stats: dict[str, Any]) -> None:
    listing = get_json(f"{API}/1/user/{user}/playlists/createdfor", headers=headers) or {}
    weekly = [
        wrapper.get("playlist") or {}
        for wrapper in listing.get("playlists") or []
        if "Weekly Exploration" in ((wrapper.get("playlist") or {}).get("title") or "")
    ]
    if not weekly:
        return
    newest = max(weekly, key=lambda p: p.get("date") or "")
    playlist_mbid = str(newest.get("identifier") or "").rstrip("/").rsplit("/", 1)[-1]
    if not playlist_mbid:
        return
    data = get_json(f"{API}/1/playlists/{playlist_mbid}", headers=headers) or {}
    ts = db.now()
    for track in (data.get("playlist") or {}).get("track") or []:
        mbid = _recording_mbid(track)
        if not mbid:
            continue
        creator = track.get("creator")
        track_id = ids.make("track", "mbid", mbid)
        db.upsert_item(conn, track_id, "track", title=track.get("title"),
                       ids={"mbid": mbid}, meta={"artist": creator} if creator else {})
        if creator:
            # JSPF carries no artist MBID; enrich upgrades this later.
            _upsert_artist(conn, ids.make("artist", "lastfm", creator), creator)
        _upsert_candidate(conn, profile, track_id, "listenbrainz_weekly", None, ts)
        stats["weekly_tracks"] += 1


def sync(conn: sqlite3.Connection, cfg: Config) -> dict[str, Any]:
    """CF recommendations + Weekly Exploration playlist → candidates rows,
    once per profile with a listenbrainz_user, each under its own profile.
    Counts stay flat totals across profiles."""
    users = {name: p.listenbrainz_user
             for name, p in cfg.profiles.items() if p.listenbrainz_user}
    if not users:
        return {"skipped": "listenbrainz not configured"}
    headers = _headers(cfg)
    stats: dict[str, Any] = {"cf_tracks": 0, "cf_artists": 0, "cf_albums": 0,
                             "weekly_tracks": 0, "profiles": 0,
                             "profiles_skipped": len(cfg.profiles) - len(users)}
    for profile, user in users.items():
        stats["profiles"] += 1
        _sync_cf(conn, profile, user, headers, stats)
        try:
            _sync_weekly(conn, profile, user, headers, stats)
        except Exception:
            # Created-for playlists are a bonus signal; absence or API churn
            # must never fail the sync.
            stats["weekly"] = "unavailable"
    return stats
