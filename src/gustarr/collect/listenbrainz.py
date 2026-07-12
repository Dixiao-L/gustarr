"""ListenBrainz collector: CF recommendations + created-for playlists.

Unlike the history collectors this one writes no events — ListenBrainz's
collaborative filtering is an external recommender, so its output feeds
the candidate pool (with provenance) rather than the taste-signal store.
Items resolve through db.resolve_item on their MBIDs (spellings when the
credit carries none), so candidate rows always point at live item ids.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db
from ..config import Config
from ..http import get_json, post_json

API = "https://api.listenbrainz.org"
METADATA_CHUNK = 100  # /1/metadata/recording/ caps the batch size


def _headers(cfg: Config) -> dict[str, str]:
    # Public reads work anonymously; the token only lifts rate limits.
    token = cfg.listenbrainz.get("token")
    return {"Authorization": f"Token {token}"} if token else {}


def _upsert_candidate(
    conn: sqlite3.Connection, profile: str, item_id: int, source: str,
    score: float | None, ts: str
) -> None:
    conn.execute(
        "INSERT INTO candidates (profile, item_id, source, external_score,"
        " first_seen, last_seen) VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(profile, item_id, source) DO UPDATE SET"
        " last_seen=excluded.last_seen, external_score=excluded.external_score",
        (profile, item_id, source, score, ts, ts),
    )


def _artist_ref(conn: sqlite3.Connection,
                artist: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    """(item id, name, mbid) for the first credited artist; name-keyed
    item when there is no MBID, all-None when the credit is empty."""
    credited = artist.get("artists") or []
    first = credited[0] if credited else {}
    name = artist.get("name") or first.get("name")
    mbid = first.get("artist_mbid")
    if mbid:
        item = db.resolve_item(conn, "artist", "mbid", mbid, title=name)
        if name:  # teach the spelling so name-keyed twins converge
            item = db.attach_identity(conn, item, "name", name)
        return item, name, mbid
    if name:
        return db.resolve_item(conn, "artist", "name", name, title=name), name, None
    return None, None, None


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
    # keyed by the stable external ref, not the item id: a later row's
    # attach can merge an earlier artist away, and a stale int would make
    # the post-loop upsert write against a deleted item
    artist_best: dict[tuple[str, str], tuple[str | None, float]] = {}
    # release mbid → (title, artist name, artist mbid, best track score):
    # several CF tracks off one record are a stronger album signal than any
    # single track, so the album inherits the max.
    album_best: dict[str, tuple[str, str | None, str | None, float]] = {}
    for mbid, score in scores.items():
        info = metadata.get(mbid) or {}
        recording = info.get("recording") or {}
        artist = info.get("artist") or {}
        release = info.get("release") or {}
        try:
            artist_id, artist_name, artist_mbid = _artist_ref(conn, artist)
        except ValueError:
            # a name that folds to nothing (whitespace-only credit): one
            # junk metadata row must not fail the whole stage
            artist_id, artist_name, artist_mbid = None, None, None
        meta: dict[str, Any] = {}
        if artist_name:
            meta["artist"] = artist_name
        if release.get("name"):
            meta["album"] = release["name"]
        if release.get("mbid"):
            meta["release_mbid"] = release["mbid"]
        try:
            track_id = db.resolve_item(conn, "track", "mbid", mbid,
                                       title=recording.get("name"), meta=meta)
        except ValueError:
            continue  # junk recording mbid: skip the row, not the stage
        _upsert_candidate(conn, profile, track_id, "listenbrainz_cf", score, ts)
        stats["cf_tracks"] += 1
        numeric = float(score) if isinstance(score, (int, float)) else 0.0
        if artist_id is not None:
            ref = ("mbid", artist_mbid) if artist_mbid else ("name", artist_name or "")
            prev = artist_best.get(ref)
            if prev is None or numeric > prev[1]:
                artist_best[ref] = (artist_name, numeric)
        if release.get("mbid") and release.get("name"):
            # LB hands back RELEASE mbids, not release-groups; store as-is,
            # enrich owns upgrading them to the release-group Lidarr wants.
            aprev = album_best.get(release["mbid"])
            if aprev is None or numeric > aprev[3]:
                album_best[release["mbid"]] = (release["name"], artist_name,
                                               artist_mbid, numeric)

    # re-resolve at flush (the loop's merges may have repointed a ref) and
    # dedupe by the resolved item: one artist reached via both a name
    # credit and an mbid credit must get its max score once, not a
    # last-writer-wins overwrite and a double-counted stat
    flush_best: dict[int, float] = {}
    for (ns, key), (artist_name, best) in artist_best.items():
        artist_id = db.resolve_item(conn, "artist", ns, key, title=artist_name)
        if best > flush_best.get(artist_id, float("-inf")):
            flush_best[artist_id] = best
    for artist_id, best in flush_best.items():
        _upsert_candidate(conn, profile, artist_id, "listenbrainz_cf_artist", best, ts)
        stats["cf_artists"] += 1

    for release_mbid, (title, artist_name, artist_mbid, best) in album_best.items():
        meta = {"artist": artist_name} if artist_name else {}
        if artist_mbid:
            # the album→artist relation actuation/Lidarr needs: a pointer to
            # another item, never one of the album's own identities
            meta["artist_mbid"] = artist_mbid
        try:
            album_id = db.resolve_item(conn, "album", "mbid", release_mbid,
                                       title=title, meta=meta)
        except ValueError:
            continue
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
        try:
            track_id = db.resolve_item(conn, "track", "mbid", mbid, title=track.get("title"),
                                       meta={"artist": creator} if creator else None)
            if creator:
                # JSPF carries no artist MBID; enrich upgrades this later.
                db.resolve_item(conn, "artist", "name", creator, title=creator)
        except ValueError:
            continue  # a key folding to nothing: skip the row, not the stage
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
