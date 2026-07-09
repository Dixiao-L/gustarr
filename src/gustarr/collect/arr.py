"""Inventory the *arrs: library contents, adds-as-signal, tag feedback.

What Sonarr/Radarr/Lidarr manage is both an exclusion list (never
recommend what's owned) and taste: a manual add is a positive
declaration, and deleting something gustarr added is a strong negative.
Items carrying the gustarr tag get no library_add event — rewarding our
own adds would feed the model its own output.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import db, http, ids
from ..config import ArrConfig, Config
from ..signals import WEIGHTS

# Per-arr API shape: version prefix, list endpoint, canonical domain and
# namespace, plus which response fields map to which id namespaces.
_SPECS: dict[str, dict[str, Any]] = {
    "radarr": {
        "ver": "v3", "endpoint": "movie", "domain": "movie", "ns": "tmdb",
        "key": "tmdbId", "title": "title",
        "id_fields": {"tmdbId": "tmdb", "imdbId": "imdb"},
    },
    "sonarr": {
        "ver": "v3", "endpoint": "series", "domain": "series", "ns": "tvdb",
        "key": "tvdbId", "title": "title",
        "id_fields": {"tvdbId": "tvdb", "tmdbId": "tmdb", "imdbId": "imdb"},
    },
    "lidarr": {
        "ver": "v1", "endpoint": "artist", "domain": "artist", "ns": "mbid",
        "key": "foreignArtistId", "title": "artistName",
        "id_fields": {"foreignArtistId": "mbid"},
    },
}


def sync(conn: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    stats = {"items": 0, "library": 0, "events": 0, "rejects": 0, "removed": 0, "skipped": 0}
    for name, spec in _SPECS.items():
        arr_cfg: ArrConfig | None = getattr(cfg, name)
        if arr_cfg is None:
            continue
        _sync_one(conn, name, spec, arr_cfg, stats)
    return stats


def _tag_id(base: str, ver: str, headers: dict[str, str], label: str) -> int | None:
    tags = http.get_json(f"{base}/api/{ver}/tag", headers=headers) or []
    for tag in tags:
        # Servarr lowercases tag labels on creation; compare accordingly.
        if str(tag.get("label", "")).lower() == label.lower():
            return tag.get("id")
    return None


def _sync_one(
    conn: sqlite3.Connection,
    name: str,
    spec: dict[str, Any],
    arr_cfg: ArrConfig,
    stats: dict[str, int],
) -> None:
    base = arr_cfg.url.rstrip("/")
    ver = spec["ver"]
    headers = {"X-Api-Key": arr_cfg.api_key}
    gustarr_tag = _tag_id(base, ver, headers, arr_cfg.tag)
    entries = http.get_json(f"{base}/api/{ver}/{spec['endpoint']}", headers=headers) or []

    current: dict[str, int | None] = {}
    for entry in entries:
        key = entry.get(spec["key"])
        if not key:
            stats["skipped"] += 1
            continue
        item_id = ids.make(spec["domain"], spec["ns"], str(key))
        ext_ids = {ns: entry[f] for f, ns in spec["id_fields"].items() if entry.get(f)}
        db.upsert_item(conn, item_id, spec["domain"], title=entry.get(spec["title"]),
                       year=entry.get("year"), ids=ext_ids,
                       meta={"genres": entry.get("genres", [])})
        stats["items"] += 1

        is_gustarr = gustarr_tag is not None and gustarr_tag in (entry.get("tags") or [])
        added = entry.get("added")
        status = "monitored" if entry.get("monitored") else "unmonitored"
        conn.execute(
            "INSERT INTO library (item_id, arr, arr_id, status, added_at, meta)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(item_id) DO UPDATE SET arr=excluded.arr, arr_id=excluded.arr_id,"
            " status=excluded.status, added_at=excluded.added_at, meta=excluded.meta",
            (item_id, name, entry.get("id"), status, added,
             json.dumps({"gustarr": True} if is_gustarr else {})),
        )
        stats["library"] += 1

        # gustarr's own adds are not taste signal — that would be feedback
        # leakage; its feedback loop is watch events + explicit approve/reject.
        if not is_gustarr and added:
            if db.add_event(conn, added, item_id, "library_add", WEIGHTS["library_add"],
                            "arr", {"arr": name}):
                stats["events"] += 1
        current[item_id] = entry.get("id")

    state_key = f"arr:known:{name}"
    known = json.loads(db.get_state(conn, state_key, "{}"))
    for item_id in set(known) - set(current):
        row = conn.execute(
            "SELECT meta FROM library WHERE item_id=? AND arr=?", (item_id, name)).fetchone()
        if row and json.loads(row["meta"]).get("gustarr"):
            # user deleted a gustarr add: the strongest negative we ever see
            if db.add_event(conn, db.now(), item_id, "reject", WEIGHTS["reject"], "arr",
                            {"deleted": True}):
                stats["rejects"] += 1
        conn.execute("DELETE FROM library WHERE item_id=? AND arr=?", (item_id, name))
        stats["removed"] += 1
    db.set_state(conn, state_key, json.dumps(current))
