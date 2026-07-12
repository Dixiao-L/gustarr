"""Inventory the *arrs: library contents, adds-as-signal, tag feedback.

What Sonarr/Radarr/Lidarr manage is both an exclusion list (never
recommend what's owned) and taste: a manual add is a positive
declaration, and deleting something gustarr added is a strong negative
for the profile whose recommendation landed it (household fan-out at
reduced weight only when no owner is on record).
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
    # library/items stay global (one disk, one *arr); only the taste events
    # below are per-profile. Guard for hand-built Configs without profiles.
    profiles = list(cfg.profiles) or ["default"]
    for name, spec in _SPECS.items():
        arr_cfg: ArrConfig | None = getattr(cfg, name)
        if arr_cfg is None:
            continue
        _sync_one(conn, name, spec, arr_cfg, profiles, stats)
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
    profiles: list[str],
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
        item = db.resolve_item(conn, spec["domain"], spec["ns"], str(key),
                               title=entry.get(spec["title"]), year=entry.get("year"),
                               meta={"genres": entry.get("genres", [])})
        for field, ns in spec["id_fields"].items():
            # a secondary id can reveal a twin another source minted first;
            # attach_identity merges them and hands back the survivor
            if ns != spec["ns"] and entry.get(field):
                item = db.attach_identity(conn, item, ns, str(entry[field]))
        stats["items"] += 1

        is_gustarr = gustarr_tag is not None and gustarr_tag in (entry.get("tags") or [])
        added = entry.get("added")
        status = "monitored" if entry.get("monitored") else "unmonitored"
        conn.execute(
            "INSERT INTO library (item_id, arr, arr_id, status, added_at, meta)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(item_id) DO UPDATE SET arr=excluded.arr, arr_id=excluded.arr_id,"
            " status=excluded.status, added_at=excluded.added_at, meta=excluded.meta",
            (item, name, entry.get("id"), status, added,
             json.dumps({"gustarr": True} if is_gustarr else {})),
        )
        stats["library"] += 1

        # gustarr's own adds are not taste signal — that would be feedback
        # leakage; its feedback loop is watch events + explicit approve/reject.
        # The *arr can't say WHO added it, so the event fans out to every
        # configured profile: the household owns the library, and the modest
        # library_add weight keeps the shared attribution harmless.
        if not is_gustarr and added:
            for profile in profiles:
                if db.add_event(conn, added, item, "library_add", WEIGHTS["library_add"],
                                "arr", {"arr": name}, profile=profile):
                    stats["events"] += 1
        current[ids.normalize_key(str(key))] = entry.get("id")

    state_key = f"arr:known:{name}"
    known = json.loads(db.get_state(conn, state_key, "{}"))
    # Known keys are the *arr's own external ids — merge-proof, unlike item
    # ids. v2 stores held full text item ids (domain:ns:key); reading just
    # the key tail keeps the first post-upgrade sync from mass-"removing"
    # (and mass-rejecting) the whole library once.
    gone = {ids.normalize_key(str(k).rsplit(":", 1)[-1]) for k in known} - set(current)
    for key in gone:
        stats["removed"] += 1
        item = db.lookup_item(conn, spec["domain"], spec["ns"], key)
        if item is None:
            continue
        row = conn.execute(
            "SELECT meta FROM library WHERE item_id=? AND arr=?", (item, name)).fetchone()
        if row and json.loads(row["meta"]).get("gustarr"):
            # user deleted a gustarr add: the strongest negative we ever
            # see. Unlike adds, this IS usually attributable — the
            # recommendation row that landed the item names whose queue it
            # came from, so the reject hits that profile alone at full
            # weight instead of poisoning everyone's model. Only an
            # ownerless add (pre-profile store, hand-applied tag) falls
            # back to a household fan-out, damped to 0.3x: deleting a
            # shared item is weak evidence about any individual's taste,
            # and meta flags it as shared so training can tell.
            owner = conn.execute(
                "SELECT profile FROM recommendations WHERE item_id=?"
                " AND status IN ('added','auto_added')"
                " ORDER BY acted_at DESC, id DESC LIMIT 1", (item,)).fetchone()
            targets = [owner["profile"]] if owner else profiles
            weight = WEIGHTS["reject"] if owner else WEIGHTS["reject"] * 0.3
            meta = {"deleted": True} if owner else {"deleted": True, "shared": True}
            for profile in targets:
                if db.add_event(conn, db.now(), item, "reject", weight, "arr",
                                meta, profile=profile):
                    stats["rejects"] += 1
        conn.execute("DELETE FROM library WHERE item_id=? AND arr=?", (item, name))
    db.set_state(conn, state_key, json.dumps(current))
