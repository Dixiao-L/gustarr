"""Mirror gustarr's adds into Jellyfin "Gustarr Discover" collections so
new arrivals surface where the user actually browses.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .. import http, ids
from ..config import Config

COLLECTIONS = {
    "movie": "Gustarr Discover: Movies",
    "series": "Gustarr Discover: Series",
    "artist": "Gustarr Discover: Music",
    "album": "Gustarr Discover: Music",
    "track": "Gustarr Discover: Music",
}

# Jellyfin AnyProviderIdEquals filter: our ids-json key → provider name.
PROVIDERS = {
    "movie": ("tmdb", "tmdb"),
    "series": ("tvdb", "tvdb"),
    "artist": ("mbid", "musicbrainzartist"),
}


def external_id(row: sqlite3.Row, ns_key: str) -> str | None:
    """Provider id from items.ids json, falling back to the canonical id
    key when the item is keyed by that same namespace."""
    ext = json.loads(row["ids"]).get(ns_key)
    if ext is not None:
        return str(ext)
    _, ns, key = ids.parse(row["item_id"])
    return key if ns == ns_key else None


def _find_item(base: str, headers: dict[str, str], provider_value: str) -> str | None:
    found = http.get_json(
        f"{base}/Items",
        params={"Recursive": "true", "AnyProviderIdEquals": provider_value},
        headers=headers)
    items = (found or {}).get("Items") or []
    return items[0]["Id"] if items else None


def _find_collection(base: str, headers: dict[str, str], name: str) -> str | None:
    found = http.get_json(
        f"{base}/Items",
        params={"IncludeItemTypes": "BoxSet", "Recursive": "true", "SearchTerm": name},
        headers=headers)
    for item in (found or {}).get("Items") or []:
        if item.get("Name") == name:
            return item["Id"]
    return None


def _members(base: str, headers: dict[str, str], collection_id: str) -> set[str]:
    found = http.get_json(
        f"{base}/Items", params={"ParentId": collection_id}, headers=headers)
    return {item["Id"] for item in (found or {}).get("Items") or []}


def sync_collections(
    conn: sqlite3.Connection, cfg: Config, dry_run: bool = False
) -> dict[str, Any]:
    base = (cfg.jellyfin.get("url") or "").rstrip("/")
    token = cfg.jellyfin.get("api_key")
    if not base or not token:
        return {"skipped": True}
    rows = conn.execute(
        "SELECT DISTINCT r.domain, i.id AS item_id, i.ids, i.title FROM recommendations r"
        " JOIN items i ON i.id = r.item_id WHERE r.status IN ('auto_added','added')"
    ).fetchall()
    if dry_run:
        return {"would_sync": len(rows)}

    headers = {"X-Emby-Token": token}
    stats = {"checked": 0, "matched": 0, "collections_created": 0, "collection_adds": 0}
    wanted: dict[str, list[str]] = {}
    for row in rows:
        provider = PROVIDERS.get(row["domain"])
        name = COLLECTIONS.get(row["domain"])
        if provider is None or name is None:
            continue
        ext = external_id(row, provider[0])
        if ext is None:
            continue
        stats["checked"] += 1
        jf_id = _find_item(base, headers, f"{provider[1]}.{ext}")
        if jf_id is None:
            continue
        stats["matched"] += 1
        bucket = wanted.setdefault(name, [])
        if jf_id not in bucket:
            bucket.append(jf_id)

    for name, jf_ids in wanted.items():
        collection_id = _find_collection(base, headers, name)
        if collection_id is None:
            http.post_json(
                f"{base}/Collections",
                params={"Name": name, "Ids": ",".join(jf_ids)}, headers=headers)
            stats["collections_created"] += 1
            stats["collection_adds"] += len(jf_ids)
            continue
        new = [i for i in jf_ids if i not in _members(base, headers, collection_id)]
        if new:
            http.post_json(
                f"{base}/Collections/{collection_id}/Items",
                params={"Ids": ",".join(new)}, headers=headers)
            stats["collection_adds"] += len(new)
    return stats
