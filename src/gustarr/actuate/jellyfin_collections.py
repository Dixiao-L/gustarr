"""Mirror gustarr's adds into Jellyfin "Gustarr Discover" collections so
new arrivals surface where the user actually browses.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db, http
from ..config import Config

COLLECTIONS = {
    "movie": "Gustarr Discover: Movies",
    "series": "Gustarr Discover: Series",
    "artist": "Gustarr Discover: Music",
    "album": "Gustarr Discover: Music",
    "track": "Gustarr Discover: Music",
}

# our identity namespace → Jellyfin ProviderIds key + item type, for
# items that carry no 'jellyfin' identity yet (the collector hands us
# the Jellyfin item id directly for library items).
PROVIDERS = {
    "movie": ("tmdb", "tmdb", "Movie"),
    "series": ("tvdb", "tvdb", "Series"),
    "artist": ("mbid", "musicbrainzartist", "MusicArtist"),
}


def _find_item(base: str, headers: dict[str, str], item_type: str,
               provider: str, ext: str, title: str | None) -> str | None:
    """Jellyfin 10.x has no provider-id query (the Emby-era
    AnyProviderIdEquals parameter is silently ignored, which would make
    every lookup 'match' an arbitrary item), so search by title and
    verify the provider id client-side."""
    if not title:
        return None
    found = http.get_json(
        f"{base}/Items",
        params={"Recursive": "true", "IncludeItemTypes": item_type,
                "SearchTerm": title, "Fields": "ProviderIds", "Limit": 25},
        headers=headers)
    want = str(ext).casefold()
    for item in (found or {}).get("Items") or []:
        pids = {k.casefold(): str(v).casefold()
                for k, v in (item.get("ProviderIds") or {}).items() if v}
        if pids.get(provider) == want:
            return item["Id"]
    return None


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
        "SELECT DISTINCT r.domain, i.id AS item_id, i.title FROM recommendations r"
        " JOIN items i ON i.id = r.item_id WHERE r.status IN ('auto_added','added')"
    ).fetchall()
    if dry_run:
        return {"would_sync": len(rows)}

    headers = {"X-Emby-Token": token}
    stats = {"checked": 0, "matched": 0, "collections_created": 0, "collection_adds": 0}
    wanted: dict[str, list[str]] = {}
    for row in rows:
        name = COLLECTIONS.get(row["domain"])
        if name is None:
            continue
        idents = db.identities_of(conn, row["item_id"])
        # A 'jellyfin' identity IS the Jellyfin item id (stored lowercase,
        # matching Jellyfin's own hex ids): use it directly and skip the
        # provider-id search round-trip entirely.
        jf_id = idents.get("jellyfin")
        if jf_id is not None:
            stats["checked"] += 1
            stats["matched"] += 1
        else:
            provider = PROVIDERS.get(row["domain"])
            if provider is None:
                continue
            ns, provider_key, item_type = provider
            ext = idents.get(ns)
            if ext is None:
                continue
            stats["checked"] += 1
            jf_id = _find_item(base, headers, item_type, provider_key, ext, row["title"])
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
