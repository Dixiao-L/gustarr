"""Human-asserted MusicBrainz ids for artists that exist only as a spelling.

Enrich refuses to adopt a fuzzy MusicBrainz match (MB_MERGE_SCORE), and
rightly so — but the refusal leaves some artists living on as a bare
name identity, holding real listening history that can never merge with
a properly-identified twin. Matching those is the one identity call
Gustarr deliberately never guesses at; this module makes it ergonomic
for the person who actually knows. pending() lists the artists worth a
human's minute — identity is global, so they rank by the history they
hold across every profile; search() asks MusicBrainz; assert_mbid()
records the verdict through db.attach_identity, so the existing merge
rules heal the split history, and reopens enrichment so the next run
fetches MB's canonical name, tags and aliases for the survivor.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from . import db, http

MB = "https://musicbrainz.org/ws/2"

# A MusicBrainz id is a UUID: 8-4-4-4-12 hex, case-insensitive. Anything
# else — a pasted artist name, a truncated copy — is a typo to refuse
# loudly, not an identity to record.
MBID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def pending(conn: sqlite3.Connection, limit: int = 25) -> list[dict[str, Any]]:
    """Artists with no mbid identity, ranked by the listening history they
    hold. Event counts span all profiles: an identity assertion changes
    the shared store, so the ranking must weigh everyone's history. One
    aggregated query — a correlated per-artist COUNT once re-scanned
    events for every artist row in the store."""
    rows = conn.execute(
        "SELECT i.id, i.title, COUNT(e.id) AS events"
        " FROM items i LEFT JOIN events e ON e.item_id = i.id"
        " WHERE i.domain='artist' AND NOT EXISTS("
        "  SELECT 1 FROM identities x WHERE x.item_id = i.id AND x.ns='mbid')"
        " GROUP BY i.id ORDER BY events DESC, i.id LIMIT ?", (limit,)).fetchall()
    return [{
        "item_id": r["id"],
        "title": r["title"],
        # every spelling, not identities_of's first-per-ns — the human
        # matches on the odd romanizations, so all of them must show
        "spellings": [s["key"] for s in conn.execute(
            "SELECT key FROM identities WHERE item_id=? AND ns='name' ORDER BY rowid",
            (r["id"],))],
        "events": r["events"],
    } for r in rows]


def search(name: str, limit: int = 8) -> list[dict[str, Any]]:
    """MusicBrainz artist search — the same endpoint (and, via http's
    per-host delay, the same politeness) as enrich's automated lookup,
    but returning the ranked hits for a person to judge instead of
    applying a score gate. Quotes are stripped so the name can't break
    out of the phrase query."""
    query = f'artist:"{name.replace(chr(34), "")}"'
    data = http.get_json(f"{MB}/artist", params={"query": query, "fmt": "json", "limit": limit})
    return [{
        "id": hit["id"],
        "name": hit.get("name"),
        # empty string, not None: the UI concatenates it into a caption
        "disambiguation": hit.get("disambiguation") or "",
        "type": hit.get("type"),
        "country": hit.get("country"),
        "score": int(hit.get("score") or 0),
    } for hit in (data or {}).get("artists") or [] if isinstance(hit, dict) and hit.get("id")]


def assert_mbid(conn: sqlite3.Connection, item_id: int, mbid: str) -> int:
    """Record the human's verdict: this spelling IS that MusicBrainz
    artist. The attach goes through db.attach_identity, so when another
    item already holds the mbid the split history merges under the
    existing rules and the survivor's id is returned (callers continue
    with it). Either way the survivor's enriched_at is cleared so the
    next enrich fetches MB's canonical data — including the alias list
    that bridges future spellings.

    The cross-entity refusal rule holds on the human path too: an artist
    that already carries its own mbid is a different entity from whatever
    the asserted id names, and a typo'd assert must never force-merge two
    authoritative artists — refused loudly, nothing written."""
    if not MBID_RE.fullmatch(mbid or ""):
        raise ValueError(
            f"{mbid!r} is not a MusicBrainz id (a UUID like"
            " 53f7a13b-0f75-4b31-9c30-2ed40de6dc35); paste the id, not the name")
    row = conn.execute("SELECT domain, title FROM items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        raise ValueError(f"no item #{item_id}")
    if row["domain"] != "artist":
        raise ValueError(f"item #{item_id} is a {row['domain']}, not an artist")
    held = db.identities_of(conn, item_id).get("mbid")
    if held is not None:
        raise ValueError(
            f"artist #{item_id} ({row['title'] or 'untitled'}) already has MusicBrainz id"
            f" {held}; identify only asserts ids for artists that lack one")
    survivor = db.attach_identity(conn, item_id, "mbid", mbid)
    conn.execute("UPDATE items SET enriched_at=NULL WHERE id=?", (survivor,))
    return survivor
