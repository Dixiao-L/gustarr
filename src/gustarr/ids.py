"""Canonical item identifiers.

Every item in the store has exactly one id of the form ``domain:ns:key``:

    movie:tmdb:603            series:tvdb:81189
    artist:mbid:5b11f4ce-...  album:mbid:1b022e01-...
    track:lastfm:radiohead\x1fparanoid android   (fallback when no MBID)

The namespace is the *most authoritative* id we have for that item. When a
collector only knows a weaker id (e.g. a Last.fm artist name), it mints the
fallback form; ``enrich`` later resolves it to the authoritative namespace
and merges the rows (see ``db.merge_item``).
"""

from __future__ import annotations

DOMAINS = ("movie", "series", "artist", "album", "track")

# Preferred namespace per domain, strongest first. Fallback namespaces
# (name-keyed) sort last so enrichment knows an upgrade is possible.
NS_PRIORITY = {
    "movie": ["tmdb", "imdb"],
    "series": ["tvdb", "tmdb", "imdb"],
    "artist": ["mbid", "lastfm"],
    "album": ["mbid", "lastfm"],
    "track": ["mbid", "lastfm"],
}

_SEP = "\x1f"  # unit separator: never occurs in artist/track names


def make(domain: str, ns: str, *key_parts: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    if not key_parts or not all(key_parts):
        raise ValueError(f"empty key for {domain}:{ns}")
    key = _SEP.join(str(p).strip().lower() for p in key_parts)
    return f"{domain}:{ns}:{key}"


def parse(item_id: str) -> tuple[str, str, str]:
    domain, ns, key = item_id.split(":", 2)
    return domain, ns, key


def is_fallback(item_id: str) -> bool:
    """True when the id is name-keyed and should be upgraded by enrich."""
    domain, ns, _ = parse(item_id)
    return ns != NS_PRIORITY[domain][0]
