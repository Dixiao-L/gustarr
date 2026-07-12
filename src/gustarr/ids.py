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

import unicodedata

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


def normalize_key(s: str) -> str:
    """Fold the spelling variants one name arrives under into one key:
    NFKC (full/half-width, decomposed kana), casefold, then collapse
    whitespace runs. Different sources hand us ＹＯＡＳＯＢＩ / YOASOBI /
    NFD kana for the SAME artist; if each minted its own fallback id, one
    person's listening history would split across duplicate items. Kana
    vs romaji is a different *script*, not a width/case variant — those
    are bridged by MusicBrainz aliases (item_aliases), not folded here.
    """
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def make(domain: str, ns: str, *key_parts: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    # Every part goes through normalize_key — authoritative keys (mbid,
    # tmdb, ...) are plain ascii it can't change, and one uniform rule
    # beats a per-namespace special case.
    parts = [normalize_key(str(p)) for p in key_parts]
    if not parts or not all(parts):
        raise ValueError(f"empty key for {domain}:{ns}")
    return f"{domain}:{ns}:{_SEP.join(parts)}"


def parse(item_id: str) -> tuple[str, str, str]:
    domain, ns, key = item_id.split(":", 2)
    return domain, ns, key


def is_fallback(item_id: str) -> bool:
    """True when the id is name-keyed and should be upgraded by enrich."""
    domain, ns, _ = parse(item_id)
    return ns != NS_PRIORITY[domain][0]
