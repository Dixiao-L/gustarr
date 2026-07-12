"""Identity vocabulary and key normalization.

Items carry surrogate integer ids; every external id or name spelling an
item is known by lives in the ``identities`` table (see ``db.resolve_item``
/ ``db.attach_identity``). This module owns the two policies those share:
which namespaces exist per domain (and their authority order), and how a
key is folded before it is compared.
"""

from __future__ import annotations

import unicodedata

DOMAINS = ("movie", "series", "artist", "album", "track")

# Authority order per domain, strongest first. When identities of two
# namespaces claim the same item during a merge, the holder of the
# stronger namespace wins; an item whose identities include none of the
# non-'name' namespaces is pending — enrich tries to attach one.
NS_PRIORITY = {
    "movie": ["tmdb", "imdb"],
    "series": ["tvdb", "tmdb", "imdb"],
    "artist": ["mbid", "name"],
    "album": ["mbid", "name"],
    "track": ["mbid", "name"],
}


def normalize_key(s: str) -> str:
    """Fold the spelling variants one name arrives under into one key:
    NFKC (full/half-width, decomposed kana), casefold, then collapse
    whitespace runs. Different sources hand us ＹＯＡＳＯＢＩ / YOASOBI /
    NFD kana for the SAME artist; if each minted its own identity, one
    person's listening history would split across duplicate items. Kana
    vs romaji is a different *script*, not a width/case variant — those
    are bridged by MusicBrainz aliases arriving as extra 'name'
    identities, not folded here.
    """
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())
