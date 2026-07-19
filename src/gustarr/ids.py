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

# Multipart-key separator (tracks: artist SEP title). Never occurs in
# artist or track names; normalize_key preserves it as a part boundary.
SEP = "\x1f"

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

    The unit separator (\\x1f) survives as a part boundary: track name
    keys are "artist\\x1ftitle", and folding it into a space would let
    two different tracks collide whenever the artist/title split shifts
    ("A B"+"C" vs "A"+"B C"). Python counts \\x1f as whitespace, hence
    the explicit part-wise fold.
    """
    parts = [" ".join(unicodedata.normalize("NFKC", part).casefold().split())
             for part in s.split(SEP)]
    return SEP.join(parts) if any(parts) else ""


def spaceless(s: str) -> str:
    """normalize_key with every space removed: the LOOKUP fold behind
    alias twin-hunting. Scrobblers hand over spellings that drop the
    spaces MusicBrainz's alias carries ("KinokoTeikoku" for "Kinoko
    Teikoku"); once whitespace is folded out the two are the same exact
    string — a deterministic fold, never a fuzzy match. Never a storage
    key: identity rows stay normalize_key'd (spaced spellings keep their
    own rows), so this fold only decides where a hunt LOOKS, never what
    is written. After normalize_key the only whitespace left is single
    ASCII spaces, so the replace covers every variant the fold saw; the
    \\x1f part boundary survives untouched, keeping multipart track keys
    unfusable even though the hunt itself is artist-only."""
    return normalize_key(s).replace(" ", "")
