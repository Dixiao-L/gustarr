"""Key normalization: spelling variants of one name must mint one id.

Different sources hand the same CJK artist over as full-width, half-width,
case-shifted or NFD-decomposed strings; before normalize_key each variant
minted its own fallback id and split listening history across twins.
"""

from __future__ import annotations

import unicodedata

import pytest

from gustarr import ids


# ── normalize_key ────────────────────────────────────────────────────


def test_fullwidth_latin_folds_to_ascii_lowercase():
    assert ids.normalize_key("ＹＯＡＳＯＢＩ") == "yoasobi"
    assert ids.normalize_key("YOASOBI") == "yoasobi"


def test_halfwidth_katakana_folds_to_fullwidth():
    assert ids.normalize_key("ﾖﾙｼｶ") == ids.normalize_key("ヨルシカ")


def test_nfc_and_nfd_forms_collide():
    nfc = "ヨルシカ"
    assert ids.normalize_key(unicodedata.normalize("NFD", nfc)) == ids.normalize_key(nfc)
    # a dakuten name actually differs between forms, proving the fold matters
    dakuten = "ずっと真夜中でいいのに。"
    nfd = unicodedata.normalize("NFD", dakuten)
    assert nfd != dakuten
    assert ids.normalize_key(nfd) == ids.normalize_key(dakuten)


def test_case_variants_collide():
    assert ids.normalize_key("Mr. OIZO") == ids.normalize_key("mr. oizo")


def test_whitespace_runs_collapse_and_strip():
    assert ids.normalize_key("  King \t Gnu\n") == "king gnu"


def test_different_scripts_stay_distinct():
    # kana vs romaji is a script difference, bridged by MB aliases
    # (item_aliases), never by key folding
    assert ids.normalize_key("ヨルシカ") != ids.normalize_key("Yorushika")


# ── make: every key part routes through normalize_key ────────────────


def test_make_folds_name_variants_to_one_fallback_id():
    a = ids.make("artist", "lastfm", "ＹＯＡＳＯＢＩ")
    b = ids.make("artist", "lastfm", "yoasobi")
    c = ids.make("artist", "lastfm", " YOASOBI ")
    assert a == b == c == "artist:lastfm:yoasobi"


def test_make_folds_every_part_of_multipart_keys():
    a = ids.make("track", "lastfm", "Radiohead", "Paranoid  Android")
    b = ids.make("track", "lastfm", "radiohead", "paranoid android")
    assert a == b


def test_make_leaves_authoritative_ascii_ids_unchanged():
    mbid = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
    assert ids.make("artist", "mbid", mbid) == f"artist:mbid:{mbid}"
    assert ids.make("movie", "tmdb", "603") == "movie:tmdb:603"


def test_make_rejects_keys_that_normalize_to_nothing():
    with pytest.raises(ValueError):
        ids.make("artist", "lastfm", "   ")


def test_make_rejects_unknown_domain():
    with pytest.raises(ValueError):
        ids.make("book", "lastfm", "x")


# ── parse / is_fallback round-trips ──────────────────────────────────


def test_parse_and_is_fallback():
    iid = ids.make("artist", "lastfm", "ヨルシカ")
    assert ids.parse(iid) == ("artist", "lastfm", "ヨルシカ")
    assert ids.is_fallback(iid)
    assert not ids.is_fallback(ids.make("artist", "mbid", "abc"))
