"""Key normalization: spelling variants of one name must land on one
identities row.

Different sources hand the same CJK artist over as full-width, half-width,
case-shifted or NFD-decomposed strings; before normalize_key each variant
minted its own identity and split listening history across twins. Since
schema v3 "fallback-ness" is a store question (does the item lack an
authoritative identity?), so this file only covers the two pure exports:
normalize_key and the NS_PRIORITY/DOMAINS vocabulary.
"""

from __future__ import annotations

import unicodedata

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
    # casefold, not lower: the ß/ss pair only meets under full casefolding
    assert ids.normalize_key("Straße") == ids.normalize_key("STRASSE")


def test_whitespace_runs_collapse_and_strip():
    assert ids.normalize_key("  King \t Gnu\n") == "king gnu"
    # the ideographic space CJK sources pad with is whitespace too
    assert ids.normalize_key("King　Gnu") == "king gnu"


def test_all_folds_compose():
    # one worst-case string exercising width + case + NFD + whitespace at once
    messy = "  ＹＯＡＳＯＢＩ　×\t" + unicodedata.normalize("NFD", "ずとまよ") + " \n"
    assert ids.normalize_key(messy) == "yoasobi × ずとまよ"


def test_authoritative_ascii_keys_pass_through():
    # tmdb/mbid keys are plain ascii the fold cannot change — resolve_item
    # runs every key through normalize_key, so this must be a no-op
    mbid = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
    assert ids.normalize_key(mbid) == mbid
    assert ids.normalize_key("603") == "603"


def test_normalize_is_idempotent():
    for s in ("ＹＯＡＳＯＢＩ", "  King \t Gnu\n", "ずっと真夜中でいいのに。", "mr. oizo"):
        once = ids.normalize_key(s)
        assert ids.normalize_key(once) == once


def test_different_scripts_stay_distinct():
    # kana vs romaji is a script difference, bridged by MusicBrainz alias
    # identities under ns 'name', never by key folding
    assert ids.normalize_key("ヨルシカ") != ids.normalize_key("Yorushika")


def test_blank_input_normalizes_to_empty():
    # resolve_item's callers must treat an empty key as "no identity";
    # the fold itself just reports the emptiness rather than raising
    assert ids.normalize_key("   \t\n") == ""


# ── NS_PRIORITY / DOMAINS vocabulary ─────────────────────────────────


def test_domains_cover_the_media_types():
    assert set(ids.DOMAINS) == {"movie", "series", "artist", "album", "track"}


def test_ns_priority_covers_every_domain_exactly():
    assert set(ids.NS_PRIORITY) == set(ids.DOMAINS)


def test_ns_priority_lists_are_nonempty_unique_and_ordered():
    for domain in ids.DOMAINS:
        nss = ids.NS_PRIORITY[domain]
        assert isinstance(nss, list) and nss, domain
        assert all(isinstance(ns, str) and ns for ns in nss), domain
        # a duplicate ns would make "strongest first" ambiguous
        assert len(set(nss)) == len(nss), domain


def test_ns_priority_strongest_first():
    # enrichment and actuation both trust element 0 as the authoritative
    # namespace per domain (Radarr wants tmdb, Sonarr tvdb, Lidarr mbid)
    assert ids.NS_PRIORITY["movie"][0] == "tmdb"
    assert ids.NS_PRIORITY["series"][0] == "tvdb"
    for domain in ("artist", "album", "track"):
        assert ids.NS_PRIORITY[domain][0] == "mbid"
