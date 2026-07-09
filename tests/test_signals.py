"""Offline tests for signal aggregation: stored event weight must reach
the label (db.py documents weight as "the label contribution"), so a
batched jellyfin scrobble event counts as its full listen delta."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gustarr import signals
from gustarr.signals import WEIGHTS, aggregate_label

NOW = datetime.now(timezone.utc).isoformat()


def test_batched_scrobble_weight_equals_individual_scrobbles():
    # One jellyfin-style event carrying 5 listens in its weight must
    # produce the same label as five last.fm-style unit-weight events.
    batched = [(NOW, "scrobble", WEIGHTS["scrobble"] * 5)]
    individual = [(NOW, "scrobble", WEIGHTS["scrobble"]) for _ in range(5)]
    assert aggregate_label(batched) == pytest.approx(aggregate_label(individual))
    # and 5 listens weigh strictly more than 1
    assert aggregate_label(batched) > aggregate_label(individual[:1])


def test_stored_weight_overrides_base_for_non_log_kinds():
    # A collector storing a non-default weight for "complete" must see it
    # honored, not silently replaced by WEIGHTS["complete"].
    label = aggregate_label([(NOW, "complete", 0.5)])
    assert label == pytest.approx(0.5 * signals.recency_factor(NOW))
    assert label != pytest.approx(WEIGHTS["complete"])


def test_all_negative_history_clamps_to_minus_one():
    rows = [
        (NOW, "reject", WEIGHTS["reject"]),
        (NOW, "abandon", WEIGHTS["abandon"]),
        (NOW, "skip", WEIGHTS["skip"]),
    ]
    assert aggregate_label(rows) == -1.0


def test_unknown_kinds_ignored():
    assert aggregate_label([(NOW, "banana", 3.0)]) == 0.0
    # unknown kinds alongside known ones contribute nothing
    only_known = aggregate_label([(NOW, "play", WEIGHTS["play"])])
    mixed = aggregate_label([(NOW, "play", WEIGHTS["play"]), (NOW, "banana", 3.0)])
    assert mixed == pytest.approx(only_known)


def test_negative_stored_weight_on_log_kind_clamped_to_zero():
    # Corrupt/negative weight on a log-scaled kind must not subtract
    # listens; the contribution clamps to zero.
    assert aggregate_label([(NOW, "scrobble", -1.0)]) == 0.0
    good = [(NOW, "scrobble", WEIGHTS["scrobble"])]
    assert aggregate_label(good + [(NOW, "scrobble", -1.0)]) == pytest.approx(
        aggregate_label(good)
    )
