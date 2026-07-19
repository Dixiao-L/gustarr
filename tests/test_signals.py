"""Offline tests for signal aggregation: events carry only a scale
multiplier (schema v4); the contribution WEIGHTS[kind] * scale is priced
here at aggregation time, so a batched jellyfin scrobble event counts as
its full listen delta and retuning WEIGHTS reprices stored history."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gustarr import signals
from gustarr.signals import WEIGHTS, aggregate_label

NOW = datetime.now(timezone.utc).isoformat()


def test_batched_scrobble_scale_equals_individual_scrobbles():
    # One jellyfin-style event carrying 5 listens in its scale must
    # produce the same label as five last.fm-style scale-1 events.
    batched = [(NOW, "scrobble", 5.0)]
    individual = [(NOW, "scrobble", 1.0) for _ in range(5)]
    assert aggregate_label(batched) == pytest.approx(aggregate_label(individual))
    # and 5 listens weigh strictly more than 1
    assert aggregate_label(batched) > aggregate_label(individual[:1])


def test_scale_multiplies_the_base_for_non_log_kinds():
    # A collector storing a non-unit scale for "complete" must see the
    # base multiplied, not silently replaced by a bare WEIGHTS["complete"].
    label = aggregate_label([(NOW, "complete", 0.5)])
    assert label == pytest.approx(WEIGHTS["complete"] * 0.5 * signals.recency_factor(NOW))
    assert label != pytest.approx(WEIGHTS["complete"])


def test_all_negative_history_clamps_to_minus_one():
    # scale stays positive; the direction of each kind comes from the table
    rows = [
        (NOW, "reject", 1.0),
        (NOW, "abandon", 1.0),
        (NOW, "skip", 1.0),
    ]
    assert aggregate_label(rows) == -1.0


def test_unknown_kinds_ignored():
    assert aggregate_label([(NOW, "banana", 3.0)]) == 0.0
    # unknown kinds alongside known ones contribute nothing
    only_known = aggregate_label([(NOW, "play", 1.0)])
    mixed = aggregate_label([(NOW, "play", 1.0), (NOW, "banana", 3.0)])
    assert mixed == pytest.approx(only_known)


def test_negative_stored_scale_on_log_kind_clamped_to_zero():
    # Corrupt/negative scale on a log-scaled kind must not subtract
    # listens; the contribution clamps to zero.
    assert aggregate_label([(NOW, "scrobble", -1.0)]) == 0.0
    good = [(NOW, "scrobble", 1.0)]
    assert aggregate_label(good + [(NOW, "scrobble", -1.0)]) == pytest.approx(
        aggregate_label(good)
    )
