"""Taste-signal weighting: how raw events become training labels.

Single place of truth so collectors stay dumb (they record *what
happened*) and the model stays honest (label policy is reviewable here,
not scattered).

Events store only a scale multiplier (schema v4); the contribution
WEIGHTS[kind] * scale is computed here at aggregation time, so tuning
WEIGHTS retroactively reprices all history at the next train — nothing
is frozen into the store at collection time. train.py aggregates per
item with recency decay and clips to [-1, 1]. Positive isn't just
"played" — the act of adding something to the library is itself a taste
declaration, which matters here because the movie/TV play history is
thin.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

WEIGHTS = {
    # explicit user verdicts on gustarr's own recommendations — the
    # strongest signal we ever get, so they dominate.
    "approve": 1.0,
    "reject": -1.0,
    # explicit library signals
    "loved": 1.0,       # last.fm loved / jellyfin favorite
    "favorite": 1.0,
    "library_add": 0.6, # user chose to acquire it (unwatched still counts)
    # implicit playback signals
    "complete": 0.8,    # watched >=85% of runtime
    "play": 0.3,        # started it
    "scrobble": 0.15,   # one music listen; repetition accumulates
    "abandon": -0.4,    # started, dropped <20%, never returned
    "skip": -0.1,
}

# Scrobbles accumulate: 40 listens of one artist shouldn't weigh 40×
# a completed movie. Aggregation is log-scaled per (item, kind).
LOG_SCALED_KINDS = {"scrobble", "play"}

HALF_LIFE_DAYS = 365.0  # taste drifts; a listen last week > one in 2020


def recency_factor(ts_iso: str, ref: datetime | None = None) -> float:
    ref = ref or datetime.now(timezone.utc)
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    age_days = max(0.0, (ref - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def aggregate_label(rows: list[tuple[str, str, float]]) -> float:
    """rows: (ts, kind, scale) for one item → label in [-1, 1].

    Each event contributes WEIGHTS[kind] * scale, priced HERE with the
    current table — collectors batch only intensity into scale (e.g.
    jellyfin writes one scrobble event with scale = capped listen count),
    never policy. Log-scaled kinds treat scale as that many
    recency-weighted listens; others take the strongest recency-weighted
    contribution in the base's own direction so one strong old signal
    survives. Kinds absent from WEIGHTS contribute 0: with the base gone
    from the table, whatever policy they once carried is no longer
    expressible as a multiplier of it.
    """
    by_kind: dict[str, list[tuple[str, float]]] = {}
    for ts, kind, scale in rows:
        by_kind.setdefault(kind, []).append((ts, scale))

    total = 0.0
    for kind, entries in by_kind.items():
        base = WEIGHTS.get(kind)
        if base is None:
            continue
        if kind in LOG_SCALED_KINDS:
            if base <= 0:  # log scaling assumes accumulating positives
                continue
            effective_count = sum(
                recency_factor(ts) * max(0.0, scale) for ts, scale in entries
            )
            total += base * math.log1p(effective_count) / math.log(2)
        else:
            contribs = [base * scale * recency_factor(ts) for ts, scale in entries]
            # negative kinds accumulate downward: min() is their strongest
            total += min(contribs) if base < 0 else max(contribs)
    return max(-1.0, min(1.0, total))
