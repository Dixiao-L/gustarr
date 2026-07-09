"""Recipe runner — the only composition in gustarr.

Each stage imports lazily inside the loop so a missing optional
dependency (e.g. the ml extra for embed) only breaks its own stage, and
every stage commits before the next starts so a crash never loses
earlier work. A failed stage is recorded and skipped over: train/rank
already no-op gracefully when embed left no vectors behind.
"""

from __future__ import annotations

import importlib
import sqlite3
from typing import Any, Callable

from .config import Config

# stage name → (module, function, configured?). configured=None: always runs;
# sync stages are skipped when their config section is absent.
STAGES: dict[str, tuple[str, str, Callable[[Config], bool] | None]] = {
    "sync_arr": ("gustarr.collect.arr", "sync",
                 lambda cfg: bool(cfg.sonarr or cfg.radarr or cfg.lidarr)),
    "sync_jellyfin": ("gustarr.collect.jellyfin", "sync", lambda cfg: bool(cfg.jellyfin)),
    "sync_lastfm": ("gustarr.collect.lastfm", "sync", lambda cfg: bool(cfg.lastfm)),
    "sync_listenbrainz": ("gustarr.collect.listenbrainz", "sync",
                          lambda cfg: bool(cfg.listenbrainz)),
    "enrich": ("gustarr.enrich", "run", None),
    "candidates": ("gustarr.candidates", "run", None),
    "embed": ("gustarr.ml.embed", "run", None),
    "train": ("gustarr.ml.train", "run", None),
    "rank": ("gustarr.ml.rank", "run", None),
    "apply": ("gustarr.actuate.apply", "run", None),
}

NIGHTLY = ["sync_arr", "sync_jellyfin", "sync_lastfm", "sync_listenbrainz",
           "enrich", "candidates", "embed", "train", "rank"]
WEEKLY = [*NIGHTLY, "apply"]
RECIPES = {"nightly": NIGHTLY, "weekly": WEEKLY}


def run_recipe(
    conn: sqlite3.Connection,
    cfg: Config,
    recipe: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r} (have: {', '.join(RECIPES)})")

    stats: dict[str, Any] = {}
    errors: list[str] = []
    for name in RECIPES[recipe]:
        module_path, func_name, configured = STAGES[name]
        if configured is not None and not configured(cfg):
            stats[name] = "skipped"
            continue
        kwargs = {"dry_run": dry_run} if name == "apply" else {}
        try:
            func = getattr(importlib.import_module(module_path), func_name)
            stats[name] = func(conn, cfg, **kwargs)
        except Exception as exc:  # noqa: BLE001 — stage isolation is the whole point
            stats[name] = {"error": str(exc)}
            errors.append(name)
        conn.commit()
    stats["errors"] = errors
    return stats
