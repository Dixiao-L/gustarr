"""Runtime settings: state-table overrides on top of the TOML config.

The TOML file is the committable baseline; anything a user flips at
runtime (web UI, CLI) lands in the state table under ``setting:<key>``
and wins over the config until cleared. Values are json-encoded in
state so types survive the round-trip.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from . import db
from .config import Config

_PREFIX = "setting:"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"expected a boolean (true/false), got {value!r}")


def _as_choice(*choices: str) -> Callable[[Any], str]:
    def coerce(value: Any) -> str:
        if isinstance(value, str) and value.strip() in choices:
            return value.strip()
        raise ValueError(f"expected one of {'/'.join(choices)}, got {value!r}")
    return coerce


def _as_int(minimum: int) -> Callable[[Any], int]:
    def coerce(value: Any) -> int:
        # bool is an int subclass; True must not sneak in as 1
        if isinstance(value, bool):
            raise ValueError(f"expected an integer >= {minimum}, got {value!r}")
        if isinstance(value, int):
            n = value
        elif isinstance(value, float) and value.is_integer():
            n = int(value)
        elif isinstance(value, str):
            try:
                n = int(value.strip())
            except ValueError:
                raise ValueError(f"expected an integer >= {minimum}, got {value!r}") from None
        else:
            raise ValueError(f"expected an integer >= {minimum}, got {value!r}")
        if n < minimum:
            raise ValueError(f"expected an integer >= {minimum}, got {n}")
        return n
    return coerce


def _as_float(lo: float, hi: float) -> Callable[[Any], float]:
    def coerce(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"expected a number in [{lo}, {hi}], got {value!r}")
        try:
            x = float(value.strip() if isinstance(value, str) else value)
        except (ValueError, TypeError):
            raise ValueError(f"expected a number in [{lo}, {hi}], got {value!r}") from None
        if not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f"expected a number in [{lo}, {hi}], got {value!r}")
        return x
    return coerce


@dataclass(frozen=True)
class _Spec:
    coerce: Callable[[Any], Any]
    default: Callable[[Config], Any]


RUNTIME_KEYS: dict[str, _Spec] = {
    "paused": _Spec(_as_bool, lambda cfg: False),
    "music_mode": _Spec(_as_choice("auto", "queue"), lambda cfg: cfg.autonomy.music_mode),
    "music_max_artists_per_week": _Spec(
        _as_int(0), lambda cfg: cfg.autonomy.music_max_artists_per_week),
    "video_queue_max_pending": _Spec(
        _as_int(1), lambda cfg: cfg.autonomy.video_queue_max_pending),
    "exploration_frac": _Spec(_as_float(0.0, 0.9), lambda cfg: cfg.model.exploration_frac),
}


def _spec(key: str) -> _Spec:
    spec = RUNTIME_KEYS.get(key)
    if spec is None:
        raise ValueError(f"unknown setting {key!r} (known: {', '.join(sorted(RUNTIME_KEYS))})")
    return spec


def get(conn: sqlite3.Connection, cfg: Config, key: str) -> Any:
    """The effective value: a state override wins, else the config value."""
    spec = _spec(key)
    raw = db.get_state(conn, _PREFIX + key)
    if raw is not None:
        return json.loads(raw)
    return spec.default(cfg)


def get_all(conn: sqlite3.Connection, cfg: Config) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, spec in RUNTIME_KEYS.items():
        raw = db.get_state(conn, _PREFIX + key)
        default = spec.default(cfg)
        out[key] = {
            "value": json.loads(raw) if raw is not None else default,
            "overridden": raw is not None,
            "default": default,
        }
    return out


def set(conn: sqlite3.Connection, key: str, value: Any) -> Any:  # noqa: A001
    """Validate + coerce (e.g. "3"→3, "true"→True), then persist the
    override json-encoded. Returns the coerced value."""
    coerced = _spec(key).coerce(value)
    db.set_state(conn, _PREFIX + key, json.dumps(coerced))
    return coerced


def clear(conn: sqlite3.Connection, key: str) -> None:
    _spec(key)
    conn.execute("DELETE FROM state WHERE key=?", (_PREFIX + key,))
