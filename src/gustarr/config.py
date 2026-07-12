"""Configuration: one TOML file, secrets from the environment.

Any string value of the form ``env:VAR_NAME`` is resolved from the
environment at load time. This keeps the TOML committable while API keys
arrive via an env file (systemd ``EnvironmentFile=`` / agenix), never on
the command line or in the store.

Search order for the config file:
  1. --config flag / GUSTARR_CONFIG env var
  2. ./gustarr.toml
  3. /etc/gustarr/gustarr.toml
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATHS = [Path("gustarr.toml"), Path("/etc/gustarr/gustarr.toml")]


class ConfigError(Exception):
    pass


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("env:"):
        var = value[4:]
        resolved = os.environ.get(var)
        if resolved is None:
            raise ConfigError(f"config references env:{var} but ${var} is not set")
        return resolved
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


@dataclass
class ArrConfig:
    url: str
    api_key: str
    quality_profile: str = ""
    root_folder: str = ""
    # Items gustarr adds are tagged so feedback can find them and so a
    # human can tell them apart from manual adds. Never remove/modify
    # anything in the *arr that doesn't carry this tag.
    tag: str = "gustarr"


@dataclass
class AutonomyConfig:
    music_mode: str = "auto"  # auto | queue
    video_mode: str = "queue"  # auto | queue
    music_max_artists_per_week: int = 3
    music_max_albums_per_week: int = 10
    video_queue_max_pending: int = 20
    # Recommendations older than this are marked expired so the queue
    # never silts up with stale suggestions.
    proposal_ttl_days: int = 30


@dataclass
class ModelConfig:
    embed_model: str = "BAAI/bge-m3"
    device: str = "cuda"  # cuda | cpu
    model_dir: str = ""  # HF cache dir; empty = default
    # Ranking knobs. exploration_frac of each run's slots are filled by
    # high-uncertainty/novel picks instead of top score, so the model
    # keeps getting signal outside its comfort zone.
    exploration_frac: float = 0.15
    diversity_lambda: float = 0.3  # MMR trade-off, 0 = pure relevance


@dataclass
class ProfileConfig:
    """One person's identity across the signal sources. The 'default'
    profile is synthesized from the legacy top-level sections so existing
    single-user configs keep working unchanged."""
    jellyfin_user: str = ""
    lastfm_user: str = ""
    listenbrainz_user: str = ""


@dataclass
class Config:
    db_path: Path
    data_dir: Path
    jellyfin: dict[str, Any] = field(default_factory=dict)
    lastfm: dict[str, Any] = field(default_factory=dict)
    listenbrainz: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    tmdb: dict[str, Any] = field(default_factory=dict)
    sonarr: ArrConfig | None = None
    radarr: ArrConfig | None = None
    lidarr: ArrConfig | None = None
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    web: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load(path: str | Path | None = None) -> Config:
    candidates = [Path(p) for p in ([path] if path else [])]
    if not candidates:
        if env_path := os.environ.get("GUSTARR_CONFIG"):
            candidates = [Path(env_path)]
        else:
            candidates = DEFAULT_PATHS
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "rb") as f:
                raw = _resolve_env(tomllib.load(f))
            return _build(raw)
    raise ConfigError(f"no config file found (tried: {', '.join(map(str, candidates))})")


def _build(raw: dict[str, Any]) -> Config:
    core = raw.get("core", {})
    data_dir = Path(core.get("data_dir", "/var/lib/gustarr"))

    def arr(section: str) -> ArrConfig | None:
        if section not in raw:
            return None
        known = {k: v for k, v in raw[section].items() if k in ArrConfig.__dataclass_fields__}
        return ArrConfig(**known)

    profiles: dict[str, ProfileConfig] = {}
    for name, p in (raw.get("profiles") or {}).items():
        profiles[name] = ProfileConfig(
            jellyfin_user=p.get("jellyfin_user", ""),
            lastfm_user=p.get("lastfm_user", ""),
            listenbrainz_user=p.get("listenbrainz_user", ""),
        )
    if not profiles:
        profiles["default"] = ProfileConfig(
            jellyfin_user=raw.get("jellyfin", {}).get("user", ""),
            lastfm_user=raw.get("lastfm", {}).get("user", ""),
            listenbrainz_user=raw.get("listenbrainz", {}).get("user", ""),
        )

    return Config(
        db_path=Path(core.get("db_path", data_dir / "gustarr.db")),
        data_dir=data_dir,
        jellyfin=raw.get("jellyfin", {}),
        lastfm=raw.get("lastfm", {}),
        listenbrainz=raw.get("listenbrainz", {}),
        profiles=profiles,
        tmdb=raw.get("tmdb", {}),
        sonarr=arr("sonarr"),
        radarr=arr("radarr"),
        lidarr=arr("lidarr"),
        autonomy=AutonomyConfig(
            **{k: v for k, v in raw.get("autonomy", {}).items()
               if k in AutonomyConfig.__dataclass_fields__}
        ),
        model=ModelConfig(
            **{k: v for k, v in raw.get("model", {}).items()
               if k in ModelConfig.__dataclass_fields__}
        ),
        web=raw.get("web", {}),
        raw=raw,
    )
