# Changelog

All notable changes to gustarr are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [0.1.0] - 2026-07-12

First public release.

### Added

- **Collectors** (idempotent, incremental, offline-testable): Jellyfin
  (library, watch/listen history, favorites, Playback Reporting), Last.fm
  (scrobbles, loved tracks), ListenBrainz (CF recommendations + Weekly
  Exploration playlist, as candidates), and Sonarr/Radarr/Lidarr inventory
  (library state, adds-as-signal, gustarr-tag feedback).
- **Taste model**: multilingual sentence embeddings (default `BAAI/bge-m3`,
  CUDA or CPU) with a per-domain numpy logistic preference head; centroid
  fallback below 8 positive examples. Label policy lives in one reviewable
  table (`signals.py`) with 1-year half-life recency decay and log-scaled
  listen counts.
- **Ranking**: MMR diversity, exploration slots (default 15%) with a novelty
  gate against the taste centroid, serendipity candidates from
  under-represented genres/decades and 2-hop artist neighborhoods,
  nearest-liked-neighbour explanations (`gustarr why`).
- **Echo-chamber guards**: soft rejects (×0.3) for labeled exploration picks,
  zero training weight for gustarr's own automatic adds, and measured drift —
  genre entropy of recommendations vs. library plus exploration approval rate
  in `gustarr stats` / `GET /api/stats`.
- **Actuation with caps**: music auto-add inside a weekly artist budget,
  video approval queue with pending cap and proposal TTL; approved items are
  actuated in every mode; only `gustarr`-tagged *arr items are ever touched.
- **Approval web UI** (FastAPI, no daemon otherwise): approve / reject /
  snooze / forgive, trailer links and artist audio previews before the
  verdict, runtime settings (pause, autonomy modes, caps, exploration
  fraction) that override the TOML until cleared, Host/Origin guard,
  "Run Now" via a sentinel-file bridge to systemd.
- **CLI**: atomic subcommands (`sync`, `enrich`, `candidates`, `embed`,
  `train`, `rank`, `apply`, `recs`, `approve`, `reject`, `why`, `stats`,
  `web`) composed by `gustarr run nightly|weekly`.
- **Config**: one committable TOML with `env:VAR` secret resolution; runtime
  overrides stored in the DB.
- **Packaging**: Nix flake (CPU and CUDA `ml` variants), NixOS module with
  hardened systemd timers and `EnvironmentFile=` secrets, Dockerfile
  (CPU-only torch) and compose example, MIT license.
- **Tests**: fully offline suite (httpx `MockTransport` / mocked
  `http.request_json`) covering collectors, enrichment, candidates, ML,
  queue, pipeline, actuation and the web API.

[0.1.0]: https://github.com/Dixiao-L/gustarr/releases/tag/v0.1.0
