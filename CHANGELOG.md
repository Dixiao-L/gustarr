# Changelog

All notable changes to Gustarr are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [0.2.0] - 2026-07-12

### Added

- **Profiles**: `[profiles.NAME]` config sections give each household member
  their own taste model, sync cursors, and approval queue over the shared
  library and shared auto-add budgets. The web UI resolves the active profile
  per request — reverse-proxy auth header (`[web] profile_header`, default
  `Remote-User`, matching Authelia forward-auth), then `?profile=`, then the
  sole configured profile; unknown names get a 403. `GET /api/profile`
  reports the active profile, shown as a header chip when more than one is
  configured. Configs without `[profiles]` keep working via an implicit
  `default` profile.
- **Built-in scheduler**: with `[scheduler] nightly = "HH:MM"` (local time),
  `gustarr web` launches `gustarr run nightly` as a subprocess once a day —
  containers no longer need host cron. Off by default; skips a slot while the
  previous run is still alive; NixOS/systemd deployments keep their timers.

### Changed

- Events, candidates and recommendations are profile-scoped in the store;
  v1 databases migrate automatically with existing rows assigned to the
  `default` profile. Sync cursors and trained models are namespaced per
  profile; items, embeddings, library state and weekly budgets stay global.
- The product is written **Gustarr** in documentation and UI prose; the
  command, package, image and *arr tag stay lowercase `gustarr`.
- README Quickstart is Docker-first, with the built-in scheduler as the
  primary scheduling path for containers.

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
  zero training weight for Gustarr's own automatic adds, and measured drift —
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
