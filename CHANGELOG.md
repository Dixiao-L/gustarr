# Changelog

All notable changes to Gustarr are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [0.4.1] - 2026-07-19

### Added
- `candidates_by_source` in `gustarr stats` / `GET /api/stats`: the
  candidate pool's provenance mix (CF vs. similar-to-known vs. the
  serendipity probes), per profile — the anti-echo-chamber composition,
  scrapeable into Grafana next to the diversity gauges.

## [0.4.0] - 2026-07-13

### Added
- **Weekly Exploration in Jellyfin**: each profile with both a
  ListenBrainz and a Jellyfin user gets a "Weekly Exploration" playlist,
  reconciled every run against what the library holds (matched by
  MusicBrainz track id, with an artist-verified name fallback). The
  week's picks are music you mostly don't have yet — Gustarr feeds them
  to Lidarr, and the playlist grows as they arrive. Gustarr tracks the
  playlist it created in the store and never touches one it didn't
  make; disable with `[listenbrainz] weekly_playlist = false`.

### Fixed
- The ListenBrainz Weekly Exploration candidates source had been dead
  since 0.1.0: the playlist fetch used a plural API route the server
  answers with a redirect into a 404. Found by pre-release review of
  the playlist feature, which shares the fetch.
- Jellyfin item matching (Discover collections, and the new playlist)
  no longer relies on the Emby-era `AnyProviderIdEquals` query, which
  Jellyfin 10.x silently ignores — matching an arbitrary item instead
  of the right one. Lookups now search by title and verify provider ids
  client-side.

## [0.3.0] - 2026-07-12

### Changed
- **Identity layer redesigned** (store schema v3; existing stores migrate
  automatically in one transaction). Items now carry a surrogate integer
  id, and every external id or name an item is known by lives in one
  `identities` table (`domain, ns, key → item`, with
  ns ∈ `tmdb`/`tvdb`/`imdb`/`mbid`/`jellyfin`/`name`). One resolver
  (`resolve_item`) is the single write path and one operation
  (`attach_identity`) teaches an item a new name, merging with the
  authoritative-namespace holder when the collision proves both rows are
  the same entity. This replaces the three overlapping v2 mechanisms —
  namespaced id-string minting, `item_aliases` redirects, and the
  enrich/dedupe merge passes — that the architecture docs carried as
  design debt: "what id does this thing have?" now has one answer.
- Jellyfin item ids are first-class identities: re-syncs and collection
  updates resolve directly by Jellyfin id, with no metadata round-trip.
- Sync cursors and *arr inventory state key on external ids instead of
  item ids, so identity merges can never strand or replay a cursor.

### Fixed
- A name spelling shared by two artists that each hold their own
  MusicBrainz id is treated as proof they are *different* — the attach is
  refused and surfaced as `alias_conflicts` in enrich/dedupe stats.
  MusicBrainz alias lists legitimately carry other entities' names
  (former band names, personas, tributes); during pre-release testing the
  interim merge-on-collision behaviour folded Michael Jackson into
  Wolfgang Amadeus Mozart. Name merges only ever absorb name-only twins,
  the cross-script healing case they exist for.
- The v3 migration is genuinely atomic, DDL included: an interrupted
  upgrade (Ctrl-C, power loss, a second process hitting the lock
  timeout) rolls back to the intact v2 store and simply retries on the
  next start. Two v2 rows claiming the same external id now merge during
  migration exactly as the runtime would (instead of one silently losing
  its identity), the user's approved recommendation survives such a
  merge, v2 event dedup markers follow their items so the first
  post-upgrade sync re-emits nothing, series cursors carry over onto
  merge-proof Jellyfin-id keys, and stale taste-model state is dropped
  for retraining.
- A retagged Jellyfin library entry (provider ids corrected in Jellyfin)
  moves its Jellyfin id to the newly-identified item instead of merging
  the wrong item's history into the right one.
- Track name keys keep the artist/title boundary (unit separator), so
  two different tracks whose concatenated spelling collides stay two
  tracks — and they match what v2 stores migrated in with.
- Collectors no longer trip over their own merges: a mid-sync merge that
  deletes a cached artist id made the Jellyfin and ListenBrainz stages
  crash with foreign-key errors; a name-collision refusal in the
  candidate pool no longer inherits the *other* artist's rejection.
- Two commands racing through the same upgrade can't hurt each other:
  version detection re-runs inside the migration's write transaction, so
  the process that loses the race no-ops instead of re-migrating the
  finished store, and the lock wait tolerates long migrations instead of
  dying "database is locked" after 30 seconds. A store half-migrated by
  a pre-hardening crash is refused with an actionable message in the
  v1→v2 shape too, not just v2→v3.
- One junk row from an external API (a name or id that normalizes to
  nothing) skips that row instead of failing the whole collector stage,
  in every collector and the candidate pool.
- Migration edge cases found by adversarial verification: a Jellyfin
  retag recorded in a v2 store hands the entry's pointer (and its series
  cursor) to the *newer* identification, matching the runtime rule; two
  cursors landing on one key keep the higher count; a merged item's
  missing year/title fill from any member; cursors keyed by
  alias-redirected ids carry over instead of being dropped; one artist
  reached by both a name credit and an mbid credit gets its best CF
  score once instead of a last-writer overwrite.

## [0.2.2] - 2026-07-12

### Fixed
- Cross-profile authorization: acting on another profile's recommendation
  by id now returns 403 (approve/reject/snooze/forgive/why).
- Run Now works in containers: gustarr schedule consumes the sentinel.
- An arr deletion attributes its reject to the profile whose approval
  added the item; ownerless deletions fan out at reduced weight.
- Config-level *arr errors (bad quality profile / root folder) and
  401/403 no longer terminally fail approved recommendations.
- Docker quickstart env example lists every key the example TOML uses;
  docs no longer describe the pre-0.2.1 in-web scheduler.

## [0.2.1] - 2026-07-12

### Added
- **Album recommendations**: Last.fm top albums of liked artists and
  ListenBrainz CF albums feed an album candidate pool, ranked like artists
  (scored against the artist-domain model until album-level feedback
  accumulates) and auto-added to Lidarr under the
  `autonomy.music_max_albums_per_week` weekly budget.
- **`gustarr dedupe`**: merges items that are the same thing under
  different spellings — re-normalizes name-keyed fallback ids and registers
  MusicBrainz alias spellings against the canonical artist, healing history
  split across width/case/script variants (romaji vs. kana/kanji).
  `--fetch` pulls missing alias lists from MusicBrainz (capped by
  `--limit`). Idempotent; run after upgrades or new-source imports.

### Changed
- Scheduling is a dedicated process (gustarr schedule) — never a thread
  inside the web UI. Compose runs it as a second service from the same
  image; the web process serves the approval queue and nothing else.

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
