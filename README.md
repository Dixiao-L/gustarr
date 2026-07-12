# Gustarr

**Learns your media taste — per person, one profile each — from what you
actually watch, listen to, and keep, then recommends and (with configurable
autonomy) drives Sonarr / Radarr / Lidarr. Music, TV, movies.**

[![CI](https://github.com/Dixiao-L/gustarr/actions/workflows/ci.yml/badge.svg)](https://github.com/Dixiao-L/gustarr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

## Why Gustarr

Gustarr focuses on a gap in the self-hosted ecosystem: learning an
*individual* person's taste from their own history. Request portals solve a
different problem (multi-user request management), and collaborative
filtering needs a large user population to work. Gustarr instead builds on
content embeddings with a small per-person preference head — an approach
that becomes useful from roughly 50 interactions and can score brand-new
releases from metadata alone. It reuses mature building blocks wherever
they exist — TMDb and MusicBrainz metadata, ListenBrainz's open
collaborative filtering, Jellyfin's Playback Reporting — and implements
only the taste-learning piece itself.
Households get one such model per person ([profiles](#profiles)), never a
population average.

## Features

- **A taste model that learns *your* labels.** Every approve/reject/watch/love
  feeds a per-domain preference head over multilingual sentence embeddings.
  Label policy is one reviewable table ([`signals.py`](src/gustarr/signals.py)),
  not scattered heuristics.
- **Profiles for households.** Each `[profiles.NAME]` gets its own taste
  model, sync cursors and approval queue over the shared library; the web UI
  resolves the active profile per request from a reverse-proxy auth header
  (Authelia-style) or `?profile=`. Single-user setups need zero config.
- **Anti-echo-chamber engineering, measured.** Serendipity candidates from
  under-represented genres/decades and 2-hop artist neighborhoods, exploration
  slots gated to be genuinely far from your taste centroid, soft rejects for
  labeled long-shots, zero training weight for Gustarr's own automatic adds —
  and `gustarr stats` reports the genre entropy of recent recommendations vs.
  your library so narrowing shows up as a number, not a feeling.
- **Taste before you approve.** Queue cards in the web UI link the movie/series
  trailer and play artist audio previews, so a verdict never requires a blind
  guess.
- **Hybrid autonomy with caps.** Music can auto-add within a weekly budget
  (default: 3 artists/week) while movies/TV wait in an approval queue with a
  pending cap and a proposal TTL — every mode and cap is a config knob.
- **Runtime settings.** Pause actuation, flip music between auto/queue, or
  tune the exploration fraction from the web UI without editing TOML or
  restarting anything; overrides persist in the store and win over the config.
- **Snooze and forgive.** "Not now" is not a verdict — snoozing teaches the
  model nothing and the item comes back after 30 days. Rejected by mistake?
  Forgiving deletes the reject event so training stops penalising the item.
- **Unix philosophy.** Atomic subcommands (`sync`, `enrich`, `candidates`,
  `embed`, `train`, `rank`, `apply`, `dedupe`), one shared SQLite store, no
  daemons except the optional web UI and the optional `gustarr schedule`
  clock process. Composition is `gustarr run nightly|weekly` or
  your own cron/systemd timers.
- **NixOS module** with systemd timers, `EnvironmentFile=` secrets, and
  optional GPU — plus Docker and plain pip/uv installs.
- **Grafana-ready stats.** `GET /api/stats` (same JSON as `gustarr stats`)
  exposes event counts, queue states, model freshness, embedding coverage and
  the diversity metrics — scrape it with Telegraf/Prometheus and graph your
  own filter bubble.

## Architecture

```
  signals in                     the store (SQLite)                  actions out
─────────────────           ─────────────────────────         ─────────────────────────
sync jellyfin ────┐          items       events           ┌──▶ apply → Lidarr (auto, capped)
sync lastfm ──────┤          library     candidates       ├──▶ apply → Radarr/Sonarr (queue)
sync listenbrainz ┼──▶       embeddings  recommendations ─┼──▶ Jellyfin collections
sync arr ─────────┘                                       └──▶ recs / approve / reject /
                                                               snooze / forgive / why
                    enrich → candidates → embed → train → rank      (CLI + web UI)
```

| stage        | what it does                                                            |
|--------------|-------------------------------------------------------------------------|
| `sync *`     | idempotent, incremental signal collection (history, loves, library)     |
| `enrich`     | TMDb / MusicBrainz metadata; upgrades name-keyed ids to canonical       |
| `candidates` | TMDb similar+discover, Last.fm similar, ListenBrainz CF, serendipity    |
| `embed`      | multilingual sentence embeddings of item metadata (GPU or CPU, fp16)    |
| `train`      | per-domain logistic preference head over embeddings (numpy, ms)         |
| `rank`       | score pool → MMR diversity → exploration slots → explanations           |
| `apply`      | actuate the *arrs within caps; only ever touches `gustarr`-tagged items |

Taste labels come from `signals.py` — one reviewable table: completion +0.8,
loved +1.0, library-add +0.6, your reject −1.0, … with 1-year half-life
recency decay and log-scaled listen counts. Details in
[docs/architecture.md](docs/architecture.md).

## Quickstart

You need a [TMDb API key](https://www.themoviedb.org/settings/api) (video
metadata) and a [Last.fm API key](https://www.last.fm/api/account/create)
(music metadata) — both free. Everything else is optional; see the
[FAQ](#faq).

### 1. Docker (self-contained)

```console
$ cp docker-compose.example.yml docker-compose.yml
$ cp gustarr.example.toml gustarr.toml     # set [web] bind = "0.0.0.0:8790",
$                                          # [model] device = "cpu", and
$                                          # [scheduler] nightly = "04:30"
$ cat > gustarr.env <<'EOF'
JELLYFIN_API_KEY=...
TMDB_API_KEY=...
LASTFM_API_KEY=...
SONARR_API_KEY=...
RADARR_API_KEY=...
LIDARR_API_KEY=...
EOF
$ docker compose up -d                     # web UI on :8790 + the scheduler
```

`gustarr.env` must define every `env:VAR` the TOML references — the example
TOML uses all six keys above; or delete the sections for services you don't
run — unset `env:` references fail fast on purpose. `docker compose up -d`
starts both services: the web UI and the scheduler.

With `[scheduler] nightly = "HH:MM"` set, the scheduler service (a dedicated
`gustarr schedule` process from the same image — the web UI never runs the
pipeline) fires it every night; no host cron needed. The scheduler also
watches for the web UI's **Run Now** button, so a manual kick works too —
or use `docker compose run --rm gustarr run nightly`. The image ships
CPU-only torch (works everywhere, ~2 GB); GPU embedding is what the NixOS
module is for. See [docs/deployment.md](docs/deployment.md).

### 2. pip / uv

```console
$ uv sync --extra ml         # full install, embedding included
$ uv sync                    # collectors/actuators/UI only (no torch)

$ cp gustarr.example.toml gustarr.toml   # edit: URLs, usernames, paths
$ export TMDB_API_KEY=... LASTFM_API_KEY=... SONARR_API_KEY=...   # env:VAR secrets

$ uv run gustarr run nightly     # sync → enrich → candidates → embed → train → rank → apply
$ uv run gustarr recs            # the approval queue
$ uv run gustarr approve 12 15   # verdicts feed straight back into training
$ uv run gustarr web             # or approve in the browser at 127.0.0.1:8790
```

`pip install .[ml]` works the same way if you're not a uv person. Schedule
`gustarr run nightly` with cron/systemd, or set `[scheduler] nightly` and run
`gustarr schedule` alongside.

One more command worth knowing: `gustarr dedupe` merges items that are the
same thing under different spellings — the same CJK artist arriving
romanized from one source and in kana/kanji from another otherwise splits
one person's history across duplicate items. It re-normalizes name-keyed
ids and folds MusicBrainz alias spellings into the canonical artist; add
`--fetch` to pull missing alias lists from MusicBrainz (rate-limited,
capped by `--limit`). Not a pipeline stage — run it once after upgrading
Gustarr or after importing history from a new source; every pass is
idempotent.

### 3. NixOS flake module

Runs the pipeline on systemd timers; secrets come from any
`EnvironmentFile=` source; GPU embedding works out of the box.

```nix
{
  inputs.gustarr.url = "github:Dixiao-L/gustarr";

  # in a nixosSystem:
  imports = [ gustarr.nixosModules.default ];

  services.gustarr = {
    enable = true;
    gpu = true;                          # CUDA embedding; false = CPU package
    environmentFiles = [ config.age.secrets.gustarr-env.path ];
    settings = {
      jellyfin = { url = "http://127.0.0.1:8096"; api_key = "env:JELLYFIN_API_KEY"; user = "me"; };
      tmdb.api_key = "env:TMDB_API_KEY";
      lastfm = { api_key = "env:LASTFM_API_KEY"; user = "me"; };
      radarr = { url = "http://127.0.0.1:7878"; api_key = "env:RADARR_API_KEY";
                 quality_profile = "HD-1080p"; root_folder = "/media/movies"; };
    };
    nightly.onCalendar = "*-*-* 04:30:00";   # defaults shown
    weekly.onCalendar = "Sat *-*-* 09:00:00";
  };
}
```

This sets up oneshot systemd timers (`gustarr-nightly`,
`gustarr-weekly`), the web UI as a service, state in `/var/lib/gustarr`, and
secrets exclusively via `EnvironmentFile=` — the generated TOML never contains
key material. Full option reference in [docs/deployment.md](docs/deployment.md).

## Profiles

Multi-user households get one taste model, one approval queue, and one set of
sync cursors *per person* — over a shared library and shared auto-add budgets
(one disk, one Lidarr):

```toml
[profiles.alice]
jellyfin_user = "alice"
lastfm_user = "alice-fm"

[profiles.bob]
jellyfin_user = "bob"
listenbrainz_user = "bob-lb"
```

The web UI resolves the active profile on every request, in order: the
reverse-proxy auth header (`[web] profile_header`, default `Remote-User` —
exactly what Authelia forward-auth sets, so profile names should match your
Authelia usernames), then `?profile=NAME`, then the sole configured profile.
Unknown names are refused with a 403 rather than silently falling back —
nobody's verdicts should ever train someone else's model. The active profile
shows as a chip in the header when more than one is configured; runtime
settings stay operator-global.

No `[profiles]` section means a single implicit `default` profile wired to
the top-level `user` keys — existing single-user setups work unchanged.

## Configuration

One TOML file ([`gustarr.example.toml`](gustarr.example.toml) is a complete
template), found via `--config`, `$GUSTARR_CONFIG`, `./gustarr.toml`, or
`/etc/gustarr/gustarr.toml`. Any string value of the form `env:VAR` is
resolved from the environment at load time, so the file is safe to commit.

| section | key | default | notes |
|---|---|---|---|
| `[core]` | `data_dir` | `/var/lib/gustarr` | store + model cache location |
| `[core]` | `db_path` | `<data_dir>/gustarr.db` | SQLite store (WAL) |
| `[jellyfin]` | `url`, `api_key`, `user` | — | watch/listen history, favorites |
| `[profiles.NAME]` | `jellyfin_user`, `lastfm_user`, `listenbrainz_user` | — | one per household member; omit for single-user |
| `[lastfm]` | `api_key` | — | required: music metadata + similar artists |
| `[lastfm]` | `user` | — | optional: enables scrobble/loved sync |
| `[listenbrainz]` | `user` | — | optional: CF recommendations + weekly playlist |
| `[listenbrainz]` | `token` | — | optional: only lifts rate limits |
| `[tmdb]` | `api_key` | — | required: movie/series metadata + candidates |
| `[sonarr]` `[radarr]` `[lidarr]` | `url`, `api_key` | — | each *arr is optional |
| ″ | `quality_profile`, `root_folder` | `""` | used when adding items |
| ″ | `tag` | `"gustarr"` | Gustarr only ever touches items with this tag |
| `[autonomy]` | `music_mode` | `"auto"` | `auto` or `queue` |
| `[autonomy]` | `video_mode` | `"queue"` | `auto` or `queue` |
| `[autonomy]` | `music_max_artists_per_week` | `3` | auto-add budget |
| `[autonomy]` | `music_max_albums_per_week` | `10` | weekly auto-add budget for album recs |
| `[autonomy]` | `video_queue_max_pending` | `20` | queue cap (and auto-mode per-run cap) |
| `[autonomy]` | `proposal_ttl_days` | `30` | stale proposals expire |
| `[model]` | `embed_model` | `"BAAI/bge-m3"` | any sentence-transformers model |
| `[model]` | `device` | `"cuda"` | `cuda` or `cpu` |
| `[model]` | `model_dir` | `""` | HF cache dir; empty = default |
| `[model]` | `exploration_frac` | `0.15` | share of slots for exploration picks |
| `[model]` | `diversity_lambda` | `0.3` | MMR trade-off, 0 = pure relevance |
| `[web]` | `bind` | `"127.0.0.1:8790"` | web UI bind address |
| `[web]` | `allowed_hosts` | `[]` | extra hostnames past the Host/Origin guard |
| `[web]` | `profile_header` | `"Remote-User"` | forward-auth header mapped to a profile name |
| `[scheduler]` | `nightly` | unset | `"HH:MM"` local time — when the dedicated `gustarr schedule` process fires the pipeline |

A handful of knobs (`paused`, `music_mode`, `music_max_artists_per_week`,
`video_queue_max_pending`, `exploration_frac`) can also be flipped at runtime
from the web UI; those overrides live in the store and win over the TOML until
cleared. Full reference: [docs/configuration.md](docs/configuration.md).

## Signals & weights

How raw events become training labels — the single reviewable policy in
[`signals.py`](src/gustarr/signals.py):

| event | weight | source |
|---|---|---|
| `approve` | **+1.0** | your verdict on a Gustarr recommendation |
| `reject` | **−1.0** | ″ (×0.3 if it was a labeled exploration pick) |
| `loved` / `favorite` | +1.0 | Last.fm loved track / Jellyfin favorite |
| `complete` | +0.8 | watched ≥85% of runtime |
| `library_add` | +0.6 | you chose to acquire it — unwatched still counts |
| `play` | +0.3 | started it (log-scaled accumulation) |
| `scrobble` | +0.15 | one music listen (log-scaled accumulation) |
| `abandon` | −0.4 | started, dropped <20%, never returned |
| `skip` | −0.1 | |

Per-item aggregation applies a 1-year half-life recency decay, log-scales
repeatable kinds (40 listens ≠ 40× a completed movie), and clips to [−1, 1].
Items Gustarr adds autonomously carry **zero** weight until you actually
watch, love, or reject them.

## Honest limitations

- Cold start is real. Until ~50 labeled interactions per domain (per
  profile), rankings lean on external sources (ListenBrainz CF for music,
  TMDb similarity for video) and your library composition.
- TMDb terms nominally restrict ML use; embeddings stay local and are never
  redistributed.
- Explanations are nearest liked-neighbours, not causal claims.

## FAQ

**Why not Jellyseerr / Ombi / SuggestArr?** Fit, not quality — they solve a
different problem. Request portals answer "anyone can ask for anything";
Gustarr quietly grows the library in each person's taste, with the model held
accountable for every pick. It learns **one person per profile** — a
household is several individually-learned tastes over one library, never a
blended average — and there is still no auth layer on the web UI (profiles
are routing, not security: bind it to localhost or put it behind your
reverse proxy).

**Which API keys do I actually need?** TMDb and Last.fm, both free. You do
**not** need a TVDB key — series resolve their TVDB ids through Sonarr's own
inventory and TMDb's external-id mapping. And there's no shared-secret /
password to configure: the web UI deliberately has no auth (localhost bind,
Host/Origin guard); wider exposure — and, for multi-profile households, the
identity header — is your reverse proxy's job.

**What does ListenBrainz add?** Optional, and free: its open collaborative
filtering feeds the music candidate pool (raw CF recs plus your "Weekly
Exploration" playlist), which carries the music side through cold start. Reads
are anonymous — a token only lifts rate limits.

**What does the web UI talk to besides Gustarr?** Your browser loads
poster and portrait images directly from TMDb- and Deezer-hosted URLs,
queries the keyless iTunes Search API for artist previews (the query is
just the artist name) and streams the returned Apple-hosted audio clips,
and trailer/preview-fallback buttons link out to YouTube. No API keys or
personal data are sent — the Gustarr server never proxies these, and
everything else (approve/reject, settings, stats) stays on your host.

**Does it delete things from my *arrs?** No. Gustarr tags everything it adds
(default tag: `gustarr`) and never removes or modifies anything that doesn't
carry that tag.

**GPU required?** No — `device = "cpu"` works, just slower on the first full
library embedding. The Docker image ships CPU-only torch; GPU users should
prefer the NixOS module (or install CUDA wheels in a venv).

## Screenshots

None yet — the approval queue, trailer/preview cards and the settings panel
are best seen live (the Docker quickstart above takes about two minutes).
Screenshot contributions are welcome.

## Documentation

- [docs/architecture.md](docs/architecture.md) — pipeline stages, store schema, id namespaces, echo-chamber design
- [docs/configuration.md](docs/configuration.md) — every TOML key, profiles, `env:` secrets, runtime-settings precedence
- [docs/deployment.md](docs/deployment.md) — Docker compose + scheduler service, NixOS module reference, Authelia profile mapping, secrets guidance
- [docs/development.md](docs/development.md) — dev setup, test conventions, how the suite is organized

## License

[MIT](LICENSE)
