# Configuration

One TOML file. [`gustarr.example.toml`](../gustarr.example.toml) is a complete
template.

## Where the file is found

Search order (`config.py`):

1. `--config PATH` flag
2. `$GUSTARR_CONFIG` environment variable
3. `./gustarr.toml`
4. `/etc/gustarr/gustarr.toml`

## Secrets: the `env:` convention

Any **string value** of the form `env:VAR_NAME` is resolved from the process
environment at load time:

```toml
[tmdb]
api_key = "env:TMDB_API_KEY"
```

This keeps the TOML committable while API keys arrive via systemd
`EnvironmentFile=`, agenix, direnv, docker `env_file:` — never on the command
line or in a world-readable file. A referenced variable that is unset is a
hard config error, so a missing secret fails loudly at startup rather than
silently mid-pipeline.

## Precedence: TOML vs. runtime settings

Two layers, strict precedence:

1. **TOML config** — the committable baseline.
2. **Runtime settings** — anything flipped in the web UI lands in the store's
   `state` table (`setting:<key>`) and **wins over the TOML** until cleared.

Only these keys are runtime-overridable (`settings.RUNTIME_KEYS`):

| key | type / range | TOML default it overrides |
|---|---|---|
| `paused` | bool | `false` (no TOML equivalent — stops `apply` entirely) |
| `music_mode` | `"auto"` \| `"queue"` | `autonomy.music_mode` |
| `music_max_artists_per_week` | int ≥ 0 | `autonomy.music_max_artists_per_week` |
| `video_queue_max_pending` | int ≥ 1 | `autonomy.video_queue_max_pending` |
| `exploration_frac` | float in [0.0, 0.9] | `model.exploration_frac` |

Manage them via the web UI's settings panel or the API:

```console
$ curl http://127.0.0.1:8790/api/settings                     # values + overridden flags
$ curl -X PUT  -d '{"value": true}' http://127.0.0.1:8790/api/settings/paused
$ curl -X DELETE http://127.0.0.1:8790/api/settings/paused    # back to the TOML value
```

Values are validated and coerced (`"3"` → `3`, `"true"` → `true`);
`gustarr stats` reports how many overrides are active
(`settings_overridden`). Runtime settings are **operator-level and global**:
one store, one set of caps and budgets, shared by every profile — the web
UI's settings dialog says so out loud. Everything else requires editing the
TOML — which takes effect on the next command; nothing long-running caches
config except the web UI (restart it after changing `[web]` or
`[scheduler]`).

## Full key reference

### `[core]`

| key | default | meaning |
|---|---|---|
| `data_dir` | `/var/lib/gustarr` | state directory: store, HF model cache, run sentinel |
| `db_path` | `<data_dir>/gustarr.db` | the SQLite store |

### `[jellyfin]` — optional section

| key | meaning |
|---|---|
| `url` | e.g. `http://127.0.0.1:8096` |
| `api_key` | Jellyfin API key (Dashboard → API Keys) |
| `user` | Jellyfin username whose history is the taste signal |

All three are required if the section is present. Provides library state,
watch/listen history (rolled up to series/artists), favorites, and — when the
Playback Reporting plugin is installed — finer-grained playback signals.

### `[profiles.NAME]` — optional; one section per household member

| key | meaning |
|---|---|
| `jellyfin_user` | this profile's Jellyfin username (overrides `[jellyfin] user`) |
| `lastfm_user` | this profile's Last.fm username (overrides `[lastfm] user`) |
| `listenbrainz_user` | this profile's ListenBrainz username (overrides `[listenbrainz] user`) |

Each profile gets its **own** taste model, sync cursors and approval queue;
the item catalogue, embeddings, *arr library state and the weekly auto-add
budgets are **shared** — one disk, one Lidarr. All keys are optional per
profile; an empty one simply contributes no signals from that source.

```toml
[profiles.alice]
jellyfin_user = "alice"
lastfm_user = "alice-fm"

[profiles.bob]
jellyfin_user = "bob"
listenbrainz_user = "bob-lb"
```

Without any `[profiles]` section, a single `default` profile is synthesized
from the top-level `user` keys — single-user configs work unchanged. The
web request → profile mapping is described under [`[web]`](#web) below.

### `[lastfm]` — optional section

| key | meaning |
|---|---|
| `api_key` | required for music metadata, similar artists, and artist bios in `enrich`/`candidates` |
| `user` | optional; enables `sync lastfm` (scrobbles, loved tracks) |

`gustarr sync lastfm --full` ignores the cursor and re-walks the entire
scrobble history.

### `[listenbrainz]` — optional section

| key | meaning |
|---|---|
| `user` | ListenBrainz username; enables CF recommendations + Weekly Exploration playlist as candidates |
| `token` | optional; reads are public — a token only lifts rate limits |

### `[tmdb]`

| key | meaning |
|---|---|
| `api_key` | required for movie/series metadata and candidates |

### `[sonarr]`, `[radarr]`, `[lidarr]` — each optional

| key | default | meaning |
|---|---|---|
| `url` | — | e.g. `http://127.0.0.1:8989` |
| `api_key` | — | the *arr's API key |
| `quality_profile` | `""` | profile name used when adding |
| `root_folder` | `""` | root folder path used when adding |
| `tag` | `"gustarr"` | items Gustarr adds carry this tag; Gustarr never removes/modifies anything without it |

Configure only the *arrs you run — unconfigured ones are skipped, and
recommendations that are ready stay queued until the *arr appears.

### `[autonomy]`

| key | default | meaning |
|---|---|---|
| `music_mode` | `"auto"` | `auto`: add proposed artists within the weekly budget; `queue`: everything waits for approval |
| `video_mode` | `"queue"` | `auto`: add top proposed video without approval (capped per run by `video_queue_max_pending`) |
| `music_max_artists_per_week` | `3` | weekly auto-add budget, counted per ISO week; approvals don't consume it |
| `music_max_albums_per_week` | `10` | reserved for future album-level recommendations |
| `video_queue_max_pending` | `20` | max open movie/series proposals; rank stops proposing at the cap, apply expires overflow |
| `proposal_ttl_days` | `30` | proposals older than this expire so the queue never silts up |

### `[model]`

| key | default | meaning |
|---|---|---|
| `embed_model` | `"BAAI/bge-m3"` | any sentence-transformers model; changing it re-embeds and re-trains cleanly |
| `device` | `"cuda"` | `cuda` or `cpu` |
| `model_dir` | `""` | HuggingFace cache dir; empty = HF default (`$HF_HOME`) |
| `exploration_frac` | `0.15` | share of each run's slots reserved for exploration picks |
| `diversity_lambda` | `0.3` | MMR trade-off; `0` = pure relevance, higher = more diverse |

### `[web]`

| key | default | meaning |
|---|---|---|
| `bind` | `"127.0.0.1:8790"` | web UI bind address (`host:port`) |
| `allowed_hosts` | `[]` | extra hostnames accepted by the Host/Origin guard |
| `profile_header` | `"Remote-User"` | request header mapped to a profile name (what Authelia forward-auth sets) |

The web UI has **no auth by design**. It binds localhost by default; a
Host/Origin guard rejects DNS-rebinding and cross-site requests even there.
If you expose it via a reverse proxy or a non-localhost bind, add the public
hostname to `allowed_hosts` — otherwise browsers get a 403 — and let the
proxy own TLS and access control.

Profiles are routing, not security. Every request resolves to a profile in
this order:

1. the header named by `profile_header` (set by your forward-auth proxy),
2. an explicit `?profile=NAME` query parameter,
3. the sole configured profile,
4. `default`.

A header or parameter naming an *unknown* profile is a hard 403 — a typo'd
or unmapped user must never silently train someone else's model. Because the
header is trusted as-is, a multi-profile instance must only be reachable
through the proxy that sets it. `GET /api/profile` returns the resolved name
and the configured list.

### `[scheduler]` — optional; built-in nightly scheduler

| key | default | meaning |
|---|---|---|
| `nightly` | unset | `"HH:MM"` **local time**; `gustarr web` runs `gustarr run nightly` as a subprocess once a day |

Meant for containers, where there is no systemd. Off by default — when
unset, `gustarr web` schedules nothing and you keep cron/systemd timers. The
pipeline runs as a subprocess (it never blocks the web UI), a slot that comes
up while the previous run is still alive is skipped, and start/exit are
logged to stdout. In containers, "local time" is the container's `TZ`.
