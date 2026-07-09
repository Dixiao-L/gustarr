# gustarr

Learns **one user's** media taste from what they actually watch, listen to,
and choose to keep — then recommends and (with configurable autonomy) drives
Sonarr / Radarr / Lidarr. Music, TV, movies.

Built after a survey of the existing ecosystem found no mature project that
genuinely learns a single user's taste: request portals (Seerr/Ombi) have no
learning, "AI" recommenders are LLM-prompt wrappers, and collaborative
filtering engines (Gorse) structurally need a user *population*. gustarr
adopts the mature periphery — TMDb/MusicBrainz metadata, ListenBrainz's
open collaborative filtering, Jellyfin Playback Reporting — and builds only
the missing core: a content-embedding taste model with a small learned
preference head, which stays useful from ~50 interactions where classic CF
collapses, and scores brand-new releases from metadata alone.

## Design

Unix philosophy: **atomic subcommands, one shared SQLite store, no daemons
except the optional web UI.** Each stage does one thing; composition is
`gustarr run nightly|weekly` or your own cron/systemd timers.

```
 signals in                 the store                    actions out
──────────────           ───────────────              ────────────────
 sync jellyfin ─┐                                  ┌─▶ apply → Lidarr   (auto, capped)
 sync lastfm   ─┤  events   items    embeddings    ├─▶ apply → Radarr/Sonarr (approval queue)
 sync listenbrainz ─┤ library  candidates  recommendations ├─▶ Jellyfin collections
 sync arr      ─┘                                  └─▶ recs/approve/reject/why (CLI + web)
                  enrich → candidates → embed → train → rank
```

| stage        | what it does                                                        |
|--------------|---------------------------------------------------------------------|
| `sync *`     | idempotent, incremental signal collection (history, loves, library) |
| `enrich`     | TMDb / MusicBrainz metadata; upgrades name-keyed ids to canonical    |
| `candidates` | TMDb similar+discover, Last.fm similar artists, ListenBrainz CF      |
| `embed`      | multilingual sentence embeddings of item metadata (GPU, fp16)        |
| `train`      | per-domain logistic preference head over embeddings (numpy, ms)      |
| `rank`       | score pool → MMR diversity → exploration slots → explanations        |
| `apply`      | actuate the *arrs within caps; only ever touches `gustarr`-tagged items |

Taste labels come from `signals.py` — one reviewable table: completion
+0.8, loved +1.0, library-add +0.6, your reject −1.0, … with 1-year
half-life recency decay and log-scaled listen counts.

## Echo-chamber resistance

A single-user recommender trained on its own output converges on a
bubble unless that's engineered against. gustarr's guards:

- **Serendipity candidates** sampled from *under*-represented genres,
  decades and 2-hop artist neighborhoods (quality-floored) — the pool
  itself contains things similarity search would never surface.
- **Exploration slots** (default 15% of each run) are filled from that
  pool, gated to be genuinely far from the taste centroid, and clearly
  labeled in the CLI/UI.
- **Soft rejects**: declining a labeled long-shot counts 0.3× — an
  experiment that didn't land shouldn't close a whole region of taste.
- **No self-reinforcement**: items gustarr adds autonomously carry zero
  training weight until *you* actually watch, love, or reject them.
- **Drift is measured**: `gustarr stats` reports genre entropy of
  recent recommendations vs. your library and the exploration approval
  rate, so narrowing shows up as a number, not a feeling.

## Honest limitations

- Single-user cold start is real. Until ~50 labeled interactions per
  domain, rankings lean on external sources (ListenBrainz CF for music,
  TMDb similarity for video) and your library composition.
- TMDb terms nominally restrict ML use; embeddings stay local and are
  never redistributed.
- Explanations are nearest liked-neighbours, not causal claims.

## Install

```
uv sync --extra ml        # full, GPU embedding
uv sync                   # collectors/actuators/UI only
```

Copy `gustarr.example.toml`, fill in URLs; secrets come from the
environment (`env:VAR` syntax) — see the example file. NixOS users:
`nix run .#gustarr`, or import the flake's `nixosModules.default`.
