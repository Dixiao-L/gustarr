# Architecture

Gustarr follows one rule: **atomic commands, one shared SQLite store, no
daemons except the optional web UI and the optional `gustarr schedule`
clock.** Every stage does one thing and communicates with the others only
through the store. Composition is
`gustarr run nightly|weekly` — or your own cron/systemd timers calling the
atomic commands directly (containers run the dedicated `gustarr schedule`
process as a second service for exactly that composition — the web process
never runs the pipeline; see [deployment](deployment.md)).

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

## Pipeline stages

The recipe runner (`pipeline.py`) is the only composition in the codebase.
Each stage imports lazily (a missing `ml` extra only breaks `embed`), commits
before the next stage starts (a crash never loses earlier work), and a failed
stage is recorded and skipped over rather than aborting the run. The
`run` command exits non-zero only when *every* executed stage failed — a
single flaky source shouldn't flip a systemd unit red every night.

1. **`sync jellyfin|lastfm|listenbrainz|arr`** — pull taste signals into the
   store. Idempotent and incremental: cursors live in the `state` table, and
   the events uniqueness key makes re-syncs write zero duplicate rows.
   Collectors are deliberately dumb — they record *what happened*; label
   policy lives elsewhere. `sync arr` also inventories what the *arrs already
   manage (never recommend what you own) and treats manual library adds as a
   taste signal.
2. **`enrich`** — fill `items.meta` from TMDb (movies/series) and
   MusicBrainz + Last.fm (music). This is also where weak ids get upgraded to
   canonical namespaces (see below). Permanent per-item failures (bad ids,
   4xx) are stamped so one bad item never wedges the queue; transient
   failures retry next run. First-run backlogs are bounded per night
   (`ENRICH_BATCH_LIMIT = 1500`, priority-ordered) because MusicBrainz
   enforces ~1.1 s between requests.
3. **`candidates`** — refresh the pool of things you *don't* have: TMDb
   recommendations/similar + a genre-keyed discover pass, Last.fm similar
   artists, ListenBrainz CF, plus the serendipity passes (below). Exclusions
   (library, open/acted recommendations, actively snoozed, ever-rejected) are
   applied at insert time; per-source caps keep the enrich/embed backlog
   bounded.
4. **`embed`** — one multilingual document per item (title, genres, tags,
   overview/bio), encoded with a sentence-transformers model (default
   `BAAI/bge-m3`, which copes with mixed CJK/Latin metadata) and stored as
   fp16 blobs. The only stage that needs torch.
5. **`train`** — per-domain logistic preference head over the embeddings,
   trained with plain numpy in milliseconds. Below `MIN_POSITIVES = 8`
   positive examples the head would just memorise, so a positive/negative
   centroid model is used instead. Labels come from `signals.aggregate_label`.
6. **`rank`** — score the pool with the head (or centroid), add small
   multi-source/external-score bonuses, then select with MMR so one taste
   mode doesn't fill the run, reserving `exploration_frac` of slots for
   novelty-gated exploration picks. Every proposal stores a `why` JSON:
   nearest liked neighbours, sources, seeds, exploration flag. Also handles
   TTL expiry and snooze lapse.
7. **`apply`** — the only stage that changes the outside world. Approved
   recommendations are actuated in every mode (an approve is consent); `auto`
   mode additionally adds proposed items within caps. Adds are committed to
   the store immediately (they're irreversible), transient *arr outages keep
   the rec status for retry, and Jellyfin collections are synced best-effort.

Both recipes (`nightly`, `weekly`) currently run the same full stage list
*including* `apply` — an explicit approval should land in the *arr within
hours, not "next Saturday". This doesn't inflate autonomy: the music budget
counts what was already acted this ISO week, so a nightly cadence spends the
same weekly allowance, just sooner. The two recipe names exist so timers can
carry different schedules and future recipes can diverge.

## Store schema overview

One SQLite database, WAL mode, created on first connect (`db.py`):

| table | role |
|---|---|
| `items` | canonical catalogue: `id`, `domain`, `title`, `year`, `ids` (JSON id map), `meta` (genres, overview, trailer, …), `enriched_at` — global |
| `events` | append-only taste signals owned by a profile: `profile`, `ts`, `item_id`, `kind`, `weight`, `source`, `dedup`, `meta`; unique on `(profile, ts, item_id, kind, source, dedup)` so re-syncs are idempotent |
| `item_aliases` | fallback → canonical id redirects recorded by merges — global |
| `library` | what the *arrs already manage — never recommended; global |
| `candidates` | the pool rank scores; PK `(profile, item_id, source)` so one item found by several sources keeps all its provenance per profile, with `seed_item_id` for "why" |
| `recommendations` | the per-profile queue: `profile`, `run_id`, `score`, `why` JSON, `status`, `acted_at` |
| `embeddings` | fp16 vectors, PK `(item_id, model)` — swapping `embed_model` re-embeds cleanly; global |
| `state` | key-value: per-profile sync cursors and trained models (`p:<profile>:*`), global runtime setting overrides (`setting:*`) and *arr inventory |

**Profile ownership in one line**: taste (events, candidates,
recommendations, cursors, models) is per-profile; the world (items,
metadata, embeddings, *arr library state) and the auto-add budgets are
global — one disk, one Lidarr, however many tastes. Library adds and
*arr-deletion rejects are attributed to *every* configured profile at a
modest weight: the household owns the library, so nobody's model gets to
claim sole credit for it.

Recommendation status flow:

```
proposed ──▶ approved ──▶ added        (apply pushed it to the *arr)
        ──▶ rejected                   (reject event written; forgivable)
        ──▶ snoozed  ──▶ expired       (after 30 days; re-proposable)
        ──▶ auto_added                 (music auto mode; zero training weight)
        ──▶ expired                    (proposal TTL / queue overflow)
        ──▶ failed                     (terminal actuation failure)
```

`added`, `auto_added` and `failed` are terminal — the item has left the
queue's control, so a late approve/reject would be a lie and is refused.

## Id namespaces

Every item has exactly one id of the form `domain:ns:key` (`ids.py`):

```
movie:tmdb:603              series:tvdb:81189
artist:mbid:5b11f4ce-...    album:mbid:1b022e01-...
track:lastfm:radiohead␟paranoid android      (fallback when no MBID)
```

Domains are `movie | series | artist | album | track`. The namespace is the
*most authoritative* id known for that item, with a per-domain priority:

| domain | namespaces, strongest first |
|---|---|
| movie | `tmdb`, `imdb` |
| series | `tvdb`, `tmdb`, `imdb` |
| artist / album / track | `mbid`, `lastfm` (name-keyed fallback) |

`series` is canonically `tvdb`-keyed because Sonarr can only add by TVDB id;
enrich resolves it via TMDb's external-id mapping, so no TVDB API key is ever
needed. Multi-part keys join with the ASCII unit separator (`\x1f`), which
never occurs in artist/track names.

## Merge & alias semantics

Collectors often only know a weak id (a Last.fm artist name, an IMDb id).
They mint the fallback form; `enrich` later resolves the authoritative id and
calls `db.merge_item`, which:

1. non-destructively folds the fallback row's `ids`/`meta` into the canonical
   item,
2. records the redirect in `item_aliases` (re-pointing older aliases too),
3. re-points `events`, `candidates`, `library` and `recommendations` rows —
   using `UPDATE OR IGNORE` + delete so rows that would collide with existing
   canonical rows (the same scrobble synced under both ids) dedupe instead of
   erroring,
4. drops the fallback item and its embeddings.

From then on `db.canonical_id` transparently redirects any collector that
re-mints the fallback id, so merged items never come back to life. Fuzzy
MusicBrainz name matches below score 90 are *not* auto-merged — keeping a
fallback id beats pointing someone's scrobble history at the wrong artist.

## Signal weighting

`signals.py` is the single place where events become labels (see the table in
the [README](../README.md#signals--weights)). Aggregation per item:

- **recency decay**: every contribution is scaled by `0.5^(age/365d)` — taste
  drifts, a listen last week outweighs one from 2020.
- **log-scaled kinds** (`scrobble`, `play`): repetition accumulates as
  `log2(1 + effective_count)` so 40 listens of one artist can't drown out a
  completed movie.
- **one-shot kinds**: the strongest single recency-weighted signal survives
  (min for negative kinds).
- the result clips to [−1, 1]; training thresholds it at +0.15 / −0.05 with
  in-between items contributing weakly.

## Echo-chamber design

A personal recommender trained on its own output converges on a bubble
unless that's engineered against. The guards, per profile, end to end:

- **Serendipity candidates** (`candidates.py`): TMDb discover over the user's
  *under*-represented genres (rarest first, deterministic) and least-visited
  decade, quality-floored at 500 votes so probes are great-but-unfamiliar,
  not random junk; plus a damped 2-hop Last.fm fan-out from the best hop-1
  candidates, skipping anything already reachable at hop 1 (bubble-adjacent
  ≠ serendipity). Capped tighter than taste sources — seasoning, not the
  meal.
- **Exploration slots** (`rank.py`): `exploration_frac` (default 15%) of each
  run's slots bypass top-score selection. Picks prefer serendipity
  candidates, must sit below the 60th percentile of similarity to the
  positive centroid (genuinely outside the taste core), above a sanity floor,
  and are chosen farthest-first from what's already picked. They're labeled
  `exploration` in the `why` JSON, the CLI and the UI.
- **Soft rejects** (`queue.py`): rejecting a labeled exploration pick counts
  ×0.3 — an experiment that didn't land shouldn't close a whole region of
  taste. Approvals stay full weight; a hit is a hit however it was found.
- **No self-reinforcement** (`apply.py`): auto-added items write an
  `auto_add` audit event with **zero** training weight. Only your actual
  watch/love/reject moves the model.
- **Drift is measured** (`queue.store_stats`): genre entropy (Shannon, bits)
  of the last 100 recommendations vs. the library it feeds on, the realised
  exploration share, and the exploration approval rate — exposed via
  `gustarr stats` and `GET /api/stats` so narrowing shows up on a Grafana
  dashboard, not as a feeling.

## Design debt

Known structural shortcuts, documented so nobody has to discover them the
hard way. None is load-bearing for correctness today; all have a planned
fix.

- **Item identity is spread across three mechanisms**: namespaced id
  minting with per-domain priority (`ids.py`), `item_aliases` redirects
  consulted via `db.canonical_id` at write time, and the merge/upgrade
  passes in `enrich` and `dedupe`. Each is individually sound, but "what
  id does this thing have?" has three answers depending on where you ask.
  Planned: consolidation behind a single resolver.
- **Recommendation status transitions are written from several call
  sites**: `queue.set_status`/`forgive` (approve/reject/snooze/un-reject),
  `apply` (added/auto_added/failed, TTL and overflow expiry) and `rank`
  (TTL and snooze-lapse expiry) each update `recommendations.status`
  directly. The legal-transition rules live in `queue.py` but are only
  enforced there. Planned: a single transition owner every writer goes
  through.
- **Event weights are frozen at write time**: collectors store each
  event's label contribution in `events.weight`, and
  `signals.aggregate_label` treats the stored value as authoritative — so
  tuning `signals.WEIGHTS` only affects events written afterwards.
  Planned: weight recomputation from `kind` at training time.
- **Cross-profile budget ordering favors profiles with saturated heads**:
  the shared weekly music budgets are spent in one score-ordered pass
  across every profile's queue, and a profile with a well-trained
  (confident, higher-scoring) head systematically outbids a cold-start
  one. Planned: round-robin allocation across profiles.

## HTTP discipline

All external calls (TMDb, Last.fm, MusicBrainz, ListenBrainz, the *arrs,
Jellyfin) go through `http.request_json`: uniform retries with exponential
backoff, `Retry-After` honored but clamped, and per-host politeness delays
(MusicBrainz ≥1.1 s). Tests inject `httpx.MockTransport` or stub
`http.request_json` — the suite never touches the network.
