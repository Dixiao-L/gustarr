# Development

## Setup

```console
$ git clone https://github.com/Dixiao-L/gustarr && cd gustarr
$ uv sync --extra dev            # everything except torch
$ uv sync --extra dev --extra ml # + torch/sentence-transformers, if you're touching embed
$ uv run pytest -q
$ uv run ruff check .
```

Python ≥ 3.12 (the code uses `tomllib` and modern typing). Nix users:
`nix develop` provides uv + python; `nix build .#default` / `.#ml` builds the
packages and runs the test suite as the check phase.

The `ml` extra is deliberately optional — collectors, the queue, actuation
and the web UI never import torch, and `cli.py` defers heavy imports into
the subcommands that need them so `gustarr recs` stays instant. Keep it that
way: if your change makes `import gustarr.queue` pull in torch, it's wrong.

## Test conventions

The suite is **fully offline by design**. The Nix build sandbox has no
network, and CI should never flake on someone else's API. Concretely:

- Anything that would touch the network in a test is a bug. There are no
  "integration" markers, no recorded cassettes, no skips-when-offline.
- All external HTTP goes through `gustarr.http.request_json` (politeness
  delays, retries, uniform errors). Tests intercept at one of two levels:
  1. **`httpx.MockTransport` via the `transport` argument** — collectors and
     arr clients accept a transport parameter precisely for this. Route on
     URL/params and return canned JSON (see `tests/test_collect_lastfm.py`,
     `tests/test_collect_jellyfin.py`, `tests/test_actuate.py`).
  2. **Monkeypatching `http.request_json`** with a URL-fragment router, for
     modules that fan out through many endpoints (see `tests/test_enrich.py`,
     `tests/test_candidates.py`).
  Either way, also `monkeypatch.setattr(http, "HOST_DELAYS", {})` so tests
  don't sleep out MusicBrainz politeness.
- **Never import sentence-transformers in tests.** `tests/test_ml.py` covers
  embed's document building and selection logic, then fabricates small
  synthetic vectors (and stubs the model module in `sys.modules` where
  needed) to exercise train/rank end-to-end. torch is not a test dependency.
- The store is real: tests build an actual SQLite DB in `tmp_path` via
  `db.connect` and assert on rows, not mocks. Config objects are constructed
  directly (`config.Config(...)` / small TOML fixtures).
- Web tests use FastAPI's `TestClient`; note the Host/Origin guard —
  fixtures put `testserver` in `[web] allowed_hosts` (and one test asserts
  the 403 without it).

## Suite organization

One test module per source module, flat in `tests/`:

| file | covers |
|---|---|
| `test_signals.py` | weight table semantics, recency decay, label aggregation |
| `test_http.py` | retries, backoff, Retry-After clamp, politeness delays |
| `test_collect_jellyfin.py` / `_lastfm.py` / `_listenbrainz.py` / `_arr.py` | each collector: cursors, idempotent re-sync, id minting |
| `test_enrich.py` | metadata fill, id upgrades, merge/alias behaviour, error stamping |
| `test_candidates.py` | pool gate, exclusions, caps, serendipity probes |
| `test_ml.py` | embed doc building, head/centroid training, ranking + exploration |
| `test_queue_pipeline.py` | approve/reject/snooze/forgive, recipe runner stage isolation |
| `test_actuate.py` | caps, budgets, dry-run, transient-vs-permanent failures |
| `test_web.py` | API endpoints, Host/Origin guard, runtime settings, profile resolution, the built-in scheduler |

Run a slice with `uv run pytest tests/test_enrich.py -q` or `-k pattern`.

## Style

- `ruff check .` must pass; line length 100, target py312 (`pyproject.toml`).
  No formatter is enforced — match the file you're in.
- Comments explain *why*, not what. The codebase leans on module docstrings
  to state each module's contract (label policy lives in `signals.py`,
  composition only in `pipeline.py`, actuation only in `apply.py`, …) —
  preserve those invariants or update the docstring making the new contract
  explicit.
- Schema changes go in `db.SCHEMA` (idempotent `CREATE ... IF NOT EXISTS`);
  there is no migration framework yet, so additive changes only.

## Release checklist

1. Bump `__version__` in `src/gustarr/__init__.py`, `version` in
   `pyproject.toml` and the flake's `mkGustarr` version.
2. Update `CHANGELOG.md`.
3. Tag `vX.Y.Z` — CI builds and pushes `ghcr.io/dixiao-l/gustarr`.
