# Deployment

Three supported shapes, in order of polish:

1. **NixOS module** — hardened systemd timers, agenix-friendly secrets, GPU
   support. The reference deployment.
2. **Docker / compose** — CPU-only image, scheduling via host cron.
3. **Bare pip/uv + your own cron** — it's just a CLI; nothing stops you.

In every shape the moving parts are the same: a *nightly* pipeline run, an
optional distinct *weekly* run, and the long-running approval web UI.

---

## NixOS flake module

```nix
{
  inputs.gustarr.url = "github:Dixiao-L/gustarr";
  # in your nixosSystem modules:
  imports = [ inputs.gustarr.nixosModules.default ];
}
```

The module runs no thinking daemon: `gustarr-nightly` and `gustarr-weekly`
are oneshot services on timers, and the only long-running process is the web
UI. Units are hardened (`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`,
`NoNewPrivileges`) and run as the `gustarr` system user.

### Option reference (`services.gustarr.*`)

| option | type | default | meaning |
|---|---|---|---|
| `enable` | bool | `false` | enable the subsystem |
| `package` | package | flake's `default`, or `ml` when `gpu = true` | gustarr package to run |
| `gpu` | bool | `true` | use the CUDA `ml` variant and grant GPU device access (adds the `video` supplementary group, exposes `/run/opengl-driver/lib`) |
| `stateDir` | path | `"/var/lib/gustarr"` | store, model cache and HF downloads. The default uses systemd `StateDirectory`; any other path (e.g. a ZFS dataset covered by backups) is created via tmpfiles and granted with `ReadWritePaths` |
| `settings` | TOML attrs | `{ }` | gustarr.toml contents (`core.data_dir` is filled in from `stateDir`). Use `"env:VAR"` for every secret |
| `environmentFiles` | list of path | `[ ]` | `EnvironmentFile=` entries (agenix `.path` values) providing the `env:VAR` secrets |
| `extraEnvironment` | attrs of str | `{ }` | extra env vars for all units (e.g. `HF_ENDPOINT` for a HuggingFace mirror) |
| `nightly.enable` | bool | `true` | enable the nightly pipeline timer |
| `nightly.onCalendar` | str | `"*-*-* 04:30:00"` | `OnCalendar` for `gustarr run nightly` |
| `weekly.enable` | bool | `true` | enable the weekly pipeline timer |
| `weekly.onCalendar` | str | `"Sat *-*-* 09:00:00"` | `OnCalendar` for `gustarr run weekly` |
| `web.enable` | bool | `true` | run the approval web UI service |

Operational details worth knowing:

- **Secrets never enter the Nix store.** The generated TOML contains only
  `env:VAR` references; key material arrives exclusively via
  `EnvironmentFile=`. With agenix:

  ```nix
  age.secrets.gustarr-env.file = ./secrets/gustarr.env.age;
  services.gustarr.environmentFiles = [ config.age.secrets.gustarr-env.path ];
  ```

  where the decrypted file is plain `VAR=value` lines
  (`TMDB_API_KEY=...`, `SONARR_API_KEY=...`).
- **Timers** have `Persistent = true` (missed runs fire on boot) and a 10 min
  randomized delay. Pipeline units get `TimeoutStartSec = 2h` because the
  first run legitimately downloads the model and embeds the whole library.
- **Deploys don't restart mid-run pipelines** (`restartIfChanged = false`) —
  a 30-minute pipeline being restarted by activation once made deploy-rs roll
  back a healthy deploy. New config applies at the next timer fire.
- **"Run Now" in the web UI** touches a `run-requested` sentinel in the state
  dir; a systemd path unit watches for it and starts `gustarr-nightly`.
  Systemd is the privilege boundary — the unprivileged web process never runs
  the pipeline in-process.
- **Partial stage failures don't flip the unit red**: `gustarr run` exits
  non-zero only when every executed stage failed. Per-stage errors stay
  visible in the JSON on stdout (journal) and in `gustarr stats`.

---

## Docker Compose

Start from [`docker-compose.example.yml`](../docker-compose.example.yml):

```console
$ cp docker-compose.example.yml docker-compose.yml
$ cp gustarr.example.toml gustarr.toml
$ $EDITOR gustarr.toml       # URLs, usernames; keep secrets as env:VAR
$ $EDITOR gustarr.env        # TMDB_API_KEY=... etc. — chmod 600, gitignored
$ docker compose up -d gustarr
```

Container specifics:

- The image ships **CPU-only torch** (~2 GB, runs anywhere). Set
  `[model] device = "cpu"` in your TOML. GPU users should prefer the NixOS
  module, or build their own image with CUDA wheels — embedding is the only
  GPU stage and it's fine on CPU for typical library sizes, just slower on
  the first full run.
- Config is mounted at `/etc/gustarr/gustarr.toml` (a default search path, so
  no flag or env var is needed). State — SQLite store and HF model cache —
  lives in the `/var/lib/gustarr` volume; keep it on real disk and back it
  up, it *is* your taste model.
- To reach the UI from outside the container, set
  `[web] bind = "0.0.0.0:8790"`. The Host/Origin guard still applies: add the
  hostname you'll browse to (e.g. `nas.lan`) to `[web] allowed_hosts`, or
  you'll get 403s.
- The container runs as a non-root `gustarr` user (uid 1000). If you bind-mount
  a host directory instead of a named volume, `chown 1000` it.

### Scheduling: host cron instead of timers

There is deliberately **no scheduler in the container** — same Unix
philosophy as everywhere else. The web service is the only long-running
process; pipelines are one-shot `run` invocations of the same image. From the
host's crontab:

```cron
# nightly at 04:30, weekly Saturday 09:00 (paths: wherever your compose file lives)
30 4 * * *  cd /opt/gustarr && docker compose run --rm gustarr run nightly >> /var/log/gustarr.log 2>&1
0  9 * * 6  cd /opt/gustarr && docker compose run --rm gustarr run weekly  >> /var/log/gustarr.log 2>&1
```

Or a systemd timer on the host with
`ExecStart=docker compose -f /opt/gustarr/docker-compose.yml run --rm gustarr run nightly`.
The compose example also contains a commented one-shot `pipeline` service if
you prefer `docker compose run pipeline`.

SQLite is in WAL mode and the pipeline commits between stages, so a pipeline
run and the web UI sharing the volume is the normal, supported arrangement.

### Secrets guidance

- Put keys in an `env_file:` (`gustarr.env`), `chmod 600`, and keep it out of
  git — the TOML stays committable because it only holds `env:VAR`
  references.
- Compose `secrets:` or Podman secrets work too; anything that lands values
  in the process environment satisfies the `env:` convention.
- Never put keys in `docker-compose.yml` `environment:` blocks that get
  committed, and never in the image.

---

## Bare metal + cron

```console
$ uv sync --extra ml    # or: pip install .[ml]
$ crontab -e
30 4 * * * cd ~/gustarr && env $(cat gustarr.env | xargs) uv run gustarr run nightly
```

Same knobs, no wrapper. `gustarr web` behind your reverse proxy of choice
(remember `[web] allowed_hosts`).

---

## Monitoring

- `gustarr stats` and `GET /api/stats` return the same JSON: table counts,
  events by kind, queue states, sync cursors, model freshness, embedding
  coverage, and the diversity block (genre entropy of recommendations vs.
  library, exploration share and approval rate). Scrape it with Telegraf's
  `inputs.http` or any JSON-capable agent and graph it in Grafana.
- Pipeline runs print a per-stage stats JSON to stdout — the journal (NixOS)
  or your cron log (Docker) is the run history.
- The compose example ships a healthcheck against `/api/stats`.
