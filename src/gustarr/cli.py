"""The gustarr command. One subcommand per atomic function; ``run`` is
the only composition. Heavy imports (torch) happen inside the commands
that need them so `gustarr recs` stays instant.
"""

from __future__ import annotations

import json
import sys

import click

from . import __version__, config as config_mod, db as db_mod


class Ctx:
    def __init__(self, config_path: str | None):
        self._config_path = config_path
        self._cfg = None
        self._conn = None

    @property
    def cfg(self):
        if self._cfg is None:
            self._cfg = config_mod.load(self._config_path)
        return self._cfg

    @property
    def conn(self):
        if self._conn is None:
            self._conn = db_mod.connect(self.cfg.db_path)
        return self._conn


@click.group()
@click.version_option(__version__)
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to gustarr.toml (default: $GUSTARR_CONFIG, ./gustarr.toml, /etc/gustarr/)")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """Learns your media taste, recommends, and drives Sonarr/Radarr/Lidarr."""
    ctx.obj = Ctx(config_path)


# ── sync ─────────────────────────────────────────────────────────────


@main.group()
def sync() -> None:
    """Pull taste signals into the store (idempotent, incremental)."""


@sync.command("jellyfin")
@click.pass_obj
def sync_jellyfin(ctx: Ctx) -> None:
    """Watch/listen history + favorites from Jellyfin."""
    from .collect import jellyfin

    stats = jellyfin.sync(ctx.conn, ctx.cfg)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@sync.command("lastfm")
@click.option("--full", is_flag=True, help="Ignore cursor, re-walk entire scrobble history.")
@click.pass_obj
def sync_lastfm(ctx: Ctx, full: bool) -> None:
    """Scrobbles, loved tracks and top artists from Last.fm."""
    from .collect import lastfm

    stats = lastfm.sync(ctx.conn, ctx.cfg, full=full)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@sync.command("listenbrainz")
@click.pass_obj
def sync_listenbrainz(ctx: Ctx) -> None:
    """Collaborative-filtering recommendations from ListenBrainz (candidates)."""
    from .collect import listenbrainz

    stats = listenbrainz.sync(ctx.conn, ctx.cfg)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@sync.command("arr")
@click.pass_obj
def sync_arr(ctx: Ctx) -> None:
    """Inventory Sonarr/Radarr/Lidarr: library + adds-as-signal + gustarr-tag feedback."""
    from .collect import arr

    stats = arr.sync(ctx.conn, ctx.cfg)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


# ── pipeline stages ──────────────────────────────────────────────────


@main.command()
@click.option("--domain", type=click.Choice(["movie", "series", "artist", "album", "track"]),
              default=None, help="Restrict to one domain.")
@click.option("--limit", type=int, default=0, help="Max items to enrich this run (0 = all).")
@click.pass_obj
def enrich(ctx: Ctx, domain: str | None, limit: int) -> None:
    """Fill item metadata from TMDb / MusicBrainz; resolve fallback ids."""
    from .enrich import run as enrich_run

    stats = enrich_run(ctx.conn, ctx.cfg, domain=domain, limit=limit or None)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@main.command()
@click.option("--domain", default=None)
@click.pass_obj
def candidates(ctx: Ctx, domain: str | None) -> None:
    """Refresh the candidate pool (TMDb similar/discover, Last.fm similar, LB CF)."""
    from .candidates import run as candidates_run

    stats = candidates_run(ctx.conn, ctx.cfg, domain=domain)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@main.command()
@click.pass_obj
def embed(ctx: Ctx) -> None:
    """Embed items lacking vectors (GPU if configured)."""
    from .ml.embed import run as embed_run

    stats = embed_run(ctx.conn, ctx.cfg)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@main.command()
@click.pass_obj
def train(ctx: Ctx) -> None:
    """Fit per-domain preference heads from events + embeddings."""
    from .ml.train import run as train_run

    stats = train_run(ctx.conn, ctx.cfg)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@main.command()
@click.option("--top", type=int, default=20, help="Slots per domain.")
@click.pass_obj
def rank(ctx: Ctx, top: int) -> None:
    """Score candidates → write proposed recommendations with explanations."""
    from .ml.rank import run as rank_run

    stats = rank_run(ctx.conn, ctx.cfg, top=top)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


@main.command()
@click.option("--dry-run", is_flag=True, help="Print what would be actuated, change nothing.")
@click.pass_obj
def apply(ctx: Ctx, dry_run: bool) -> None:
    """Actuate: auto-add music (capped), push approved video, sync Jellyfin collections."""
    from .actuate.apply import run as apply_run

    stats = apply_run(ctx.conn, ctx.cfg, dry_run=dry_run)
    ctx.conn.commit()
    click.echo(json.dumps(stats))


# ── queue management ─────────────────────────────────────────────────


@main.command()
@click.option("--domain", default=None)
@click.option("--status", default="proposed")
@click.option("--profile", default="default", help="Whose queue to list.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def recs(ctx: Ctx, domain: str | None, status: str, profile: str, as_json: bool) -> None:
    """List recommendations (default: the open approval queue)."""
    from .queue import list_recs

    rows = list_recs(ctx.conn, domain=domain, status=status, profile=profile)
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for r in rows:
        click.echo(f"#{r['id']:<5} {r['domain']:<7} {r['score']:+.2f}  "
                   f"{r['title']} ({r['year'] or '?'})  [{r['status']}]")


# A rec id already implies its profile, so --profile on the id-taking
# commands is a guard, not a selector: given, it must match the rec's
# owner (catches acting on someone else's queue by mistyped id); omitted,
# the id alone is trusted.
_PROFILE_GUARD = click.option(
    "--profile", default=None,
    help="Fail unless the recommendation belongs to this profile.")


@main.command()
@click.argument("rec_ids", type=int, nargs=-1, required=True)
@_PROFILE_GUARD
@click.pass_obj
def approve(ctx: Ctx, rec_ids: tuple[int, ...], profile: str | None) -> None:
    """Approve queued recommendations (adds on next `apply`; feeds training)."""
    from .queue import set_status

    for rid in rec_ids:
        set_status(ctx.conn, rid, "approved", profile=profile)
    ctx.conn.commit()
    click.echo(f"approved: {', '.join(map(str, rec_ids))}")


@main.command()
@click.argument("rec_ids", type=int, nargs=-1, required=True)
@_PROFILE_GUARD
@click.pass_obj
def reject(ctx: Ctx, rec_ids: tuple[int, ...], profile: str | None) -> None:
    """Reject recommendations — the model learns from this immediately."""
    from .queue import set_status

    for rid in rec_ids:
        set_status(ctx.conn, rid, "rejected", profile=profile)
    ctx.conn.commit()
    click.echo(f"rejected: {', '.join(map(str, rec_ids))}")


@main.command()
@click.argument("rec_id", type=int)
@_PROFILE_GUARD
@click.pass_obj
def why(ctx: Ctx, rec_id: int, profile: str | None) -> None:
    """Explain a recommendation: nearest liked neighbours + sources."""
    from .queue import explain

    click.echo(explain(ctx.conn, rec_id, profile=profile))


@main.command()
@click.option("--profile", default="default", help="Profile for the per-person numbers.")
@click.pass_obj
def stats(ctx: Ctx, profile: str) -> None:
    """Store overview: events, items, candidates, queue, model freshness."""
    from .queue import store_stats

    click.echo(json.dumps(store_stats(ctx.conn, profile=profile), indent=2))


# ── composition + web ────────────────────────────────────────────────


@main.command()
@click.argument("recipe", type=click.Choice(["nightly", "weekly"]))
@click.option("--dry-run", is_flag=True)
@click.pass_obj
def run(ctx: Ctx, recipe: str, dry_run: bool) -> None:
    """Full pipeline: sync→enrich→candidates→embed→train→rank→apply. Both recipes are identical; weekly is kept as a timer alias."""
    from .pipeline import run_recipe

    stats = run_recipe(ctx.conn, ctx.cfg, recipe, dry_run=dry_run)
    click.echo(json.dumps(stats))
    # Partial stage errors are operational normal (an unconfigured or
    # flaky source shouldn't flip the systemd unit red nightly — that
    # noise once made deploy-rs roll back a healthy deploy). They stay
    # visible in the stats JSON, journal and telegraf; the unit fails
    # only when nothing ran to completion.
    executed = [k for k, v in stats.items() if k != "errors" and v != "skipped"]
    if stats.get("errors") and len(stats["errors"]) == len(executed):
        sys.exit(1)


@main.command()
@click.pass_obj
def schedule(ctx: Ctx) -> None:
    """Run the nightly pipeline on a clock ([scheduler] nightly = "HH:MM").

    A dedicated foreground process — container users run it as a second
    service from the same image; systemd/cron users don't need it."""
    from .scheduler import main as schedule_main

    schedule_main(ctx.cfg)


@main.command()
@click.pass_obj
def web(ctx: Ctx) -> None:
    """Serve the approval UI (bind address from [web] config)."""
    import uvicorn

    from .web.app import create_app

    bind = ctx.cfg.web.get("bind", "127.0.0.1:8790")
    host, _, port = bind.rpartition(":")
    uvicorn.run(create_app(ctx.cfg), host=host or "127.0.0.1", port=int(port))
