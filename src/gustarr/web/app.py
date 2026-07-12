"""Approval web UI: a thin FastAPI shell around queue.py.

No auth by design: Gustarr binds 127.0.0.1 by default (``[web] bind``,
see cli.py); any wider exposure happens intranet-only behind Traefik,
which owns TLS and access control. A Host/Origin guard still runs on
every request (see ``guard`` below) because a localhost bind alone does
not stop the user's own browser: extra hostnames (e.g. the Traefik one)
go in ``[web] allowed_hosts``.

Multi-user households resolve a *profile* per request instead of adding
auth here: a forward-auth proxy (Authelia) stamps the identity header
named by ``[web] profile_header`` (default ``Remote-User``), with
``?profile=`` as the header-less fallback. The header is trusted as-is,
so a multi-profile instance must only be reachable through the proxy
that sets it. Runtime settings stay operator-global — one store, one
budget, every profile.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from .. import db, queue, settings
from ..config import Config

_INDEX = Path(__file__).parent / "static" / "index.html"


def _hostname(value: str) -> str | None:
    """Lowercased host from a Host header, Origin, or bind string; port-insensitive."""
    try:
        # urlsplit needs '//' to treat scheme-less values ("127.0.0.1:8790") as netloc.
        return urlsplit(value if "//" in value else f"//{value}").hostname
    except ValueError:
        return None


def _allowed_hosts(cfg: Config) -> set[str]:
    allowed = {"127.0.0.1", "localhost"}
    for entry in [cfg.web.get("bind", "127.0.0.1:8790"), *cfg.web.get("allowed_hosts", [])]:
        host = _hostname(str(entry))
        if host:
            allowed.add(host)
    return allowed


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Gustarr", docs_url=None, redoc_url=None)
    allowed = _allowed_hosts(cfg)
    # No scheduling here by design: the web process serves the approval UI
    # and nothing else. `gustarr schedule` is its own process.
    profiles = list(cfg.profiles) or ["default"]
    bind_host = _hostname(str(cfg.web.get("bind", "127.0.0.1:8790"))) or "127.0.0.1"
    if len(profiles) > 1 and bind_host not in ("127.0.0.1", "::1", "localhost"):
        # Gustarr never authenticates; it maps identities the deployment
        # vouches for. Say so where the operator will see it.
        print("gustarr web: multiple profiles on a non-loopback bind — profile"
              " identity is only as trustworthy as the path to this port"
              " (authenticating reverse proxy setting"
              f" {cfg.web.get('profile_header', 'Remote-User')!r}, or network"
              " isolation). See docs/security.md.", flush=True)
    profile_header = cfg.web.get("profile_header", "Remote-User")

    def get_profile(request: Request) -> str:
        # Resolution order: forward-auth header (Authelia sets it), explicit
        # query param, then the sole configured profile. An explicit name
        # must match the config — a typo'd or unmapped user seeing the
        # 'default' queue would silently cross-pollinate taste models.
        for name in (request.headers.get(profile_header),
                     request.query_params.get("profile")):
            if name:
                if name not in profiles:
                    raise HTTPException(
                        status_code=403,
                        detail=f"unknown profile {name!r}"
                               f" (configured: {', '.join(profiles)})")
                return name
        return profiles[0] if len(profiles) == 1 else "default"

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # The 127.0.0.1 bind doesn't stop the user's own browser acting as a
        # confused deputy: a foreign Host header means DNS rebinding, and a
        # cross-site page can fire Origin-carrying simple POSTs at localhost.
        if _hostname(request.headers.get("host", "")) not in allowed:
            return JSONResponse({"detail": "unrecognized Host header"}, status_code=403)
        origin = request.headers.get("origin")
        # Absent Origin = same-origin navigation or CLI client; allowed.
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin is not None:
            if _hostname(origin) not in allowed:
                return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
        return await call_next(request)

    def get_conn() -> Iterator[sqlite3.Connection]:
        # One connection per request: same-machine SQLite opens are cheap,
        # and never holding one across requests keeps WAL locks short.
        conn = db.connect(cfg.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def act(conn: sqlite3.Connection, rec_id: int, status: str) -> dict[str, Any]:
        try:
            stats = queue.set_status(conn, rec_id, status)
        except ValueError as exc:  # unknown rec / already acted / terminal status
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        conn.commit()
        return {"id": rec_id, "status": status, **stats}

    @app.get("/api/profile")
    def api_profile(profile: str = Depends(get_profile)) -> dict[str, Any]:
        # Feeds the header chip: which queue this request sees, and whether
        # showing that is even interesting (>1 configured profile).
        return {"name": profile, "profiles": profiles}

    @app.get("/api/recs")
    def api_recs(
        status: str = "proposed",
        domain: str | None = None,
        profile: str = Depends(get_profile),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> list[dict[str, Any]]:
        # domain='music' expands to artist+album inside list_recs, so the
        # web UI and the CLI share one alias implementation.
        return queue.list_recs(conn, domain=domain or None, status=status, profile=profile)

    @app.post("/api/recs/{rec_id}/approve")
    def api_approve(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return act(conn, rec_id, "approved")

    @app.post("/api/recs/{rec_id}/reject")
    def api_reject(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return act(conn, rec_id, "rejected")

    @app.post("/api/recs/{rec_id}/snooze")
    def api_snooze(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return act(conn, rec_id, "snoozed")

    @app.post("/api/recs/{rec_id}/forgive")
    def api_forgive(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        try:
            stats = queue.forgive(conn, rec_id)
        except ValueError as exc:  # unknown rec / not rejected
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        conn.commit()
        return {"id": rec_id, "status": "expired", **stats}

    @app.get("/api/recs/{rec_id}/why")
    def api_why(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, str]:
        try:
            return {"text": queue.explain(conn, rec_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/stats")
    def api_stats(
        profile: str = Depends(get_profile),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, Any]:
        # Cursors, models and the diversity block are per-profile now;
        # table counts and budgets in the payload stay store-global.
        return queue.store_stats(conn, profile=profile)

    @app.get("/api/settings")
    def api_settings(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return settings.get_all(conn, cfg)

    @app.put("/api/settings/{key}")
    def api_settings_set(
        key: str, payload: dict[str, Any], conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, Any]:
        if "value" not in payload:
            raise HTTPException(status_code=400, detail="body must be {\"value\": ...}")
        try:
            value = settings.set(conn, key, payload["value"])
        except ValueError as exc:  # unknown key / invalid value
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        conn.commit()
        return {"key": key, "value": value}

    @app.delete("/api/settings/{key}")
    def api_settings_clear(
        key: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, Any]:
        try:
            settings.clear(conn, key)
        except ValueError as exc:  # unknown key
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        conn.commit()
        return {"key": key, "cleared": True}

    @app.post("/api/run")
    def api_run() -> dict[str, bool]:
        # The sentinel file is the whole IPC: a systemd path unit on the
        # host watches data_dir and starts the pipeline when it appears,
        # so the web process never runs the pipeline in-process.
        sentinel = Path(cfg.data_dir) / "run-requested"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch(mode=0o644)
        return {"requested": True}

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX.read_text(encoding="utf-8"))

    return app
