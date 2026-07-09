"""Approval web UI: a thin FastAPI shell around queue.py.

No auth by design: gustarr is single-user and binds 127.0.0.1 by default
(``[web] bind``, see cli.py); any wider exposure happens intranet-only
behind Traefik, which owns TLS and access control.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .. import db, queue
from ..config import Config

_INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="gustarr", docs_url=None, redoc_url=None)

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

    @app.get("/api/recs")
    def api_recs(
        status: str = "proposed",
        domain: str | None = None,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> list[dict[str, Any]]:
        return queue.list_recs(conn, domain=domain or None, status=status)

    @app.post("/api/recs/{rec_id}/approve")
    def api_approve(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return act(conn, rec_id, "approved")

    @app.post("/api/recs/{rec_id}/reject")
    def api_reject(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return act(conn, rec_id, "rejected")

    @app.get("/api/recs/{rec_id}/why")
    def api_why(rec_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, str]:
        try:
            return {"text": queue.explain(conn, rec_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/stats")
    def api_stats(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        return queue.store_stats(conn)

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX.read_text(encoding="utf-8"))

    return app
