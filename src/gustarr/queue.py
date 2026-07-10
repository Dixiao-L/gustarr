"""The approval queue: list, approve/reject, explain, and store overview.

Approve/reject is itself a taste event (the strongest one we get, see
signals.WEIGHTS) so the model learns from every verdict immediately —
the status flip and the event are written together here.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Iterable

from . import db, ids, signals

# Statuses set by `apply` (or auto mode) — the item already left the
# queue's control, so a late approve/reject would be a lie.
TERMINAL_STATUSES = {"added", "auto_added", "failed"}

_TABLES = ("items", "events", "library", "candidates", "recommendations", "embeddings", "state")


def list_recs(
    conn: sqlite3.Connection,
    domain: str | None = None,
    status: str = "proposed",
) -> list[dict[str, Any]]:
    sql = (
        "SELECT r.id, r.ts, r.domain, r.item_id, r.score, r.why, r.status, r.acted_at,"
        " i.title, i.year, i.ids, i.meta"
        " FROM recommendations r JOIN items i ON i.id = r.item_id"
    )
    where: list[str] = []
    params: list[Any] = []
    if status != "all":
        where.append("r.status = ?")
        params.append(status)
    if domain:
        where.append("r.domain = ?")
        params.append(domain)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.score DESC, r.id"

    rows: list[dict[str, Any]] = []
    for r in conn.execute(sql, params):
        meta = json.loads(r["meta"])
        overview = meta.get("overview")
        rows.append({
            "id": r["id"],
            "domain": r["domain"],
            "item_id": r["item_id"],
            "title": r["title"],
            "year": r["year"],
            "ids": json.loads(r["ids"]),
            "genres": meta.get("genres") or [],
            "poster_path": meta.get("poster_path"),
            "overview": overview[:220] if overview else None,
            "trailer": meta.get("trailer"),
            "score": r["score"],
            "why": json.loads(r["why"] or "{}"),
            "status": r["status"],
            "ts": r["ts"],
            "acted_at": r["acted_at"],
        })
    return rows


def set_status(conn: sqlite3.Connection, rec_id: int, status: str) -> dict[str, int]:
    if status not in ("approved", "rejected"):
        raise ValueError(f"only approved/rejected can be set here, not {status!r}")
    row = conn.execute(
        "SELECT item_id, status, why FROM recommendations WHERE id=?", (rec_id,)).fetchone()
    if row is None:
        raise ValueError(f"no recommendation #{rec_id}")
    if row["status"] in TERMINAL_STATUSES:
        raise ValueError(f"recommendation #{rec_id} is already {row['status']}; too late")
    if row["status"] == status:
        raise ValueError(f"recommendation #{rec_id} is already {status}")

    ts = db.now()
    conn.execute(
        "UPDATE recommendations SET status=?, acted_at=? WHERE id=?", (status, ts, rec_id))
    kind = "approve" if status == "approved" else "reject"
    weight = signals.WEIGHTS[kind]
    meta: dict[str, Any] = {"rec_id": rec_id}
    if kind == "reject" and json.loads(row["why"] or "{}").get("exploration"):
        # Exploration picks are deliberate off-policy probes; a full-strength
        # reject would teach the model never to leave the bubble again.
        # Approvals stay full weight — a hit is a hit however it was found.
        weight *= 0.3
        meta["exploration"] = True
    added = db.add_event(conn, ts, row["item_id"], kind, weight, "user", meta=meta)
    return {"updated": 1, "events": int(added)}


# ── explanations ─────────────────────────────────────────────────────


def _title_for(conn: sqlite3.Connection, ref: Any) -> str:
    """Best display name for a why-json reference: dict / item id / plain text."""
    if isinstance(ref, dict):
        return ref.get("title") or _title_for(conn, ref.get("item_id") or ref.get("id") or "?")
    ref = str(ref)
    if ":" not in ref:
        return ref
    row = conn.execute("SELECT title FROM items WHERE id=?", (ref,)).fetchone()
    if row and row["title"]:
        return row["title"]
    try:
        _, _, key = ids.parse(ref)
    except ValueError:
        return ref
    return key.replace("\x1f", " — ")


def _fmt_sim(sim: Any) -> str:
    return f"{float(sim):.2f}".removeprefix("0")


def _neighbour_line(conn: sqlite3.Connection, nb: Any) -> str:
    sim = None
    if isinstance(nb, dict):
        sim = nb.get("sim", nb.get("score"))
        title = _title_for(conn, nb)
    elif isinstance(nb, (list, tuple)) and len(nb) == 2:
        title, sim = _title_for(conn, nb[0]), nb[1]
    else:
        title = _title_for(conn, nb)
    if sim is None:
        return f"because you liked {title}"
    return f"because you liked {title} (sim {_fmt_sim(sim)})"


def explain(conn: sqlite3.Connection, rec_id: int) -> str:
    row = conn.execute(
        "SELECT r.id, r.domain, r.item_id, r.score, r.why, r.status,"
        " i.title, i.year, i.meta"
        " FROM recommendations r JOIN items i ON i.id = r.item_id WHERE r.id=?",
        (rec_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no recommendation #{rec_id}")

    why = json.loads(row["why"] or "{}")
    meta = json.loads(row["meta"])
    header = (
        f"#{row['id']} {row['title'] or row['item_id']} ({row['year'] or '?'}) — {row['domain']}"
    )
    if why.get("serendipity"):
        header += " (serendipity)"
    lines = [header]
    genres = meta.get("genres") or []
    if genres:
        lines.append("genres: " + ", ".join(map(str, genres)))
    lines.append(f"score: {row['score']:+.2f} [{row['status']}]")
    sources = why.get("sources") or []
    if sources:
        lines.append("sources: " + ", ".join(map(str, sources)))
    seeds = why.get("seeds") or []
    if seeds:
        lines.append("seeded by: " + ", ".join(_title_for(conn, s) for s in seeds))
    for nb in why.get("neighbors") or why.get("neighbours") or []:
        lines.append(_neighbour_line(conn, nb))
    if why.get("exploration"):
        lines.append("exploration pick: chosen to widen the model's signal, not by top score")
    return "\n".join(lines)


# ── store overview ───────────────────────────────────────────────────


def _genre_entropy(metas: Iterable[str]) -> float:
    """Shannon entropy (bits, 3dp) of the genre histogram over items.meta
    json strings; genre-less items contribute nothing."""
    counts: dict[str, int] = {}
    for meta in metas:
        for g in json.loads(meta).get("genres") or []:
            counts[str(g)] = counts.get(str(g), 0) + 1
    total = sum(counts.values())
    if not total:
        return 0.0
    return round(-sum(n / total * math.log2(n / total) for n in counts.values()), 3)


def _diversity(conn: sqlite3.Connection) -> dict[str, Any]:
    """Is the queue actually wider than the library it feeds on?"""
    recent = conn.execute(
        "SELECT r.why, i.meta FROM recommendations r JOIN items i ON i.id = r.item_id"
        " ORDER BY r.id DESC LIMIT 100").fetchall()
    n_explore = sum(1 for r in recent if json.loads(r["why"] or "{}").get("exploration"))
    acted = positive = 0
    for r in conn.execute(
            "SELECT status, why FROM recommendations"
            " WHERE status IN ('approved','rejected','added','auto_added')"):
        if json.loads(r["why"] or "{}").get("exploration"):
            acted += 1
            # 'added' implies a prior approve; 'auto_added' skipped the user,
            # so it counts as acted but never as an approval.
            positive += r["status"] in ("approved", "added")
    return {
        "genre_entropy_recs": _genre_entropy(r["meta"] for r in recent),
        "genre_entropy_library": _genre_entropy(r["meta"] for r in conn.execute(
            "SELECT i.meta FROM library l JOIN items i ON i.id = l.item_id")),
        "exploration_share": round(n_explore / len(recent), 3) if recent else 0.0,
        "exploration_approval_rate": round(positive / acted, 3) if acted else None,
    }


def store_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    events_by_kind = {
        r[0]: r[1]
        for r in conn.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind ORDER BY kind")
    }
    recs_by_status = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT status, COUNT(*) FROM recommendations GROUP BY status ORDER BY status")
    }

    sync: dict[str, Any] = {}
    for r in conn.execute(
            "SELECT key, value FROM state"
            " WHERE key = 'lastfm:last_uts' OR key LIKE 'arr:known:%' ORDER BY key"):
        # arr cursors are big id maps — presence is the interesting bit.
        sync[r["key"]] = r["value"] if r["key"] == "lastfm:last_uts" else True

    models: dict[str, Any] = {}
    for r in conn.execute("SELECT key, value FROM state WHERE key LIKE 'model:%' ORDER BY key"):
        try:
            val: Any = json.loads(r["value"])
        except ValueError:
            val = r["value"]
        if isinstance(val, dict):
            val = val.get("trained_at", val)
        models[r["key"].removeprefix("model:")] = val

    interesting = "SELECT item_id FROM events UNION SELECT item_id FROM candidates"
    needed = conn.execute(f"SELECT COUNT(*) FROM ({interesting})").fetchone()[0]
    embedded = conn.execute(
        f"SELECT COUNT(*) FROM ({interesting})"
        " WHERE item_id IN (SELECT item_id FROM embeddings)").fetchone()[0]

    return {
        "tables": tables,
        "events_by_kind": events_by_kind,
        "recs_by_status": recs_by_status,
        "sync": sync,
        "models": models,
        "embedding_coverage": {"needed": needed, "embedded": embedded},
        "diversity": _diversity(conn),
    }
