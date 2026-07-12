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
    profile: str = "default",
) -> list[dict[str, Any]]:
    sql = (
        "SELECT r.id, r.profile, r.ts, r.domain, r.item_id, r.score, r.why, r.status,"
        " r.acted_at, i.title, i.year, i.ids, i.meta"
        " FROM recommendations r JOIN items i ON i.id = r.item_id"
    )
    # Always one profile's queue: 'default' keeps single-user setups (and
    # callers written before profiles existed) seeing everything they did.
    where: list[str] = ["r.profile = ?"]
    params: list[Any] = [profile]
    if status != "all":
        where.append("r.status = ?")
        params.append(status)
    if domain == "music":
        # 'music' is the UI's tab, never a stored domain — expanding the
        # alias here gives every caller (web, CLI) the same behaviour.
        where.append("r.domain IN ('artist', 'album')")
    elif domain:
        where.append("r.domain = ?")
        params.append(domain)
    sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.score DESC, r.id"

    rows: list[dict[str, Any]] = []
    for r in conn.execute(sql, params):
        meta = json.loads(r["meta"])
        overview = meta.get("overview")
        rows.append({
            "id": r["id"],
            "profile": r["profile"],
            "domain": r["domain"],
            "item_id": r["item_id"],
            "title": r["title"],
            "year": r["year"],
            "ids": json.loads(r["ids"]),
            "genres": meta.get("genres") or [],
            "poster_path": meta.get("poster_path"),
            # Music cards need these: image is a direct URL (no TMDB prefix),
            # artist feeds the album card's "by <artist>" byline, and type
            # ("Album"/"Single"/"EP") feeds the card's type chip.
            "image": meta.get("image"),
            "artist": meta.get("artist"),
            "type": meta.get("type"),
            "overview": overview[:220] if overview else None,
            "trailer": meta.get("trailer"),
            "score": r["score"],
            "why": json.loads(r["why"] or "{}"),
            "status": r["status"],
            "ts": r["ts"],
            "acted_at": r["acted_at"],
        })
    return rows


def set_status(
    conn: sqlite3.Connection, rec_id: int, status: str, profile: str | None = None,
) -> dict[str, int]:
    if status not in ("approved", "rejected", "snoozed"):
        raise ValueError(f"only approved/rejected/snoozed can be set here, not {status!r}")
    row = conn.execute(
        "SELECT item_id, profile, status, why FROM recommendations WHERE id=?",
        (rec_id,)).fetchone()
    if row is None:
        raise ValueError(f"no recommendation #{rec_id}")
    _check_profile(row, rec_id, profile)
    if row["status"] in TERMINAL_STATUSES:
        raise ValueError(f"recommendation #{rec_id} is already {row['status']}; too late")
    if row["status"] == status:
        raise ValueError(f"recommendation #{rec_id} is already {status}")
    if status == "snoozed" and row["status"] != "proposed":
        raise ValueError(
            f"recommendation #{rec_id} is {row['status']}; only proposed recs can be snoozed")

    ts = db.now()
    conn.execute(
        "UPDATE recommendations SET status=?, acted_at=? WHERE id=?", (status, ts, rec_id))
    if status == "snoozed":
        # "Not now" is not a verdict — deliberately no taste event, so the
        # model learns nothing from a snooze. Rank's expiry pass flips
        # snoozed→expired after the TTL, making the item proposable again.
        return {"updated": 1, "events": 0}
    kind = "approve" if status == "approved" else "reject"
    weight = signals.WEIGHTS[kind]
    meta: dict[str, Any] = {"rec_id": rec_id}
    if kind == "reject" and json.loads(row["why"] or "{}").get("exploration"):
        # Exploration picks are deliberate off-policy probes; a full-strength
        # reject would teach the model never to leave the bubble again.
        # Approvals stay full weight — a hit is a hit however it was found.
        weight *= 0.3
        meta["exploration"] = True
    # The verdict trains the rec owner's model, so the event carries the
    # rec's own profile — never the caller-supplied guard value.
    added = db.add_event(conn, ts, row["item_id"], kind, weight, "user", meta=meta,
                         profile=row["profile"])
    return {"updated": 1, "events": int(added)}


def _check_profile(row: sqlite3.Row, rec_id: int, profile: str | None) -> None:
    """A rec id already implies its profile, so profile here is a guard,
    not a lookup key: None (the default) trusts the id, anything else
    must match — catches a --profile flag pointed at someone else's rec."""
    if profile is not None and row["profile"] != profile:
        raise ValueError(
            f"recommendation #{rec_id} belongs to profile {row['profile']!r}, not {profile!r}")


def forgive(
    conn: sqlite3.Connection, rec_id: int, profile: str | None = None,
) -> dict[str, int]:
    """Undo a reject: delete that rec's reject event(s) so training stops
    penalising the item, and flip the rec to 'expired' so rank may
    propose it again. Only this rec's events go — a reject of the same
    item via another recommendation is a separate verdict and stays."""
    row = conn.execute(
        "SELECT status, profile FROM recommendations WHERE id=?", (rec_id,)).fetchone()
    if row is None:
        raise ValueError(f"no recommendation #{rec_id}")
    _check_profile(row, rec_id, profile)
    if row["status"] != "rejected":
        raise ValueError(f"recommendation #{rec_id} is {row['status']}; only rejected"
                         " recs can be forgiven")

    cur = conn.execute(
        "DELETE FROM events WHERE kind='reject' AND json_extract(meta, '$.rec_id') = ?",
        (rec_id,))
    conn.execute(
        "UPDATE recommendations SET status='expired', acted_at=? WHERE id=?",
        (db.now(), rec_id))
    return {"deleted_events": cur.rowcount}


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


def explain(conn: sqlite3.Connection, rec_id: int, profile: str | None = None) -> str:
    row = conn.execute(
        "SELECT r.id, r.profile, r.domain, r.item_id, r.score, r.why, r.status,"
        " i.title, i.year, i.meta"
        " FROM recommendations r JOIN items i ON i.id = r.item_id WHERE r.id=?",
        (rec_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"no recommendation #{rec_id}")
    _check_profile(row, rec_id, profile)

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


def _diversity(conn: sqlite3.Connection, profile: str) -> dict[str, Any]:
    """Is this profile's queue actually wider than the (shared) library
    it feeds on? Recs are per person; the library baseline is global."""
    recent = conn.execute(
        "SELECT r.why, i.meta FROM recommendations r JOIN items i ON i.id = r.item_id"
        " WHERE r.profile=? ORDER BY r.id DESC LIMIT 100", (profile,)).fetchall()
    n_explore = sum(1 for r in recent if json.loads(r["why"] or "{}").get("exploration"))
    acted = positive = 0
    for r in conn.execute(
            "SELECT status, why FROM recommendations WHERE profile=?"
            " AND status IN ('approved','rejected','added','auto_added')", (profile,)):
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


def store_stats(conn: sqlite3.Connection, profile: str = "default") -> dict[str, Any]:
    # Table totals stay global (one store, one disk); anything that is a
    # judgement about a person — rec counts, model freshness, sync
    # cursors, diversity — is reported for the requested profile.
    tables = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    events_by_kind = {
        r[0]: r[1]
        for r in conn.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind ORDER BY kind")
    }
    recs_by_status = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT status, COUNT(*) FROM recommendations WHERE profile=?"
            " GROUP BY status ORDER BY status", (profile,))
    }

    sync: dict[str, Any] = {}
    uts = db.pget_state(conn, profile, "lastfm:last_uts")
    if uts is not None:
        sync["lastfm:last_uts"] = uts
    for r in conn.execute(
            "SELECT key FROM state WHERE key LIKE 'arr:known:%' ORDER BY key"):
        # arr cursors are global (one arr inventory) and big id maps —
        # presence is the interesting bit.
        sync[r["key"]] = True

    models: dict[str, Any] = {}
    prefix = db.pkey(profile, "model:")
    for r in conn.execute("SELECT key, value FROM state WHERE key LIKE ? ORDER BY key",
                          (prefix + "%",)):
        try:
            val: Any = json.loads(r["value"])
        except ValueError:
            val = r["value"]
        if isinstance(val, dict):
            val = val.get("trained_at", val)
        models[r["key"].removeprefix(prefix)] = val

    interesting = "SELECT item_id FROM events UNION SELECT item_id FROM candidates"
    needed = conn.execute(f"SELECT COUNT(*) FROM ({interesting})").fetchone()[0]
    embedded = conn.execute(
        f"SELECT COUNT(*) FROM ({interesting})"
        " WHERE item_id IN (SELECT item_id FROM embeddings)").fetchone()[0]

    return {
        "profile": profile,
        "tables": tables,
        "events_by_kind": events_by_kind,
        "recs_by_status": recs_by_status,
        "sync": sync,
        "models": models,
        "settings_overridden": conn.execute(
            "SELECT COUNT(*) FROM state WHERE key LIKE 'setting:%'").fetchone()[0],
        "embedding_coverage": {"needed": needed, "embedded": embedded},
        "diversity": _diversity(conn, profile),
    }
