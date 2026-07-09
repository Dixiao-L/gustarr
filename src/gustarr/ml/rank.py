"""Score the candidate pool and queue proposals.

MMR keeps a run's picks from clustering on one taste mode; a slice of
each run is reserved for exploration so ranking never goes fully
exploit-only — the model needs off-policy feedback to keep learning.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import db
from ..config import Config

# Music autonomy caps are low; a 20-deep artist queue just goes stale.
ARTIST_TOP = 10
SOURCE_BONUS = 0.03
EXT_BONUS = 0.05
NEG_PULL = 0.6
EXPLORE_BAND = (40.0, 90.0)  # score percentiles: novel-ish but not junk


def _state_json(conn: sqlite3.Connection, key: str) -> dict | None:
    raw = db.get_state(conn, key)
    return json.loads(raw) if raw else None


def _vec(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _pool(conn: sqlite3.Connection, domain: str) -> dict[str, dict]:
    """Candidates still worth proposing: not owned, not already queued,
    never explicitly rejected."""
    rows = conn.execute(
        "SELECT c.item_id, c.source, c.external_score, c.seed_item_id"
        " FROM candidates c JOIN items i ON i.id = c.item_id"
        " WHERE i.domain = ?"
        "   AND c.item_id NOT IN (SELECT item_id FROM library)"
        "   AND c.item_id NOT IN (SELECT item_id FROM recommendations"
        "                         WHERE status IN ('proposed','approved'))"
        "   AND c.item_id NOT IN (SELECT item_id FROM events WHERE kind='reject')",
        (domain,),
    )
    info: dict[str, dict] = {}
    for r in rows:
        d = info.setdefault(r["item_id"], {"sources": set(), "ext": 0.0, "seeds": set()})
        d["sources"].add(r["source"])
        if r["external_score"] is not None:
            d["ext"] = max(d["ext"], float(r["external_score"]))
        if r["seed_item_id"]:
            d["seeds"].add(r["seed_item_id"])
    return info


def _titles(conn: sqlite3.Connection, item_ids) -> dict[str, str]:
    ids = [i for i in set(item_ids) if i]
    out: dict[str, str] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT id, title FROM items WHERE id IN ({marks})", chunk):
            out[r["id"]] = r["title"]
    return out


def _base_scores(
    x: np.ndarray, xn: np.ndarray, head: dict | None, centroid: dict | None,
) -> np.ndarray | None:
    dim = x.shape[1]
    if head is not None and head.get("dim") == dim:
        w = _vec(head["w"])
        z = np.clip(x @ w + float(head["b"]), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
    if centroid is not None and centroid.get("dim") == dim and centroid.get("pos"):
        s = xn @ _unit(_vec(centroid["pos"]))
        if centroid.get("neg"):
            s = s - NEG_PULL * (xn @ _unit(_vec(centroid["neg"])))
        return (s + 1.0) / 2.0
    return None


def _rescore_open(
    conn: sqlite3.Connection,
    domain: str,
    model_name: str,
    head: dict | None,
    centroid: dict | None,
) -> int:
    """Put still-open proposals on the current scorer's scale.

    Frozen scores from earlier runs are not comparable to fresh ones
    (head sigmoid vs centroid (s+1)/2, plus drift across retrains), yet
    apply ranks the whole open queue numerically for the weekly budget
    and overflow expiry. Base score only: the source/ext bonus is
    normalised against a single run's pool, so it has no comparable
    value here. Items without embeddings keep their old score."""
    rows = conn.execute(
        "SELECT id, item_id FROM recommendations WHERE domain=? AND status='proposed'",
        (domain,),
    ).fetchall()
    if not rows:
        return 0
    vecs = {
        item_id: np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        for item_id, _dim, blob in db.iter_embeddings(
            conn, model_name, [r["item_id"] for r in rows])
    }
    scorable = [r for r in rows if r["item_id"] in vecs]
    if not scorable:
        return 0
    x = np.stack([vecs[r["item_id"]] for r in scorable])
    base = _base_scores(x, _unit_rows(x), head, centroid)
    if base is None:
        return 0
    conn.executemany(
        "UPDATE recommendations SET score=? WHERE id=?",
        [(float(s), r["id"]) for r, s in zip(scorable, base)])
    return len(scorable)


def _select(
    scores: np.ndarray, xn: np.ndarray, slots: int, lam: float, explore_frac: float,
) -> tuple[list[int], set[int]]:
    n_mmr = min(slots, round(slots * (1.0 - explore_frac)))
    picked: list[int] = []
    remaining = list(range(len(scores)))
    while len(picked) < n_mmr and remaining:
        best, best_val = remaining[0], -np.inf
        for j in remaining:
            max_cos = max((float(xn[j] @ xn[k]) for k in picked), default=0.0)
            val = float(scores[j]) - lam * max_cos
            if val > best_val:
                best, best_val = j, val
        picked.append(best)
        remaining.remove(best)

    explored: set[int] = set()
    n_explore = slots - len(picked)
    if n_explore > 0 and remaining:
        lo, hi = np.percentile(scores[remaining], EXPLORE_BAND)
        band = [j for j in remaining if lo <= scores[j] <= hi] or list(remaining)
        for _ in range(n_explore):
            if not band:
                break
            best, best_val = band[0], -np.inf
            for j in band:
                min_dist = min((1.0 - float(xn[j] @ xn[k]) for k in picked), default=1.0)
                if min_dist > best_val:
                    best, best_val = j, min_dist
            band.remove(best)
            picked.append(best)
            explored.add(best)
    return picked, explored


def run(conn: sqlite3.Connection, cfg: Config, top: int = 20) -> dict:
    model_name = cfg.model.embed_model
    lam = cfg.model.diversity_lambda
    explore_frac = cfg.model.exploration_frac
    now_dt = datetime.now(timezone.utc)
    run_id = now_dt.strftime("%Y%m%d%H%M%S")
    ts = db.now()
    stats: dict = {
        "proposed": 0, "expired": 0, "unembedded": 0, "rescored": 0, "stale_state": 0,
    }

    domains = [r["domain"] for r in conn.execute(
        "SELECT DISTINCT i.domain FROM candidates c JOIN items i ON i.id = c.item_id"
        " ORDER BY i.domain")]
    for domain in domains:
        head = _state_json(conn, f"model:{domain}")
        centroid = _state_json(conn, f"centroid:{domain}")
        # State trained in another embedding space scores garbage even when
        # the dims coincide — treat it as absent until train refits.
        if head is not None and head.get("embed_model") != model_name:
            head = None
            stats["stale_state"] += 1
        if centroid is not None and centroid.get("embed_model") != model_name:
            centroid = None
            stats["stale_state"] += 1
        if head is None and centroid is None:
            continue
        stats["rescored"] += _rescore_open(conn, domain, model_name, head, centroid)
        info = _pool(conn, domain)
        if not info:
            continue
        vecs = {
            item_id: np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            for item_id, _dim, blob in db.iter_embeddings(conn, model_name, list(info))
        }
        stats["unembedded"] += len(info) - len(vecs)
        if not vecs:
            continue
        ids = list(vecs)
        x = np.stack([vecs[i] for i in ids])
        xn = _unit_rows(x)

        base = _base_scores(x, xn, head, centroid)
        if base is None:
            continue
        max_ext = max(d["ext"] for d in info.values())
        bonus = np.array([
            SOURCE_BONUS * min(len(info[i]["sources"]), 3)
            + (EXT_BONUS * info[i]["ext"] / max_ext if max_ext > 0 else 0.0)
            for i in ids
        ])
        scores = base + bonus

        slots = min(top, ARTIST_TOP) if domain == "artist" else top
        slots = min(slots, len(ids))
        picked, explored = _select(scores, xn, slots, lam, explore_frac)

        exemplars = (centroid or {}).get("exemplars") or []
        ex_vecs = {
            item_id: _unit(np.frombuffer(blob, dtype=np.float16).astype(np.float32))
            for item_id, _dim, blob in db.iter_embeddings(
                conn, model_name, [e[0] for e in exemplars])
        }
        seed_ids = {s for j in picked for s in info[ids[j]]["seeds"]}
        titles = _titles(conn, [*ex_vecs, *seed_ids])

        for j in picked:
            item_id = ids[j]
            sims = sorted(
                ((e, float(xn[j] @ ev)) for e, ev in ex_vecs.items()),
                key=lambda t: -t[1])[:3]
            why = {
                "neighbors": [
                    {"item_id": e, "title": titles.get(e), "sim": round(s, 4)}
                    for e, s in sims
                ],
                "sources": sorted(info[item_id]["sources"]),
                "seeds": sorted(titles.get(s) or s for s in info[item_id]["seeds"]),
                "exploration": j in explored,
            }
            # OR IGNORE: the partial unique index on open recommendations
            # is the last line of defence against double-queueing.
            cur = conn.execute(
                "INSERT OR IGNORE INTO recommendations"
                " (run_id, ts, domain, item_id, score, why, status)"
                " VALUES (?,?,?,?,?,?,'proposed')",
                (run_id, ts, domain, item_id, float(scores[j]), json.dumps(why)))
            if cur.rowcount:
                stats["proposed"] += 1
                stats[domain] = stats.get(domain, 0) + 1

    cutoff = (now_dt - timedelta(days=cfg.autonomy.proposal_ttl_days)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        "UPDATE recommendations SET status='expired', acted_at=?"
        " WHERE status='proposed' AND ts < ?", (ts, cutoff))
    stats["expired"] = cur.rowcount
    return stats
