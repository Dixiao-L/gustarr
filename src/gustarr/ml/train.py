"""Per-domain preference heads + taste centroids.

The head is a tiny logistic regression trained with plain numpy so the
nightly pipeline never needs torch at train time; the embeddings carry
all the representational weight.
"""

from __future__ import annotations

import base64
import json
import sqlite3

import numpy as np

from .. import db, signals
from ..config import Config

DOMAINS = ("movie", "series", "artist")
POS_THRESHOLD = 0.15
NEG_THRESHOLD = -0.05
# Below this a linear head just memorises the handful of positives;
# centroid similarity generalises better until more signal arrives.
MIN_POSITIVES = 8
WEAK_WEIGHT = 0.3
L2_LAMBDA = 1e-2
LR = 0.01
STEPS = 300
PLATEAU = 1e-5


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def _item_labels(conn: sqlite3.Connection, domain: str) -> dict[str, float]:
    per_item: dict[str, list[tuple[str, str, float]]] = {}
    for r in conn.execute(
        "SELECT e.item_id, e.ts, e.kind, e.weight FROM events e"
        " JOIN items i ON i.id = e.item_id WHERE i.domain = ?",
        (domain,),
    ):
        per_item.setdefault(r["item_id"], []).append((r["ts"], r["kind"], r["weight"]))
    return {item: signals.aggregate_label(evts) for item, evts in per_item.items()}


def _vectors(conn: sqlite3.Connection, model: str, item_ids: list[str]) -> dict[str, np.ndarray]:
    return {
        item_id: np.frombuffer(vec, dtype=np.float16).astype(np.float32)
        for item_id, _dim, vec in db.iter_embeddings(conn, model, item_ids)
    }


def _weak_negatives(
    conn: sqlite3.Connection, domain: str, model: str, n: int, rng: np.random.Generator,
) -> list[str]:
    """Unlabelled candidates as soft negatives: most of the pool is stuff
    the user never chose, which is weak evidence against."""
    rows = conn.execute(
        "SELECT DISTINCT c.item_id FROM candidates c"
        " JOIN items i ON i.id = c.item_id"
        " JOIN embeddings emb ON emb.item_id = c.item_id AND emb.model = ?"
        " WHERE i.domain = ?"
        "   AND NOT EXISTS (SELECT 1 FROM events e WHERE e.item_id = c.item_id)",
        (model, domain),
    ).fetchall()
    ids = [r["item_id"] for r in rows]
    if len(ids) <= n:
        return ids
    return [ids[i] for i in rng.choice(len(ids), size=n, replace=False)]


def _fit_head(x: np.ndarray, y: np.ndarray, sw: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted L2-regularised logistic regression, full-batch Adam."""
    dim = x.shape[1]
    theta = np.zeros(dim + 1, dtype=np.float64)  # [w..., b]
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    swn = sw / sw.sum()
    prev = np.inf
    for t in range(1, STEPS + 1):
        w, b = theta[:dim], theta[dim]
        z = np.clip(x @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(
            -(swn * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum()
            + L2_LAMBDA * (w @ w)
        )
        if abs(prev - loss) < PLATEAU:
            break
        prev = loss
        err = swn * (p - y)
        grad = np.concatenate([x.T @ err + 2 * L2_LAMBDA * w, [err.sum()]])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        theta -= LR * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    return theta[:dim].astype(np.float32), float(theta[dim])


def run(conn: sqlite3.Connection, cfg: Config) -> dict:
    model_name = cfg.model.embed_model
    rng = np.random.default_rng(0)
    stats: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        labels = _item_labels(conn, domain)
        pos = {i: v for i, v in labels.items() if v >= POS_THRESHOLD}
        neg = {i: v for i, v in labels.items() if v <= NEG_THRESHOLD}
        vecs = _vectors(conn, model_name, [*pos, *neg])
        pos_ids = [i for i in pos if i in vecs]
        neg_ids = [i for i in neg if i in vecs]
        dstats = {"pos": len(pos_ids), "neg": len(neg_ids), "weak": 0, "head": 0}
        stats[domain] = dstats
        if not pos_ids:
            continue

        dim = int(vecs[pos_ids[0]].shape[0])
        pos_mean = np.stack([vecs[i] for i in pos_ids]).mean(axis=0)
        neg_mean = np.stack([vecs[i] for i in neg_ids]).mean(axis=0) if neg_ids else None
        exemplars = sorted(((i, pos[i]) for i in pos_ids), key=lambda t: -t[1])[:50]
        db.set_state(conn, f"centroid:{domain}", json.dumps({
            "pos": _b64(pos_mean),
            "neg": _b64(neg_mean) if neg_mean is not None else None,
            "dim": dim,
            "embed_model": model_name,
            "exemplars": exemplars,
        }))

        if len(pos_ids) < MIN_POSITIVES:
            continue
        weak_ids = _weak_negatives(conn, domain, model_name, max(20, 2 * len(pos_ids)), rng)
        weak_vecs = _vectors(conn, model_name, weak_ids)
        if not neg_ids and not weak_vecs:
            continue  # nothing to push against; the centroid alone must do

        rows_x = (
            [vecs[i] for i in pos_ids]
            + [vecs[i] for i in neg_ids]
            + [weak_vecs[i] for i in weak_ids]
        )
        y = np.array([1.0] * len(pos_ids) + [0.0] * (len(rows_x) - len(pos_ids)))
        sw = np.array(
            [1.0] * (len(pos_ids) + len(neg_ids))
            + [WEAK_WEIGHT] * (len(rows_x) - len(pos_ids) - len(neg_ids))
        )
        w, b = _fit_head(np.stack(rows_x), y, sw)
        db.set_state(conn, f"model:{domain}", json.dumps({
            "w": _b64(w),
            "b": b,
            "dim": dim,
            "embed_model": model_name,
            "trained_at": db.now(),
            "n_pos": len(pos_ids),
            "n_neg": len(neg_ids),
            "n_weak": len(weak_ids),
        }))
        dstats["weak"] = len(weak_ids)
        dstats["head"] = 1
    return stats
