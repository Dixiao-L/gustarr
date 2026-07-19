"""Score each profile's candidate pool and queue proposals for them.

MMR keeps a run's picks from clustering on one taste mode; a slice of
each run is reserved for exploration so ranking never goes fully
exploit-only — the model needs off-policy feedback to keep learning.
Exploration slots prefer serendipity_* candidates (sampled from
under-represented regions, so their low scores are expected) and are
novelty-gated against the positive centroid so the "exploration" label
never lands on a near-core item.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import db, queue, settings
from ..candidates import snooze_cutoff
from ..config import Config

# Music autonomy caps are low; a 20-deep artist queue just goes stale.
ARTIST_TOP = 10
# Albums actuate under their own weekly budget (music_max_albums_per_week),
# so their queue depth mirrors the artist one. Like artists they are
# exempt from the video queue cap below.
ALBUM_TOP = 10
DOMAIN_TOP = {"artist": ARTIST_TOP, "album": ALBUM_TOP}
SOURCE_BONUS = 0.03
EXT_BONUS = 0.05
NEG_PULL = 0.6
EXPLORE_BAND = (40.0, 90.0)  # score percentiles: novel-ish but not junk
SERENDIPITY_SOURCES = frozenset({"serendipity_tmdb", "serendipity_lastfm"})
# Serendipity skips the EXPLORE_BAND quality gate (its scores are
# legitimately low — that's the point) but not this sanity floor.
SEREN_FLOOR_PCT = 10.0
# Exploration picks must sit below this percentile of the pool's
# centroid similarities: outside the taste core, not near-duplicates.
NOVELTY_PCT = 60.0


def _state_json(conn: sqlite3.Connection, profile: str, key: str) -> dict | None:
    # train writes heads/centroids under the profile namespace now.
    raw = db.pget_state(conn, profile, key)
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


def _pool(conn: sqlite3.Connection, profile: str, domain: str) -> dict[int, dict]:
    """Candidates still worth proposing to this profile: not owned, not
    already queued for them, not failed for them, not actively snoozed
    by them, never explicitly rejected by them. The library block stays
    global (one disk); the verdict blocks are personal — another
    profile's reject or open rec must not veto this one's proposal."""
    # The partial unique index only guards (profile,item_id) over
    # proposed/approved, so the active-snooze block must live in this
    # query, not in the schema. 'failed' is terminal for automation
    # (docs: "rank never re-proposes it" — re-proposing would just
    # re-fail); only a human retry() revives it.
    rows = conn.execute(
        "SELECT c.item_id, c.source, c.external_score, c.seed_item_id"
        " FROM candidates c JOIN items i ON i.id = c.item_id"
        " WHERE i.domain = ? AND c.profile = ?"
        "   AND c.item_id NOT IN (SELECT item_id FROM library)"
        "   AND c.item_id NOT IN (SELECT item_id FROM recommendations"
        "                         WHERE profile = ?"
        "                           AND (status IN ('proposed','approved','failed')"
        "                                OR (status='snoozed' AND acted_at >= ?)))"
        "   AND c.item_id NOT IN (SELECT item_id FROM events"
        "                         WHERE profile = ? AND kind='reject')",
        (domain, profile, profile, snooze_cutoff(), profile),
    )
    info: dict[int, dict] = {}
    for r in rows:
        d = info.setdefault(r["item_id"], {"sources": set(), "ext": 0.0, "seeds": set()})
        d["sources"].add(r["source"])
        if r["external_score"] is not None:
            d["ext"] = max(d["ext"], float(r["external_score"]))
        if r["seed_item_id"] is not None:
            d["seeds"].add(r["seed_item_id"])
    return info


def _titles(conn: sqlite3.Connection, item_ids) -> dict[int, str]:
    ids = [i for i in set(item_ids) if i is not None]
    out: dict[int, str] = {}
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
    profile: str,
    domain: str,
    model_name: str,
    head: dict | None,
    centroid: dict | None,
) -> int:
    """Put this profile's still-open proposals on its current scorer's
    scale.

    Frozen scores from earlier runs are not comparable to fresh ones
    (head sigmoid vs centroid (s+1)/2, plus drift across retrains), yet
    apply ranks the whole open queue numerically for the weekly budget
    and overflow expiry. Base score only: the source/ext bonus is
    normalised against a single run's pool, so it has no comparable
    value here. Items without embeddings keep their old score. Scoped
    per profile because each profile's scorer is a different function —
    rescoring alice's queue with bob's head would be nonsense."""
    rows = conn.execute(
        "SELECT id, item_id FROM recommendations"
        " WHERE profile=? AND domain=? AND status='proposed'",
        (profile, domain),
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


def _farthest(pool: list[int], picked: list[int], xn: np.ndarray) -> int:
    best, best_val = pool[0], -np.inf
    for j in pool:
        min_dist = min((1.0 - float(xn[j] @ xn[k]) for k in picked), default=1.0)
        if min_dist > best_val:
            best, best_val = j, min_dist
    return best


def _select(
    scores: np.ndarray,
    xn: np.ndarray,
    slots: int,
    lam: float,
    explore_frac: float,
    serendipity: set[int],
    cent_sims: np.ndarray | None,
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
        novel = set(remaining)
        if cent_sims is not None:
            cap = float(np.percentile(cent_sims, NOVELTY_PCT))
            novel = {j for j in remaining if float(cent_sims[j]) < cap}
        floor = float(np.percentile(scores, SEREN_FLOOR_PCT))
        seren = [j for j in remaining
                 if j in serendipity and j in novel and float(scores[j]) >= floor]
        lo, hi = np.percentile(scores[remaining], EXPLORE_BAND)
        band = [j for j in remaining if lo <= scores[j] <= hi] or list(remaining)
        band_novel = [j for j in band if j in novel]
        for _ in range(n_explore):
            # Preference: serendipity, then the novelty-gated band, then the
            # ungated band (the old behaviour) so gating never starves slots.
            pool = seren or band_novel or band
            if not pool:
                break
            best = _farthest(pool, picked, xn)
            for p in (seren, band_novel, band):
                if best in p:
                    p.remove(best)
            picked.append(best)
            explored.add(best)
    return picked, explored


def run(conn: sqlite3.Connection, cfg: Config, top: int = 20) -> dict:
    now_dt = datetime.now(timezone.utc)
    run_id = now_dt.strftime("%Y%m%d%H%M%S")
    ts = db.now()
    stats: dict = {
        "proposed": 0, "expired": 0, "unsnoozed": 0, "unembedded": 0, "rescored": 0,
        "stale_state": 0,
    }
    ttl_cutoff = (now_dt - timedelta(days=cfg.autonomy.proposal_ttl_days)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    for profile in cfg.profiles:
        _rank_profile(conn, cfg, profile, run_id, ts, top, stats)
        # Expiry/unsnooze run inside the profile loop so a profile removed
        # from config keeps its rows frozen instead of silently expiring.
        stats["expired"] += queue.transition_where(
            conn, "expired", "proposed", "profile=? AND ts < ?",
            (profile, ttl_cutoff), ts=ts)
        # A snooze is a timer, not a verdict: once it lapses the rec becomes
        # 'expired', which candidates/rank treat as re-proposable.
        stats["unsnoozed"] += queue.transition_where(
            conn, "expired", "snoozed", "profile=? AND acted_at < ?",
            (profile, snooze_cutoff()), ts=ts)
    return stats


def _rank_profile(
    conn: sqlite3.Connection, cfg: Config, profile: str,
    run_id: str, ts: str, top: int, stats: dict,
) -> None:
    model_name = cfg.model.embed_model
    lam = cfg.model.diversity_lambda
    explore_frac = settings.get(conn, cfg, "exploration_frac")

    domains = [r["domain"] for r in conn.execute(
        "SELECT DISTINCT i.domain FROM candidates c JOIN items i ON i.id = c.item_id"
        " WHERE c.profile = ? ORDER BY i.domain", (profile,))]
    for domain in domains:
        head = _state_json(conn, profile, f"model:{domain}")
        centroid = _state_json(conn, profile, f"centroid:{domain}")
        # State trained in another embedding space scores garbage even when
        # the dims coincide — treat it as absent until train refits.
        if head is not None and head.get("embed_model") != model_name:
            head = None
            stats["stale_state"] += 1
        if centroid is not None and centroid.get("embed_model") != model_name:
            centroid = None
            stats["stale_state"] += 1
        if head is None and centroid is None and domain == "album":
            # Nothing listens "to an album" in the event stream yet, so the
            # album domain has no labels of its own — score against the
            # profile's artist taste model (same embedding space, same
            # musical taste) until album-level feedback accumulates.
            head = _state_json(conn, profile, "model:artist")
            centroid = _state_json(conn, profile, "centroid:artist")
            if head is not None and head.get("embed_model") != model_name:
                head = None
            if centroid is not None and centroid.get("embed_model") != model_name:
                centroid = None
        if head is None and centroid is None:
            continue
        stats["rescored"] += _rescore_open(conn, profile, domain, model_name, head, centroid)
        info = _pool(conn, profile, domain)
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

        seren_idx = {j for j, i in enumerate(ids)
                     if info[i]["sources"] & SERENDIPITY_SOURCES}
        cent_sims = None
        if centroid is not None and centroid.get("dim") == x.shape[1] and centroid.get("pos"):
            cent_sims = xn @ _unit(_vec(centroid["pos"]))

        slots = min(top, DOMAIN_TOP.get(domain, top))
        if domain in ("movie", "series"):
            # Respect the video queue cap here rather than letting apply
            # mass-expire the overflow later — proposing 80 and trimming
            # to the cap churned 180 enriched items in one night. The cap
            # bounds each profile's own approval queue, so it counts only
            # this profile's open video proposals.
            open_video = conn.execute(
                "SELECT COUNT(*) FROM recommendations"
                " WHERE profile=? AND status='proposed' AND domain IN ('movie','series')",
                (profile,)).fetchone()[0]
            cap = settings.get(conn, cfg, "video_queue_max_pending")
            slots = max(0, min(slots, cap - open_video))
        slots = min(slots, len(ids))
        if slots == 0:
            continue
        picked, explored = _select(scores, xn, slots, lam, explore_frac, seren_idx, cent_sims)

        exemplars = (centroid or {}).get("exemplars") or []
        ex_vecs = {
            item_id: _unit(np.frombuffer(blob, dtype=np.float16).astype(np.float32))
            for item_id, _dim, blob in db.iter_embeddings(
                conn, model_name, [e[0] for e in exemplars])
        }
        titles = _titles(conn, list(ex_vecs))

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
                # int item ids: queue resolves display titles at read time,
                # so a later merge or retitle never bakes a stale name in
                "seeds": sorted(info[item_id]["seeds"]),
                "exploration": j in explored,
            }
            if j in explored and info[item_id]["sources"] & SERENDIPITY_SOURCES:
                why["serendipity"] = True
            # OR IGNORE: the partial unique index on (profile, item_id)
            # over open recommendations is the last line of defence
            # against double-queueing — per profile, so two profiles may
            # each hold their own open rec for the same item.
            cur = conn.execute(
                "INSERT OR IGNORE INTO recommendations"
                " (profile, run_id, ts, domain, item_id, score, why, status)"
                " VALUES (?,?,?,?,?,?,?,'proposed')",
                (profile, run_id, ts, domain, item_id, float(scores[j]), json.dumps(why)))
            if cur.rowcount:
                stats["proposed"] += 1
                stats[domain] = stats.get(domain, 0) + 1
