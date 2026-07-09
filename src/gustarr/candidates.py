"""Candidate pool refresh: fan out from liked items to unseen neighbours.

Seeds are the top positively-labelled items per domain (same label
policy as training — signals.aggregate_label). Movies/series fan out
through TMDb recommendations/similar plus a genre-keyed discover pass
for exploration; artists through Last.fm similar. A serendipity pass
counters the bubble those sources create: TMDb discover over the user's
under-represented genres and least-visited decade (quality-floored),
and a damped 2-hop Last.fm fan-out from the best hop-1 candidates.
Exclusions (library, open/acted recommendations, rejected items) are
applied at insert time — nothing is deleted from the pool, excluded
rows simply never enter.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any

from . import db, http, ids, signals
from .config import Config

TMDB = "https://api.themoviedb.org/3"
LASTFM = "https://ws.audioscrobbler.com/2.0/"

SEED_LABEL_MIN = 0.3
SEED_LIMIT = 25
TOP_GENRES = 5
# rank only consumes a few hundred candidates; unbounded fan-out would
# just balloon the enrich/embed backlog without improving picks.
MAX_NEW_PER_SOURCE = 200

# Serendipity: deliberate off-taste probes. Tighter cap than the taste
# sources — seasoning, not the meal.
SERENDIPITY_MAX_NEW = 100
SOURCE_CAPS = {"serendipity_tmdb": SERENDIPITY_MAX_NEW, "serendipity_lastfm": SERENDIPITY_MAX_NEW}
SERENDIPITY_GENRE_PROBES = 4
# Quality floor: serendipity should be great-but-unfamiliar, not random junk.
SERENDIPITY_VOTE_FLOOR = 500
SERENDIPITY_HOP_SEEDS = 5
SERENDIPITY_HOP_LIMIT = 20
SERENDIPITY_DAMPING = 0.8  # 2 hops out: trust transitive similarity less
# Pre-1960 discover mostly surfaces archive noise; post-2019 the regular
# discover pass already skews recent.
SERENDIPITY_DECADES = range(1960, 2020, 10)

# Anything already decided on (queued, added by apply, or terminally
# failed to actuate) or ever rejected must never re-enter the pool —
# 'failed' is terminal, so re-proposing it would just re-fail in apply.
BLOCKED_REC_STATUSES = ("proposed", "approved", "auto_added", "added", "failed")


def run(conn: sqlite3.Connection, cfg: Config, domain: str | None = None) -> dict[str, Any]:
    domains = [domain] if domain else ["movie", "series", "artist"]
    stats: dict[str, Any] = {
        "seeds": {}, "new": {}, "updated": {}, "skipped": 0, "capped": [], "errors": 0,
        "serendipity": 0,
    }
    pool = _Pool(conn, stats)
    tmdb_key = cfg.tmdb.get("api_key")
    lastfm_key = cfg.lastfm.get("api_key")
    for dom in domains:
        if dom in ("movie", "series") and tmdb_key:
            positives = _positive_items(conn, dom)
            seeds = [(iid, tid) for iid, tid in
                     ((iid, _tmdb_id(conn, iid)) for iid, _ in positives[:SEED_LIMIT]) if tid]
            stats["seeds"][dom] = len(seeds)
            _tmdb_similar(tmdb_key, dom, seeds, pool, stats)
            _tmdb_discover(conn, tmdb_key, dom, positives, pool, stats)
            _tmdb_serendipity(conn, tmdb_key, dom, positives, pool, stats)
        elif dom == "artist" and lastfm_key:
            positives = _positive_items(conn, "artist")
            _lastfm_similar(conn, lastfm_key, [iid for iid, _ in positives[:SEED_LIMIT]],
                            pool, stats)
            _lastfm_serendipity(conn, lastfm_key, positives, pool, stats)
    stats["serendipity"] = sum(
        n for src, n in stats["new"].items() if src.startswith("serendipity_"))
    return stats


# ── pool gate ────────────────────────────────────────────────────────


class _Pool:
    """Insert-time gate: exclusion sets, per-source new-row cap, upsert."""

    def __init__(self, conn: sqlite3.Connection, stats: dict[str, Any]):
        self.conn = conn
        self.stats = stats
        self.excluded = _excluded_ids(conn)
        self.new_rows: Counter[str] = Counter()

    def add(self, item_id: str, domain: str, title: str | None, year: int | None,
            ids_d: dict[str, Any], meta: dict[str, Any], source: str,
            seed_id: str | None, score: float | None) -> None:
        # A merged-away fallback id must land on its canonical row, or the
        # candidates FK insert fails (upsert_item redirects the items write
        # but this raw insert would still use the stale id).
        item_id = db.canonical_id(self.conn, item_id)
        if item_id in self.excluded:
            self.stats["skipped"] += 1
            return
        exists = self.conn.execute(
            "SELECT 1 FROM candidates WHERE item_id=? AND source=?", (item_id, source)
        ).fetchone() is not None
        if not exists and self.new_rows[source] >= SOURCE_CAPS.get(source, MAX_NEW_PER_SOURCE):
            if source not in self.stats["capped"]:
                self.stats["capped"].append(source)
            return
        db.upsert_item(self.conn, item_id, domain, title=title, year=year, ids=ids_d, meta=meta)
        ts = db.now()
        self.conn.execute(
            "INSERT INTO candidates (item_id, source, seed_item_id, external_score,"
            " first_seen, last_seen) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(item_id, source) DO UPDATE SET last_seen=excluded.last_seen,"
            " external_score=CASE"
            "   WHEN candidates.external_score IS NULL THEN excluded.external_score"
            "   WHEN excluded.external_score IS NULL THEN candidates.external_score"
            "   ELSE max(candidates.external_score, excluded.external_score) END",
            (item_id, source, seed_id, score, ts, ts),
        )
        bucket = "updated" if exists else "new"
        self.stats[bucket][source] = self.stats[bucket].get(source, 0) + 1
        if not exists:
            self.new_rows[source] += 1


def _excluded_ids(conn: sqlite3.Connection) -> set[str]:
    marks = ",".join("?" * len(BLOCKED_REC_STATUSES))
    q = (
        "SELECT x.item_id, i.domain, i.ids FROM ("
        "SELECT item_id FROM library"
        f" UNION SELECT item_id FROM recommendations WHERE status IN ({marks})"
        " UNION SELECT item_id FROM events WHERE kind='reject') x"
        " LEFT JOIN items i ON i.id = x.item_id"
    )
    excluded: set[str] = set()
    for r in conn.execute(q, BLOCKED_REC_STATUSES):
        excluded.add(r["item_id"])
        if r["domain"] is None:
            continue
        # An excluded item is often known under several namespaces (a Sonarr
        # library row is series:tvdb:<id> while TMDb mints series:tmdb:<id>),
        # so block every alternate id derivable from its ids JSON as well.
        for ns, key in json.loads(r["ids"]).items():
            if key:
                excluded.add(ids.make(r["domain"], ns, str(key)))
    return excluded


# ── seed selection ───────────────────────────────────────────────────


def _positive_items(conn: sqlite3.Connection, domain: str) -> list[tuple[str, float]]:
    """Items with aggregate label >= SEED_LABEL_MIN, best first
    (ties broken by most recent event)."""
    rows = conn.execute(
        "SELECT e.item_id, e.ts, e.kind, e.weight FROM events e"
        " JOIN items i ON i.id = e.item_id WHERE i.domain=?",
        (domain,),
    ).fetchall()
    events: dict[str, list[tuple[str, str, float]]] = {}
    latest: dict[str, str] = {}
    for r in rows:
        events.setdefault(r["item_id"], []).append((r["ts"], r["kind"], r["weight"]))
        if r["ts"] > latest.get(r["item_id"], ""):
            latest[r["item_id"]] = r["ts"]
    positives = [(iid, label) for iid, evs in events.items()
                 if (label := signals.aggregate_label(evs)) >= SEED_LABEL_MIN]
    positives.sort(key=lambda t: (t[1], latest[t[0]]), reverse=True)
    return positives


def _tmdb_id(conn: sqlite3.Connection, item_id: str) -> str | None:
    _, ns, key = ids.parse(item_id)
    if ns == "tmdb":
        return key
    row = conn.execute("SELECT ids FROM items WHERE id=?", (item_id,)).fetchone()
    if row is None:
        return None
    tid = json.loads(row["ids"]).get("tmdb")
    return str(tid) if tid else None


# ── TMDb ─────────────────────────────────────────────────────────────


def _tmdb_result(domain: str, r: dict[str, Any]) -> tuple | None:
    tid = r.get("id")
    if not tid:
        return None
    title = r.get("title") if domain == "movie" else r.get("name")
    date = (r.get("release_date") if domain == "movie" else r.get("first_air_date")) or ""
    year = int(date[:4]) if date[:4].isdigit() else None
    meta: dict[str, Any] = {}
    if r.get("popularity") is not None:
        meta["popularity"] = r["popularity"]
    if r.get("overview"):
        meta["overview"] = r["overview"]
    item_id = ids.make(domain, "tmdb", str(tid))
    return item_id, title, year, {"tmdb": tid}, meta, r.get("vote_average")


def _tmdb_similar(api_key: str, domain: str, seeds: list[tuple[str, str]],
                  pool: _Pool, stats: dict[str, Any]) -> None:
    media = "movie" if domain == "movie" else "tv"
    # TMDb has no /tv/{id}/similar worth using; recommendations covers it.
    endpoints = ("recommendations", "similar") if domain == "movie" else ("recommendations",)
    for seed_id, tmdb_id in seeds:
        for endpoint in endpoints:
            try:
                data = http.get_json(f"{TMDB}/{media}/{tmdb_id}/{endpoint}",
                                     params={"api_key": api_key, "page": 1})
            except http.ApiError:
                stats["errors"] += 1
                continue
            for r in (data or {}).get("results") or []:
                parsed = _tmdb_result(domain, r)
                if parsed:
                    item_id, title, year, ids_d, meta, score = parsed
                    pool.add(item_id, domain, title, year, ids_d, meta,
                             "tmdb_similar", seed_id, score)


def _genre_map(conn: sqlite3.Connection, api_key: str, media: str) -> dict[str, int]:
    state_key = f"tmdb:genres:{media}"
    cached = db.get_state(conn, state_key)
    if cached:
        return json.loads(cached)
    data = http.get_json(f"{TMDB}/genre/{media}/list", params={"api_key": api_key})
    mapping = {str(g["name"]).strip().lower(): g["id"]
               for g in (data or {}).get("genres") or [] if g.get("name")}
    db.set_state(conn, state_key, json.dumps(mapping))
    return mapping


def _genre_counts(conn: sqlite3.Connection, positives: list[tuple[str, float]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for iid, _ in positives:
        row = conn.execute("SELECT meta FROM items WHERE id=?", (iid,)).fetchone()
        if row is None:
            continue
        for g in json.loads(row["meta"]).get("genres") or []:
            name = g.get("name") if isinstance(g, dict) else g
            if name:
                counts[str(name).strip().lower()] += 1
    return counts


def _top_genres(conn: sqlite3.Connection, positives: list[tuple[str, float]]) -> list[str]:
    return [name for name, _ in _genre_counts(conn, positives).most_common(TOP_GENRES)]


def _tmdb_discover(conn: sqlite3.Connection, api_key: str, domain: str,
                   positives: list[tuple[str, float]], pool: _Pool,
                   stats: dict[str, Any]) -> None:
    genre_names = _top_genres(conn, positives)
    if not genre_names:
        return
    media = "movie" if domain == "movie" else "tv"
    try:
        gmap = _genre_map(conn, api_key, media)
    except http.ApiError:
        stats["errors"] += 1
        return
    genre_ids = [str(gmap[n]) for n in genre_names if n in gmap]
    if not genre_ids:
        return
    try:
        # pipe = OR: exploration pool should span the taste genres, not
        # demand their intersection.
        data = http.get_json(f"{TMDB}/discover/{media}", params={
            "api_key": api_key, "with_genres": "|".join(genre_ids),
            "sort_by": "popularity.desc", "vote_count.gte": 100, "page": 1,
        })
    except http.ApiError:
        stats["errors"] += 1
        return
    for r in (data or {}).get("results") or []:
        parsed = _tmdb_result(domain, r)
        if parsed:
            item_id, title, year, ids_d, meta, score = parsed
            pool.add(item_id, domain, title, year, ids_d, meta, "tmdb_discover", None, score)


# ── serendipity: TMDb ────────────────────────────────────────────────


def _under_represented_genres(gmap: dict[str, int], counts: Counter[str]) -> list[str]:
    """Genres whose share of the positive mentions falls below the
    uniform share — the user's blind spots, absent genres included.
    Rarest first, tie by name, so probe order is deterministic."""
    if not gmap:
        return []
    total = sum(counts.values())
    uniform = 1.0 / len(gmap)
    under = sorted(
        (counts.get(name, 0), name) for name in gmap
        if (counts.get(name, 0) / total if total else 0.0) < uniform)
    return [name for _, name in under[:SERENDIPITY_GENRE_PROBES]]


def _least_seen_decade(conn: sqlite3.Connection, positives: list[tuple[str, float]]) -> int:
    counts: Counter[int] = Counter()
    for iid, _ in positives:
        row = conn.execute("SELECT year FROM items WHERE id=?", (iid,)).fetchone()
        year = row["year"] if row else None
        if year and SERENDIPITY_DECADES[0] <= year < SERENDIPITY_DECADES[-1] + 10:
            counts[year - year % 10] += 1
    # earliest decade wins ties so the probe is deterministic
    return min(SERENDIPITY_DECADES, key=lambda d: (counts[d], d))


def _tmdb_serendipity(conn: sqlite3.Connection, api_key: str, domain: str,
                      positives: list[tuple[str, float]], pool: _Pool,
                      stats: dict[str, Any]) -> None:
    """Anti-bubble discover: top-rated titles from under-represented
    genres (one genre per call) plus the least-visited decade."""
    if not positives:
        return  # no taste baseline to diverge from
    media = "movie" if domain == "movie" else "tv"
    try:
        gmap = _genre_map(conn, api_key, media)
    except http.ApiError:
        stats["errors"] += 1
        gmap = {}
    probes: list[dict[str, Any]] = [
        {"with_genres": str(gmap[name])}
        for name in _under_represented_genres(gmap, _genre_counts(conn, positives))]
    decade = _least_seen_decade(conn, positives)
    date_field = "primary_release_date" if domain == "movie" else "first_air_date"
    probes.append({f"{date_field}.gte": f"{decade}-01-01",
                   f"{date_field}.lte": f"{decade + 9}-12-31"})
    for probe in probes:
        try:
            data = http.get_json(f"{TMDB}/discover/{media}", params={
                "api_key": api_key, "sort_by": "vote_average.desc",
                "vote_count.gte": SERENDIPITY_VOTE_FLOOR, "page": 1, **probe,
            })
        except http.ApiError:
            stats["errors"] += 1
            continue
        for r in (data or {}).get("results") or []:
            parsed = _tmdb_result(domain, r)
            if parsed:
                item_id, title, year, ids_d, meta, score = parsed
                pool.add(item_id, domain, title, year, ids_d, meta,
                         "serendipity_tmdb", None, score)


# ── Last.fm ──────────────────────────────────────────────────────────


def _artist_query_params(conn: sqlite3.Connection, item_id: str, api_key: str,
                         limit: int) -> dict[str, Any] | None:
    """artist.getsimilar params for an item, mbid-first; None when the
    item offers neither an mbid nor a usable name."""
    _, ns, key = ids.parse(item_id)
    row = conn.execute("SELECT title, ids FROM items WHERE id=?", (item_id,)).fetchone()
    stored = json.loads(row["ids"]) if row else {}
    mbid = key if ns == "mbid" else stored.get("mbid")
    params: dict[str, Any] = {"method": "artist.getsimilar", "api_key": api_key,
                              "format": "json", "limit": limit}
    if mbid:
        params["mbid"] = mbid
    else:
        name = row["title"] if row and row["title"] else (key if ns == "lastfm" else None)
        if not name:
            return None
        params["artist"] = name
    return params


def _similar_artists(data: Any) -> list[dict[str, Any]]:
    artists = ((data or {}).get("similarartists") or {}).get("artist") or []
    if isinstance(artists, dict):  # last.fm collapses single-element lists
        artists = [artists]
    return artists


def _resolve_artist(conn: sqlite3.Connection, pool: _Pool,
                    a: dict[str, Any]) -> tuple[str, str, str | None, float] | None:
    """(item_id, name, mbid, match) for one similar-artist entry."""
    name = a.get("name")
    if not name:
        return None
    mb = a.get("mbid") or None
    if mb:
        item_id = ids.make("artist", "mbid", mb)
        fallback = ids.make("artist", "lastfm", name)
        # Last.fm asserting name↔mbid identity beats the fuzzy
        # MusicBrainz search enrich relies on: fold any name-keyed
        # twin (scrobbles often omit the mbid) into the canonical
        # item so one artist never lives on as two ids.
        db.merge_item(conn, fallback, item_id)
        if fallback in pool.excluded:
            pool.excluded.add(item_id)
    else:
        item_id = ids.make("artist", "lastfm", name)
    try:
        score = float(a.get("match") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return item_id, name, mb, score


def _lastfm_similar(conn: sqlite3.Connection, api_key: str, seed_ids: list[str],
                    pool: _Pool, stats: dict[str, Any]) -> None:
    used = 0
    for seed_id in seed_ids:
        params = _artist_query_params(conn, seed_id, api_key, limit=50)
        if params is None:
            continue
        used += 1
        try:
            data = http.get_json(LASTFM, params=params)
        except http.ApiError:
            stats["errors"] += 1
            continue
        for a in _similar_artists(data):
            resolved = _resolve_artist(conn, pool, a)
            if resolved is None:
                continue
            item_id, name, mb, score = resolved
            pool.add(item_id, "artist", name, None, {"mbid": mb} if mb else {}, {},
                     "lastfm_similar", seed_id, score)
    stats["seeds"]["artist"] = used


# ── serendipity: Last.fm ─────────────────────────────────────────────


def _lastfm_serendipity(conn: sqlite3.Connection, api_key: str,
                        positives: list[tuple[str, float]], pool: _Pool,
                        stats: dict[str, Any]) -> None:
    """2-hop neighbourhood: getSimilar on the strongest hop-1 candidates
    (never library rows or seeds), scores damped for the extra hop."""
    seeds = {iid for iid, _ in positives}
    hops = [r["item_id"] for r in conn.execute(
        "SELECT item_id FROM candidates WHERE source='lastfm_similar'"
        " AND item_id NOT IN (SELECT item_id FROM library)"
        " ORDER BY external_score DESC, item_id")
        if r["item_id"] not in seeds][:SERENDIPITY_HOP_SEEDS]
    for hop_id in hops:
        params = _artist_query_params(conn, hop_id, api_key, limit=SERENDIPITY_HOP_LIMIT)
        if params is None:
            continue
        try:
            data = http.get_json(LASTFM, params=params)
        except http.ApiError:
            stats["errors"] += 1
            continue
        for a in _similar_artists(data):
            resolved = _resolve_artist(conn, pool, a)
            if resolved is None:
                continue
            item_id, name, mb, score = resolved
            item_id = db.canonical_id(conn, item_id)
            # anything already reachable at hop 1 (or via another source)
            # is bubble-adjacent, not serendipity — leave it be
            known = conn.execute(
                "SELECT 1 FROM candidates WHERE item_id=? AND source!='serendipity_lastfm'",
                (item_id,)).fetchone()
            if known is not None:
                continue
            pool.add(item_id, "artist", name, None, {"mbid": mb} if mb else {}, {},
                     "serendipity_lastfm", hop_id, score * SERENDIPITY_DAMPING)
