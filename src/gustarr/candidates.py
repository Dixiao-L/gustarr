"""Candidate pool refresh: fan out from liked items to unseen neighbours.

The whole pass runs once per configured profile: seeds, exclusions and
caps are that profile's own, and candidate rows carry the profile.
Seeds are the top positively-labelled items per domain (same label
policy as training — signals.aggregate_label). Movies/series fan out
through TMDb recommendations/similar plus a genre-keyed discover pass
for exploration; artists through Last.fm similar; albums through
Last.fm top-albums of the liked artists. A serendipity pass
counters the bubble those sources create: TMDb discover over the user's
under-represented genres and least-visited decade (quality-floored),
and a damped 2-hop Last.fm fan-out from the best hop-1 candidates.
Exclusions (library, open/acted recommendations, actively snoozed or
rejected items) are applied at insert time — nothing is deleted from
the pool, excluded rows simply never enter. New items mint through
db.resolve_item, so a spelling or external id seen anywhere before
lands on its existing row.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
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

# Albums fan out from the already-curated artist seed list, so a modest
# per-artist depth and run cap keep the pool from drowning in one
# discography-heavy artist.
ALBUMS_PER_ARTIST = 8
# Raw playcounts span orders of magnitude between a mega-act and a niche
# artist; rank within the artist keeps external_score comparable.
# 1.0 for the artist's #1 album stepping down to 0.3 for #8.
ALBUM_RANK_STEP = 0.1
ALBUM_RANK_FLOOR = 0.3
ALBUM_MAX_NEW = 100

SOURCE_CAPS = {
    "serendipity_tmdb": SERENDIPITY_MAX_NEW,
    "serendipity_lastfm": SERENDIPITY_MAX_NEW,
    "lastfm_top_albums": ALBUM_MAX_NEW,
}
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

# 'snoozed' blocks re-entry only while the snooze is active; rank's
# expiry pass flips snoozes older than this window back to 'expired'
# (re-proposable). rank._pool mirrors the same acted_at condition.
SNOOZE_DAYS = 30


def snooze_cutoff() -> str:
    """acted_at values at/after this mark a still-active snooze."""
    return (datetime.now(timezone.utc) - timedelta(days=SNOOZE_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def run(conn: sqlite3.Connection, cfg: Config, domain: str | None = None) -> dict[str, Any]:
    domains = [domain] if domain else ["movie", "series", "artist", "album"]
    stats: dict[str, Any] = {
        "seeds": {}, "new": {}, "updated": {}, "skipped": 0, "capped": [], "errors": 0,
        "serendipity": 0, "profiles": 0,
    }
    tmdb_key = cfg.tmdb.get("api_key")
    lastfm_key = cfg.lastfm.get("api_key")
    # One full pass per profile: seeds come from that profile's own labels,
    # exclusions from its own recs/rejects (library stays global), and the
    # per-source new-row caps reset with each pool. Stats stay flat totals
    # across profiles — the per-profile split isn't actionable in a log line.
    for profile in cfg.profiles or {"default": None}:
        stats["profiles"] += 1
        pool = _Pool(conn, profile, stats)
        for dom in domains:
            if dom in ("movie", "series") and tmdb_key:
                positives = _positive_items(conn, profile, dom)
                seeds = [(iid, tid) for iid, tid in
                         ((iid, _tmdb_id(conn, iid)) for iid, _ in positives[:SEED_LIMIT])
                         if tid]
                stats["seeds"][dom] = stats["seeds"].get(dom, 0) + len(seeds)
                _tmdb_similar(tmdb_key, dom, seeds, pool, stats)
                _tmdb_discover(conn, tmdb_key, dom, positives, pool, stats)
                _tmdb_serendipity(conn, tmdb_key, dom, positives, pool, stats)
            elif dom == "artist" and lastfm_key:
                positives = _positive_items(conn, profile, "artist")
                _lastfm_similar(conn, lastfm_key, [iid for iid, _ in positives[:SEED_LIMIT]],
                                pool, stats)
                _lastfm_serendipity(conn, lastfm_key, positives, pool, stats)
            elif dom == "album" and lastfm_key:
                # Album taste rides on artist taste: reuse the artist seed list
                # rather than inventing a separate album label policy.
                positives = _positive_items(conn, profile, "artist")
                _lastfm_top_albums(conn, lastfm_key,
                                   [iid for iid, _ in positives[:SEED_LIMIT]], pool, stats)
    stats["serendipity"] = sum(
        n for src, n in stats["new"].items() if src.startswith("serendipity_"))
    return stats


# ── pool gate ────────────────────────────────────────────────────────


class _Pool:
    """Insert-time gate for one profile: exclusion sets, per-source
    new-row cap, upsert onto the (profile, item, source) key."""

    def __init__(self, conn: sqlite3.Connection, profile: str, stats: dict[str, Any]):
        self.conn = conn
        self.profile = profile
        self.stats = stats
        self.excluded = _excluded_ids(conn, profile)
        self.new_rows: Counter[str] = Counter()

    def add(self, domain: str, ns: str, key: str, title: str | None, year: int | None,
            meta: dict[str, Any], source: str, seed_id: int | None,
            score: float | None) -> None:
        # Gate on the looked-up identity before minting: an excluded or
        # capped result must not create (or freshen) an item row, or the
        # enrich/embed backlog balloons with rows rank never sees.
        if not ids.normalize_key(str(key)):
            self.stats["skipped"] += 1  # a key folding to nothing can't be an item
            return
        item_id = db.lookup_item(self.conn, domain, ns, key)
        if item_id is not None and item_id in self.excluded:
            self.stats["skipped"] += 1
            return
        exists = item_id is not None and self.conn.execute(
            "SELECT 1 FROM candidates WHERE profile=? AND item_id=? AND source=?",
            (self.profile, item_id, source)
        ).fetchone() is not None
        if not exists and self.new_rows[source] >= SOURCE_CAPS.get(source, MAX_NEW_PER_SOURCE):
            if source not in self.stats["capped"]:
                self.stats["capped"].append(source)
            return
        item_id = db.resolve_item(self.conn, domain, ns, key,
                                  title=title, year=year, meta=meta)
        ts = db.now()
        self.conn.execute(
            "INSERT INTO candidates (profile, item_id, source, seed_item_id, external_score,"
            " first_seen, last_seen) VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(profile, item_id, source) DO UPDATE SET"
            " last_seen=excluded.last_seen,"
            " external_score=CASE"
            "   WHEN candidates.external_score IS NULL THEN excluded.external_score"
            "   WHEN excluded.external_score IS NULL THEN candidates.external_score"
            "   ELSE max(candidates.external_score, excluded.external_score) END",
            (self.profile, item_id, source, seed_id, score, ts, ts),
        )
        bucket = "updated" if exists else "new"
        self.stats[bucket][source] = self.stats[bucket].get(source, 0) + 1
        if not exists:
            self.new_rows[source] += 1


def _excluded_ids(conn: sqlite3.Connection, profile: str) -> set[int]:
    marks = ",".join("?" * len(BLOCKED_REC_STATUSES))
    # library is the household's one disk — global; recommendation verdicts
    # and rejects are one person's taste — scoped to the profile. Identity
    # resolution guarantees one item per entity, so a plain int set is the
    # whole block list — no alternate-namespace synthesis needed.
    q = (
        "SELECT item_id FROM library"
        f" UNION SELECT item_id FROM recommendations WHERE profile=? AND (status IN ({marks})"
        "   OR (status='snoozed' AND acted_at >= ?))"
        " UNION SELECT item_id FROM events WHERE profile=? AND kind='reject'"
    )
    return {r["item_id"] for r in conn.execute(
        q, (profile, *BLOCKED_REC_STATUSES, snooze_cutoff(), profile))}


# ── seed selection ───────────────────────────────────────────────────


def _positive_items(conn: sqlite3.Connection, profile: str,
                    domain: str) -> list[tuple[int, float]]:
    """One profile's items with aggregate label >= SEED_LABEL_MIN, best
    first (ties broken by most recent event)."""
    rows = conn.execute(
        "SELECT e.item_id, e.ts, e.kind, e.weight FROM events e"
        " JOIN items i ON i.id = e.item_id WHERE e.profile=? AND i.domain=?",
        (profile, domain),
    ).fetchall()
    events: dict[int, list[tuple[str, str, float]]] = {}
    latest: dict[int, str] = {}
    for r in rows:
        events.setdefault(r["item_id"], []).append((r["ts"], r["kind"], r["weight"]))
        if r["ts"] > latest.get(r["item_id"], ""):
            latest[r["item_id"]] = r["ts"]
    positives = [(iid, label) for iid, evs in events.items()
                 if (label := signals.aggregate_label(evs)) >= SEED_LABEL_MIN]
    positives.sort(key=lambda t: (t[1], latest[t[0]]), reverse=True)
    return positives


def _tmdb_id(conn: sqlite3.Connection, item_id: int) -> str | None:
    return db.identities_of(conn, item_id).get("tmdb")


# ── TMDb ─────────────────────────────────────────────────────────────


def _tmdb_result(domain: str, r: dict[str, Any]) -> tuple | None:
    """(tmdb key, title, year, meta, external score) for one result row."""
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
        meta["overview"] = r["overview"][:300]
    if r.get("poster_path"):
        meta["poster_path"] = r["poster_path"]
    return str(tid), title, year, meta, r.get("vote_average")


def _tmdb_similar(api_key: str, domain: str, seeds: list[tuple[int, str]],
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
                    key, title, year, meta, score = parsed
                    pool.add(domain, "tmdb", key, title, year, meta,
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


def _genre_counts(conn: sqlite3.Connection, positives: list[tuple[int, float]]) -> Counter[str]:
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


def _top_genres(conn: sqlite3.Connection, positives: list[tuple[int, float]]) -> list[str]:
    return [name for name, _ in _genre_counts(conn, positives).most_common(TOP_GENRES)]


def _tmdb_discover(conn: sqlite3.Connection, api_key: str, domain: str,
                   positives: list[tuple[int, float]], pool: _Pool,
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
            key, title, year, meta, score = parsed
            pool.add(domain, "tmdb", key, title, year, meta, "tmdb_discover", None, score)


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


def _least_seen_decade(conn: sqlite3.Connection, positives: list[tuple[int, float]]) -> int:
    counts: Counter[int] = Counter()
    for iid, _ in positives:
        row = conn.execute("SELECT year FROM items WHERE id=?", (iid,)).fetchone()
        year = row["year"] if row else None
        if year and SERENDIPITY_DECADES[0] <= year < SERENDIPITY_DECADES[-1] + 10:
            counts[year - year % 10] += 1
    # earliest decade wins ties so the probe is deterministic
    return min(SERENDIPITY_DECADES, key=lambda d: (counts[d], d))


def _tmdb_serendipity(conn: sqlite3.Connection, api_key: str, domain: str,
                      positives: list[tuple[int, float]], pool: _Pool,
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
                key, title, year, meta, score = parsed
                pool.add(domain, "tmdb", key, title, year, meta,
                         "serendipity_tmdb", None, score)


# ── Last.fm ──────────────────────────────────────────────────────────


def _artist_query_params(conn: sqlite3.Connection, item_id: int, api_key: str,
                         limit: int) -> dict[str, Any] | None:
    """artist.getsimilar params for an item, mbid-first; None when the
    item offers neither an mbid nor a usable name."""
    idents = db.identities_of(conn, item_id)
    params: dict[str, Any] = {"method": "artist.getsimilar", "api_key": api_key,
                              "format": "json", "limit": limit}
    if mbid := idents.get("mbid"):
        params["mbid"] = mbid
    else:
        row = conn.execute("SELECT title FROM items WHERE id=?", (item_id,)).fetchone()
        name = (row["title"] if row else None) or idents.get("name")
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
                    a: dict[str, Any]) -> tuple[str, str, str, str | None, float] | None:
    """(ns, key, name, mbid, match) for one similar-artist entry.

    Entries carrying an mbid resolve the item and teach it the name too:
    Last.fm asserting name↔mbid identity beats the fuzzy MusicBrainz
    search enrich relies on, so a name-keyed twin (scrobbles often omit
    the mbid) merges into the mbid item and one artist never lives on as
    two rows. A verdict on either half must survive the merge, so the
    winner inherits the pool exclusion."""
    name = a.get("name")
    if not name or not ids.normalize_key(str(name)):
        return None
    mb = a.get("mbid") or None
    if mb:
        item_id = db.resolve_item(conn, "artist", "mbid", mb, title=name)
        twin = db.lookup_item(conn, "artist", "name", name)
        winner = db.attach_identity(conn, item_id, "name", name)
        # a refused attach (the spelling belongs to a DIFFERENT artist
        # with its own mbid) must not carry that artist's verdict over —
        # only a twin that actually merged in propagates its exclusion
        merged = twin is not None and twin != item_id \
            and db.lookup_item(conn, "artist", "name", name) != twin
        if item_id in pool.excluded or (merged and twin in pool.excluded):
            pool.excluded.add(winner)
        ns, key = "mbid", mb
    else:
        ns, key = "name", name
    try:
        score = float(a.get("match") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return ns, key, name, mb, score


def _lastfm_similar(conn: sqlite3.Connection, api_key: str, seed_ids: list[int],
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
            ns, key, name, _mb, score = resolved
            pool.add("artist", ns, key, name, None, {}, "lastfm_similar", seed_id, score)
    # accumulate: run() calls this once per profile
    stats["seeds"]["artist"] = stats["seeds"].get("artist", 0) + used


# ── Last.fm: top albums ──────────────────────────────────────────────


def _artist_mbid(conn: sqlite3.Connection, item_id: int) -> str | None:
    return db.identities_of(conn, item_id).get("mbid")


def _top_albums(data: Any) -> list[dict[str, Any]]:
    albums = ((data or {}).get("topalbums") or {}).get("album") or []
    if isinstance(albums, dict):  # last.fm collapses single-element lists
        albums = [albums]
    return albums


def _lastfm_top_albums(conn: sqlite3.Connection, api_key: str, seed_ids: list[int],
                       pool: _Pool, stats: dict[str, Any]) -> None:
    """Most-played albums of the liked artists.

    Albums by artists already in the library are exactly the point of
    this source (own the artist, missing the record), so only the usual
    item-level exclusions apply — no artist-level filtering.

    Last.fm album mbids are sometimes RELEASE mbids rather than the
    release-group mbids Lidarr wants; store them as-is. Enrich owns the
    release-group resolution — MB /release-group/{id} 404s on a release
    id, and enrich absorbs that failure gracefully.
    """
    used = 0
    for seed_id in seed_ids:
        mbid = _artist_mbid(conn, seed_id)
        if not mbid:
            continue  # name-only artists wait until enrich reveals an mbid
        used += 1
        try:
            data = http.get_json(LASTFM, params={
                "method": "artist.gettopalbums", "api_key": api_key, "format": "json",
                "mbid": mbid, "limit": ALBUMS_PER_ARTIST})
        except http.ApiError:
            stats["errors"] += 1
            continue
        # enumerate over the full response: an mbid-less album still holds
        # its playcount rank, it just never becomes an item.
        for rank, album in enumerate(_top_albums(data)):
            album_mbid, title = album.get("mbid") or None, album.get("name")
            if not (album_mbid and title):
                continue  # nothing Lidarr could ever act on
            artist = album.get("artist") or {}
            artist_mbid = artist.get("mbid") or mbid
            score = max(ALBUM_RANK_FLOOR, 1.0 - ALBUM_RANK_STEP * rank)
            # artist_mbid is a relation to another item, not a name of this
            # one, so it lives in meta (for actuate, the UI and enrich) —
            # never in identities.
            meta: dict[str, Any] = {"artist_mbid": artist_mbid}
            if artist.get("name"):
                meta["artist"] = artist["name"]
            pool.add("album", "mbid", album_mbid, title, None, meta,
                     "lastfm_top_albums", seed_id, score)
    # accumulate: run() calls this once per profile
    stats["seeds"]["album"] = stats["seeds"].get("album", 0) + used


# ── serendipity: Last.fm ─────────────────────────────────────────────


def _lastfm_serendipity(conn: sqlite3.Connection, api_key: str,
                        positives: list[tuple[int, float]], pool: _Pool,
                        stats: dict[str, Any]) -> None:
    """2-hop neighbourhood: getSimilar on the strongest hop-1 candidates
    (never library rows or seeds), scores damped for the extra hop."""
    seeds = {iid for iid, _ in positives}
    # hop-1 pool and the bubble check below are this profile's own view:
    # another profile's reachability says nothing about ours
    hops = [r["item_id"] for r in conn.execute(
        "SELECT item_id FROM candidates WHERE profile=? AND source='lastfm_similar'"
        " AND item_id NOT IN (SELECT item_id FROM library)"
        " ORDER BY external_score DESC, item_id", (pool.profile,))
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
            ns, key, name, _mb, score = resolved
            # anything already reachable at hop 1 (or via another source)
            # is bubble-adjacent, not serendipity — leave it be
            item_id = db.lookup_item(conn, "artist", ns, key)
            if item_id is not None and conn.execute(
                    "SELECT 1 FROM candidates WHERE profile=? AND item_id=?"
                    " AND source!='serendipity_lastfm'",
                    (pool.profile, item_id)).fetchone() is not None:
                continue
            pool.add("artist", ns, key, name, None, {},
                     "serendipity_lastfm", hop_id, score * SERENDIPITY_DAMPING)
