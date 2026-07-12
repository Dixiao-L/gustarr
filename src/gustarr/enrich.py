"""Metadata enrichment: TMDb for movies/series, MusicBrainz + Last.fm for music.

Second pipeline stage. Besides filling items.meta for the embedder, this
is where weak ids get upgraded: imdb-only movies, tmdb-only series and
Last.fm name-keyed artists resolve to their authoritative namespace and
merge (db.merge_item) so accumulated events/candidates follow the item.
Permanent per-item failures (bad/missing ids, 4xx) stamp enriched_at
with an enrich_error note so one bad item can never wedge the queue;
transient failures (outages, rate limits, bad credentials) leave
enriched_at NULL so the item retries next run.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from . import db, http, ids
from .config import Config

TMDB = "https://api.themoviedb.org/3"
MB = "https://musicbrainz.org/ws/2"
LASTFM = "https://ws.audioscrobbler.com/2.0/"
DEEZER = "https://api.deezer.com"
CAA = "https://coverartarchive.org"

# Below this MusicBrainz search score a name match is too fuzzy to
# auto-merge; keeping the fallback id beats pointing someone's scrobble
# history at the wrong artist.
MB_MERGE_SCORE = 90

_LINK_RE = re.compile(r"<a\s[^>]*>.*?</a>\.?", re.S)
_READ_MORE_RE = re.compile(r"\s*Read more(?: on Last\.fm)?\.?\s*$", re.I)


def run(
    conn: sqlite3.Connection,
    cfg: Config,
    domain: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    stats = {"enriched": 0, "merged": 0, "skipped": 0, "errors": 0, "alias_conflicts": 0}
    sql = (
        "SELECT * FROM items WHERE enriched_at IS NULL"
        + (" AND domain=?" if domain else "")
        # Rankable domains first: a first sync can mint 10k+ tracks, and
        # artists/albums/movies/series are what train/rank actually consume.
        + " ORDER BY CASE WHEN domain IN ('artist','album','movie','series') THEN 0 ELSE 1 END,"
        " (EXISTS(SELECT 1 FROM candidates c WHERE c.item_id = items.id)"
        "   OR EXISTS(SELECT 1 FROM events e WHERE e.item_id = items.id)) DESC,"
        " updated_at DESC LIMIT ?"
    )
    args = ([domain] if domain else []) + [-1 if limit is None else limit]
    processed = 0
    for row in conn.execute(sql, args).fetchall():
        # MusicBrainz politeness makes long runs span hours; commit as we
        # go so a timeout/restart never rolls back finished work.
        processed += 1
        if processed % 50 == 0:
            conn.commit()
        cur = conn.execute("SELECT enriched_at FROM items WHERE id=?", (row["id"],)).fetchone()
        if cur is None or cur["enriched_at"] is not None:
            continue  # merged away or already handled earlier this run
        eff = {"id": row["id"]}  # tracks the post-merge id for error attribution
        try:
            _enrich_one(conn, cfg, row, eff, stats)
        except Exception as exc:
            # Only genuinely per-item failures may stamp enriched_at:
            # 401/403/429, 5xx, transport errors (status None) and
            # unexpected bugs are service-level, so the item stays
            # queued instead of being poisoned by one bad night.
            permanent = isinstance(exc, LookupError) or (
                isinstance(exc, http.ApiError) and exc.status is not None
                and 400 <= exc.status < 500 and exc.status not in (401, 403, 429))
            db.upsert_item(conn, eff["id"], row["domain"],
                           meta={"enrich_error": str(exc)}, enriched=permanent)
            stats["errors"] += 1
    return stats


def _enrich_one(
    conn: sqlite3.Connection,
    cfg: Config,
    row: sqlite3.Row,
    eff: dict[str, str],
    stats: dict[str, int],
) -> None:
    item_id = row["id"]
    ids_map = json.loads(row["ids"])
    _, ns, key = ids.parse(item_id)
    if ns != "lastfm":  # the id's own namespace is an external id too
        ids_map.setdefault(ns, key)
    domain = row["domain"]
    if domain == "movie":
        _movie(conn, cfg, item_id, ids_map, eff, stats)
    elif domain == "series":
        _series(conn, cfg, item_id, ids_map, eff, stats)
    elif domain == "artist":
        _artist(conn, cfg, item_id, row["title"], ids_map, eff, stats)
    else:  # album | track
        _release_or_recording(conn, item_id, domain, row["title"], ids_map,
                              json.loads(row["meta"] or "{}"), stats)


# ── movies / series (TMDb) ───────────────────────────────────────────


def _movie(conn, cfg, item_id, ids_map, eff, stats) -> None:
    api_key = cfg.tmdb.get("api_key")
    if not api_key:
        stats["skipped"] += 1
        return
    tmdb_id = ids_map.get("tmdb")
    if not tmdb_id:
        imdb = ids_map.get("imdb")
        if not imdb:
            raise LookupError("movie has neither tmdb nor imdb id")
        found = http.get_json(f"{TMDB}/find/{imdb}",
                              params={"api_key": api_key, "external_source": "imdb_id"})
        results = (found or {}).get("movie_results") or []
        if not results:
            raise LookupError(f"tmdb /find: no movie for imdb {imdb}")
        tmdb_id = results[0]["id"]
        canonical = ids.make("movie", "tmdb", str(tmdb_id))
        if canonical != item_id:
            db.merge_item(conn, item_id, canonical)
            stats["merged"] += 1
            item_id = eff["id"] = canonical
    data = http.get_json(
        f"{TMDB}/movie/{tmdb_id}",
        params={"api_key": api_key, "append_to_response": "keywords,videos",
                "language": "en-US"})
    kw = data.get("keywords") or {}
    meta = {
        "genres": [g["name"] for g in data.get("genres") or []],
        "keywords": [k["name"] for k in kw.get("keywords") or kw.get("results") or []],
        "overview": data.get("overview"),
        "original_language": data.get("original_language"),
        "popularity": data.get("popularity"),
        "vote_average": data.get("vote_average"),
        "runtime": data.get("runtime"),
    }
    if data.get("poster_path"):
        meta["poster_path"] = data["poster_path"]
    trailer = _trailer_key(data)
    if trailer:
        meta["trailer"] = trailer
    db.upsert_item(conn, item_id, "movie", title=data.get("title"),
                   year=_year(data.get("release_date")), ids={**ids_map, "tmdb": tmdb_id},
                   meta=meta, enriched=True)
    stats["enriched"] += 1


def _series(conn, cfg, item_id, ids_map, eff, stats) -> None:
    api_key = cfg.tmdb.get("api_key")
    if not api_key:
        stats["skipped"] += 1
        return
    tmdb_id = ids_map.get("tmdb")
    if not tmdb_id:
        tvdb = ids_map.get("tvdb")
        if not tvdb:
            raise LookupError("series has neither tvdb nor tmdb id")
        found = http.get_json(f"{TMDB}/find/{tvdb}",
                              params={"api_key": api_key, "external_source": "tvdb_id"})
        results = (found or {}).get("tv_results") or []
        if not results:
            raise LookupError(f"tmdb /find: no series for tvdb {tvdb}")
        tmdb_id = results[0]["id"]
    data = http.get_json(
        f"{TMDB}/tv/{tmdb_id}",
        params={"api_key": api_key, "append_to_response": "keywords,external_ids,videos",
                "language": "en-US"})
    # Sonarr can only add by tvdb id, so tmdb-keyed candidates must
    # upgrade to the canonical tvdb namespace (mirrors _movie's
    # imdb→tmdb path) or actuation can never reach them.
    tvdb_id = (data.get("external_ids") or {}).get("tvdb_id") or ids_map.get("tvdb")
    if tvdb_id:
        ids_map["tvdb"] = tvdb_id
        canonical = ids.make("series", "tvdb", str(tvdb_id))
        if canonical != item_id:
            db.merge_item(conn, item_id, canonical)
            stats["merged"] += 1
            item_id = eff["id"] = canonical
    kw = data.get("keywords") or {}
    meta = {
        "genres": [g["name"] for g in data.get("genres") or []],
        # TV nests keywords under "results", unlike movies' "keywords"
        "keywords": [k["name"] for k in kw.get("results") or kw.get("keywords") or []],
        "overview": data.get("overview"),
        "original_language": data.get("original_language"),
        "popularity": data.get("popularity"),
        "number_of_seasons": data.get("number_of_seasons"),
    }
    if data.get("poster_path"):
        meta["poster_path"] = data["poster_path"]
    trailer = _trailer_key(data)
    if trailer:
        meta["trailer"] = trailer
    if not tvdb_id:
        meta["no_tvdb"] = True  # un-addable in Sonarr; apply reports why
    db.upsert_item(conn, item_id, "series", title=data.get("name"),
                   year=_year(data.get("first_air_date")), ids={**ids_map, "tmdb": tmdb_id},
                   meta=meta, enriched=True)
    stats["enriched"] += 1


def _trailer_key(data: dict[str, Any]) -> str | None:
    """Best embeddable clip: official Trailer > any Trailer > Teaser; YouTube only,
    since that's the only site the UI can embed by key."""
    vids = [v for v in (data.get("videos") or {}).get("results") or []
            if isinstance(v, dict) and v.get("site") == "YouTube" and v.get("key")]
    for want in (lambda v: v.get("official") and v.get("type") == "Trailer",
                 lambda v: v.get("type") == "Trailer",
                 lambda v: v.get("type") == "Teaser"):
        for v in vids:
            if want(v):
                return v["key"]
    return None


# ── artists (MusicBrainz + Last.fm) ──────────────────────────────────


def _artist(conn, cfg, item_id, title, ids_map, eff, stats) -> None:
    name = title or ids.parse(item_id)[2]
    mbid = ids_map.get("mbid")
    if not mbid:
        hit = _mb_search_artist(name)
        if hit is None:
            # No confident MB match: enrich the fallback item from
            # Last.fm alone so it still leaves the queue.
            meta = _lastfm_artist_meta(cfg, None, name)
            image = _deezer_artist_image(name)
            if image:
                meta["image"] = image
            db.upsert_item(conn, item_id, "artist", meta=meta, enriched=True)
            stats["enriched"] += 1
            return
        mbid = hit["id"]
        canonical = ids.make("artist", "mbid", mbid)
        if canonical != item_id:
            db.merge_item(conn, item_id, canonical)
            stats["merged"] += 1
            item_id = eff["id"] = canonical
    data = http.get_json(f"{MB}/artist/{mbid}",
                         params={"inc": "tags+genres+aliases", "fmt": "json"})
    # Some artists carry hundreds of locale aliases; 30 covers every real
    # spelling without bloating meta. Stored raw (as MB wrote them) so the
    # UI can show them; normalization happens only at comparison time.
    alias_names = [a["name"] for a in data.get("aliases") or []
                   if isinstance(a, dict) and a.get("name")][:30]
    meta = {
        "tags": _top_tag_names(data.get("tags")),
        "genres": _top_tag_names(data.get("genres"), cap=None),
        "type": data.get("type"),
        "country": data.get("country"),
        "begin_year": _year((data.get("life-span") or {}).get("begin")),
        # stored even when empty: key-present is `dedupe --fetch`'s
        # already-fetched marker, sparing a 1.1s MB round-trip per artist
        "aliases": alias_names,
    }
    # Bridge before the enriched write: a merged-in fallback twin drags
    # its stale title/meta along, and MB's authoritative values must land
    # last so they win the upsert merge.
    _bridge_artist_aliases(conn, item_id, [data.get("name") or name, *alias_names], stats)
    db.upsert_item(conn, item_id, "artist", title=data.get("name"),
                   ids={**ids_map, "mbid": mbid}, meta=meta, enriched=True)
    lf = _lastfm_artist_meta(cfg, mbid, data.get("name") or name)
    image = _deezer_artist_image(data.get("name") or name)
    if image:
        lf["image"] = image
    if lf:
        db.upsert_item(conn, item_id, "artist", meta=lf)
    stats["enriched"] += 1


def _mb_search_artist(name: str) -> dict[str, Any] | None:
    query = f'artist:"{name.replace(chr(34), "")}"'
    data = http.get_json(f"{MB}/artist", params={"query": query, "fmt": "json", "limit": 3})
    hits = (data or {}).get("artists") or []
    if not hits:
        return None
    if int(hits[0].get("score") or 0) >= MB_MERGE_SCORE:
        return hits[0]
    # A romaji query for a kana-primary artist legitimately scores far
    # below MB_MERGE_SCORE on the name field while being exactly the
    # artist asked for. An exact normalized match against a hit's alias
    # list is that proof, so it overrides the score gate.
    want = ids.normalize_key(name)
    for i, hit in enumerate(hits):
        aliases = hit.get("aliases")
        if aliases is None and i == 0 and hit.get("id"):
            # Search payloads can omit alias lists; one direct lookup of
            # the top hit settles it before the merge is given up on.
            detail = http.get_json(f"{MB}/artist/{hit['id']}",
                                   params={"inc": "aliases", "fmt": "json"})
            aliases = (detail or {}).get("aliases")
        for a in aliases or []:
            if isinstance(a, dict) and a.get("name") and ids.normalize_key(a["name"]) == want:
                return hit
    return None


def _bridge_artist_aliases(conn, canonical: str, names: list[str],
                           stats: dict[str, int]) -> None:
    """Point every spelling MB knows for this artist at its canonical mbid
    item. Cross-script variants (romaji vs kana/kanji) mint different
    fallback ids that normalize_key cannot fold, so the alias list is the
    only bridge: a fallback twin that already accumulated events (months
    of romaji scrobbles) merges in now, and the item_aliases row makes any
    future encounter of that spelling land on the canonical id on arrival.
    dedupe.py runs the same registration offline for pre-existing stores.
    """
    fallback_ids = set()
    for raw in names:
        if not raw or not isinstance(raw, str):
            continue
        try:
            # aliases are stored raw; ids.make is the normalizer, so this
            # is exactly the id a collector would mint for that spelling
            fallback_ids.add(ids.make("artist", "lastfm", raw))
        except ValueError:
            continue  # name normalizes to nothing
    fallback_ids.discard(canonical)
    for fid in sorted(fallback_ids):  # deterministic conflict attribution
        row = conn.execute(
            "SELECT canonical_id FROM item_aliases WHERE alias_id=?", (fid,)).fetchone()
        if row is not None:
            if row["canonical_id"] != canonical:
                # first-writer-wins: two artists genuinely sharing a
                # spelling would ping-pong the mapping forever otherwise
                stats["alias_conflicts"] += 1
            continue
        if conn.execute("SELECT 1 FROM items WHERE id=?", (fid,)).fetchone() is not None:
            db.merge_item(conn, fid, canonical)  # records the alias row itself
            stats["merged"] += 1
        else:
            conn.execute(
                "INSERT OR IGNORE INTO item_aliases (alias_id, canonical_id) VALUES (?,?)",
                (fid, canonical))


def _lastfm_artist_meta(cfg: Config, mbid: str | None, name: str | None) -> dict[str, Any]:
    api_key = cfg.lastfm.get("api_key")
    if not api_key:
        return {}
    base = {"method": "artist.getInfo", "api_key": api_key, "format": "json"}
    data = None
    if mbid:
        # Last.fm's mbid index is patchy; fall back to the name lookup.
        try:
            data = http.get_json(LASTFM, params={**base, "mbid": mbid})
        except http.ApiError:
            data = None
        if not (isinstance(data, dict) and data.get("artist")):
            data = None
    if data is None and name:
        data = http.get_json(LASTFM, params={**base, "artist": name})
    artist = data.get("artist") if isinstance(data, dict) else None
    if not isinstance(artist, dict):
        return {}
    meta: dict[str, Any] = {}
    summary = (artist.get("bio") or {}).get("summary")
    if summary:
        meta["bio"] = _clean_bio(summary)
    listeners = (artist.get("stats") or {}).get("listeners")
    try:
        meta["listeners"] = int(listeners)
    except (TypeError, ValueError):
        pass
    similar = (artist.get("similar") or {}).get("artist") or []
    meta["similar"] = [a["name"] for a in similar if isinstance(a, dict) and a.get("name")]
    return meta


def _clean_bio(summary: str) -> str:
    return _READ_MORE_RE.sub("", _LINK_RE.sub("", summary)).strip()


def _deezer_artist_image(name: str) -> str | None:
    """MusicBrainz stopped serving artist images and Last.fm's are blank
    stars, so the UI portrait comes from Deezer's keyless search. Purely
    decorative: any failure returns None rather than failing the item."""
    try:
        data = http.get_json(f"{DEEZER}/search/artist", params={"q": name, "limit": 1})
        hits = (data or {}).get("data") or []
        pic = hits[0].get("picture_big") if hits and isinstance(hits[0], dict) else None
    except Exception:
        return None
    # Deezer serves a generic silhouette from an empty /artist// path
    # for unknown names; the monogram beats a wrong-looking placeholder.
    if not pic or "/artist//" in pic:
        return None
    return pic


# ── albums / tracks (MusicBrainz) ────────────────────────────────────


def _release_or_recording(conn, item_id, domain, title, ids_map, meta, stats) -> None:
    mbid = ids_map.get("mbid")
    if domain == "album" and mbid and (not title or not meta.get("tags")):
        # Albums rank and actuate in their own slots now, so unlike
        # tracks they earn a real lookup. Missing tags also qualifies
        # so rows fast-stamped before albums ranked get upgraded when
        # requeued; titled+tagged albums stay on the cheap path below.
        _album(conn, item_id, mbid, ids_map, stats)
        return
    if not mbid or title:
        # Tracks feed artist aggregation, not ranking, and the
        # collectors already deliver their titles. An MB lookup per track
        # at the mandatory 1.1s spacing turns a first Last.fm sync (10k+
        # tracks) into hours of grind for metadata nothing consumes —
        # only title-less mbid items are worth the round-trip.
        db.upsert_item(conn, item_id, domain, enriched=True)
        stats["skipped"] += 1
        return
    data = http.get_json(f"{MB}/recording/{mbid}",
                         params={"inc": "artist-credits+tags", "fmt": "json"})
    meta = {
        "artists": _credit_names(data),
        "tags": _top_tag_names(data.get("tags")),
    }
    db.upsert_item(conn, item_id, domain, title=data.get("title"),
                   year=_year(data.get("date") or data.get("first-release-date")),
                   meta=meta, enriched=True)
    stats["enriched"] += 1


def _album(conn, item_id, mbid, ids_map, stats) -> None:
    # An album mbid is a RELEASE-GROUP id — that is the namespace
    # Lidarr's foreignAlbumId speaks, and it carries the first-release
    # date instead of one pressing's date.
    rg_params = {"inc": "artist-credits+tags+genres", "fmt": "json"}
    try:
        data = http.get_json(f"{MB}/release-group/{mbid}", params=rg_params)
    except http.ApiError as exc:
        if exc.status != 404:
            raise
        # Last.fm hands out RELEASE mbids for albums often enough that a
        # 404 here usually means wrong id *kind*, not a dead id: ask MB
        # which release-group the release belongs to before giving up.
        rel = http.get_json(f"{MB}/release/{mbid}",
                            params={"inc": "release-groups", "fmt": "json"})
        rg_id = ((rel or {}).get("release-group") or {}).get("id")
        if not rg_id:
            raise  # true double-404 (or group-less release): permanent
        canonical = ids.make("album", "mbid", rg_id)
        if canonical != item_id:
            db.merge_item(conn, item_id, canonical)
            stats["merged"] += 1
            item_id = canonical
        mbid = rg_id
        ids_map = {**ids_map, "mbid": rg_id}
        data = http.get_json(f"{MB}/release-group/{rg_id}", params=rg_params)
    artists = _credit_names(data)
    meta = {
        "artists": artists,
        "tags": _top_tag_names(data.get("tags")),
        "genres": _top_tag_names(data.get("genres"), cap=None),
        "type": data.get("primary-type"),
        # Stored unprobed: CAA 404s for coverless groups, but the UI
        # paints the monogram under the image, so a dead URL costs
        # nothing while probing would cost a round-trip per album.
        "image": f"{CAA}/release-group/{mbid}/front-250",
    }
    if artists:
        meta["artist"] = artists[0]
    new_ids = {**ids_map, "mbid": mbid}
    artist_mbid = _credit_artist_mbid(data)
    if artist_mbid:
        new_ids["artist_mbid"] = artist_mbid  # lets actuation add the artist to Lidarr too
    db.upsert_item(conn, item_id, "album", title=data.get("title"),
                   year=_year(data.get("first-release-date")),
                   ids=new_ids, meta=meta, enriched=True)
    stats["enriched"] += 1


def _credit_names(data: dict[str, Any]) -> list[str]:
    return [c["artist"]["name"] for c in data.get("artist-credit") or []
            if isinstance(c, dict) and isinstance(c.get("artist"), dict)]


def _credit_artist_mbid(data: dict[str, Any]) -> str | None:
    for c in data.get("artist-credit") or []:
        if isinstance(c, dict) and isinstance(c.get("artist"), dict) and c["artist"].get("id"):
            return c["artist"]["id"]
    return None


# ── shared ───────────────────────────────────────────────────────────


def _top_tag_names(tags: list[dict[str, Any]] | None, cap: int | None = 10) -> list[str]:
    ranked = sorted(tags or [], key=lambda t: -(t.get("count") or 0))
    names = [t["name"] for t in ranked if t.get("name")]
    return names[:cap] if cap else names


def _year(date_str: str | None) -> int | None:
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return None
