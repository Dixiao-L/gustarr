"""Vectorise items for ranking.

One multilingual document per item (bge-m3 copes with mixed CJK/Latin
metadata), encoded lazily so every other command works without the ml
extra installed.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np

from .. import db
from ..config import Config

_OVERVIEW_CHARS = 1200
_BIO_CHARS = 800


def _names(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, (list, tuple)):
        parts = []
        for v in values:
            if isinstance(v, dict):
                v = v.get("name") or v.get("title") or ""
            if v:
                parts.append(str(v))
        return ", ".join(parts)
    return str(values)


def build_doc(row: dict[str, Any]) -> str:
    """Domain-tagged text document for one item row; pure, no I/O."""
    meta = row.get("meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    title = (row.get("title") or "").strip()
    if not title:
        return ""
    domain = row.get("domain") or ""
    year = row.get("year")
    lines = [f"{domain}: {title} ({year})" if year else f"{domain}: {title}"]
    if genres := _names(meta.get("genres")):
        lines.append(f"genres: {genres}")
    if domain == "artist":
        if tags := _names(meta.get("tags")):
            lines.append(f"tags: {tags}")
        if similar := _names(meta.get("similar")):
            lines.append(f"similar: {similar}")
        if bio := str(meta.get("bio") or "").strip():
            lines.append(f"bio: {bio[:_BIO_CHARS]}")
    else:
        if keywords := _names(meta.get("keywords")):
            lines.append(f"keywords: {keywords}")
        if tags := _names(meta.get("tags")):
            lines.append(f"tags: {tags}")
        if lang := meta.get("original_language"):
            lines.append(f"language: {lang}")
        if overview := str(meta.get("overview") or "").strip():
            lines.append(f"overview: {overview[:_OVERVIEW_CHARS]}")
    return "\n".join(lines)


def run(conn: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    """Embed enriched items that matter (have events / library / candidate
    rows) and lack a vector for the configured model."""
    model_name = cfg.model.embed_model
    rows = conn.execute(
        "SELECT id, domain, title, year, meta FROM items i"
        " WHERE i.enriched_at IS NOT NULL"
        "   AND NOT EXISTS (SELECT 1 FROM embeddings e"
        "                   WHERE e.item_id = i.id AND e.model = ?)"
        "   AND (EXISTS (SELECT 1 FROM events ev WHERE ev.item_id = i.id)"
        "        OR EXISTS (SELECT 1 FROM library l WHERE l.item_id = i.id)"
        "        OR EXISTS (SELECT 1 FROM candidates c WHERE c.item_id = i.id))",
        (model_name,),
    ).fetchall()

    ids: list[int] = []
    docs: list[str] = []
    skipped = 0
    for row in rows:
        doc = build_doc(dict(row))
        if not doc:
            skipped += 1
            continue
        ids.append(row["id"])
        docs.append(doc)
    if not ids:
        return {"embedded": 0, "skipped": skipped}

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("embedding requires: uv sync --extra ml") from exc

    encoder = SentenceTransformer(
        model_name, device=cfg.model.device, cache_folder=cfg.model.model_dir or None)
    vecs = np.asarray(
        encoder.encode(docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    ).astype(np.float16)
    for item_id, vec in zip(ids, vecs, strict=True):
        db.put_embedding(conn, item_id, model_name, vec.tobytes(), int(vec.shape[0]))
    return {"embedded": len(ids), "skipped": skipped}
