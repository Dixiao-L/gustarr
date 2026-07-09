"""Shared HTTP plumbing: retries, backoff, per-host politeness delays.

Every external API (TMDb, Last.fm, MusicBrainz, ListenBrainz, the *arrs,
Jellyfin) goes through get_json/post_json so rate-limit behaviour is
uniform and testable — tests inject an httpx.MockTransport via the
``transport`` argument.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

USER_AGENT = "gustarr/0.1 (+https://github.com/Dixiao-L/gustarr)"

# MusicBrainz requires >=1s between requests; the rest tolerate more.
HOST_DELAYS = {
    "musicbrainz.org": 1.1,
    "ws.audioscrobbler.com": 0.25,
    "api.listenbrainz.org": 0.5,
    "api.themoviedb.org": 0.05,
}

# A proxy can send Retry-After: 86400; honoring it verbatim stalls the whole
# single-process pipeline, so clamp it near the 8s exponential-backoff ceiling.
RETRY_AFTER_CAP = 60.0

_last_call: dict[str, float] = {}


class ApiError(Exception):
    def __init__(self, url: str, status: int | None, detail: str = ""):
        self.url, self.status, self.detail = url, status, detail
        super().__init__(f"{status or 'ERR'} {url} {detail}".strip())


def _polite_wait(host: str) -> None:
    delay = HOST_DELAYS.get(host)
    if delay is None:
        return
    elapsed = time.monotonic() - _last_call.get(host, 0.0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_call[host] = time.monotonic()


def request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    merged_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    host = httpx.URL(url).host
    last_exc: Exception | None = None
    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as client:
        for attempt in range(retries + 1):
            _polite_wait(host)
            try:
                resp = client.request(
                    method, url, params=params, json=json_body, headers=merged_headers)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = min(float(retry_after), RETRY_AFTER_CAP) \
                    if retry_after and retry_after.isdigit() else min(2**attempt, 8)
                last_exc = ApiError(url, resp.status_code, resp.text[:200])
                if attempt < retries:
                    time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise ApiError(url, resp.status_code, resp.text[:200])
            if not resp.content:
                return None
            return resp.json()
    raise ApiError(url, getattr(last_exc, "status", None),
                   f"gave up after {retries + 1} attempts: {last_exc}")


def get_json(url: str, **kw: Any) -> Any:
    return request_json("GET", url, **kw)


def post_json(url: str, **kw: Any) -> Any:
    return request_json("POST", url, **kw)
