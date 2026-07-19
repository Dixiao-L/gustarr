"""Offline tests for the shared HTTP layer (httpx.MockTransport)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from gustarr import http as ghttp

URL = "https://example.test/api"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(ghttp, "HOST_DELAYS", {})
    ghttp._last_call.clear()


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr(ghttp.time, "sleep", recorded.append)
    return recorded


def _transport(handler):
    return httpx.MockTransport(handler)


def test_retry_after_is_capped(sleeps):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(429, headers={"Retry-After": "86400"}, text="slow down")

    with pytest.raises(ghttp.ApiError) as ei:
        ghttp.request_json("GET", URL, retries=3, transport=_transport(handler))
    assert len(calls) == 4
    assert sleeps and all(w <= ghttp.RETRY_AFTER_CAP for w in sleeps)
    assert ei.value.status == 429


def test_no_sleep_after_final_attempt_status(sleeps):
    def handler(request):
        return httpx.Response(503, headers={"Retry-After": "86400"})

    with pytest.raises(ghttp.ApiError):
        ghttp.request_json("GET", URL, retries=2, transport=_transport(handler))
    # retries=2 -> 3 attempts, but only 2 sleeps: the last attempt raises immediately.
    assert len(sleeps) == 2


def test_no_sleep_after_final_attempt_transport_error(sleeps):
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(ghttp.ApiError) as ei:
        ghttp.request_json("GET", URL, retries=2, transport=_transport(handler))
    assert len(sleeps) == 2
    assert ei.value.status is None
    assert "gave up after 3 attempts" in str(ei.value)


def test_no_sleep_at_all_with_zero_retries(sleeps):
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(ghttp.ApiError):
        ghttp.request_json("GET", URL, retries=0, transport=_transport(handler))
    assert sleeps == []


def test_backoff_without_retry_after_is_capped(sleeps):
    def handler(request):
        return httpx.Response(503)

    with pytest.raises(ghttp.ApiError):
        ghttp.request_json("GET", URL, retries=3, transport=_transport(handler))
    assert sleeps == [1, 2, 4]


def test_api_error_detail_populated_on_400():
    body = '[{"errorMessage": "This movie has already been added"}]'

    def handler(request):
        return httpx.Response(400, text=body)

    with pytest.raises(ghttp.ApiError) as ei:
        ghttp.request_json("POST", URL, retries=0, transport=_transport(handler))
    assert ei.value.status == 400
    assert ei.value.detail == body
    assert getattr(ei.value, "detail", "") == body


def test_api_error_detail_defaults_empty():
    err = ghttp.ApiError("https://x.test", 401)
    assert err.detail == ""


def test_polite_wait_serializes_concurrent_threads(monkeypatch):
    """Two web threads hitting a politeness-delayed host must space their
    requests by the per-host delay: without the lock both read the same
    last-call time and fire 0s apart (MusicBrainz bans for that)."""
    delay = 0.1
    monkeypatch.setattr(ghttp, "HOST_DELAYS", {"example.test": delay})
    times: list[float] = []

    def handler(request):
        times.append(time.monotonic())
        return httpx.Response(200, json={"ok": True})

    transport = _transport(handler)

    def call():
        ghttp.request_json("GET", URL, retries=0, transport=transport)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(times) == 2
    first, second = sorted(times)
    # the small slack absorbs the handler overhead between the spacing
    # write and the recorded call time
    assert second - first >= delay - 0.02


def test_success_after_retry(sleeps):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})

    assert ghttp.request_json("GET", URL, retries=3,
                              transport=_transport(handler)) == {"ok": True}
    # Small Retry-After values below the cap are honored verbatim.
    assert sleeps == [2.0, 2.0]
