"""Built-in pipeline scheduler for container deployments.

Off by default. When ``[scheduler] nightly = "HH:MM"`` (local time) is
set, ``gustarr schedule`` — a dedicated foreground process, never a
thread inside ``gustarr web`` — checks once a minute and launches
``gustarr run nightly`` as a *subprocess*: a pipeline crash must never
take the scheduler down. Two triggers share that check: the clock (one
fire per day; a slot that comes up while the previous run is still
alive is skipped, not queued) and the ``{data_dir}/run-requested``
sentinel that the web UI's RUN NOW button touches — present means fire
immediately, and the file is consumed (deleted) even when a live run
forces a skip. NixOS/systemd users should leave this unset and keep
their timer and path units — systemd is the better privilege boundary
and survives restarts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import Config, ConfigError


def _parse_hhmm(at: str) -> tuple[int, int]:
    hh, sep, mm = at.partition(":")
    if sep and hh.isdigit() and mm.isdigit():
        hour, minute = int(hh), int(mm)
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    raise ConfigError(f'[scheduler] nightly must be "HH:MM" (local time), got {at!r}')


def _command() -> list[str]:
    # The console script re-resolves the running venv/container install;
    # sys.argv[0] covers odd launches where nothing is on PATH.
    return [shutil.which("gustarr") or sys.argv[0], "run", "nightly"]


class Scheduler:
    """Fire-at-HH:MM state machine, separated from the thread so tests can
    drive it with a fake clock and a fake Popen."""

    def __init__(
        self,
        at: str,
        popen: Callable[[list[str]], Any] = subprocess.Popen,
        sentinel: Path | None = None,
    ):
        self.hour, self.minute = _parse_hhmm(at)
        self.sentinel = sentinel
        self._popen = popen
        self._proc: Any = None
        self._last_day: str | None = None

    def prime(self, now: datetime) -> None:
        # Booting the web UI at noon must not instantly fire a pipeline
        # whose slot was 04:30 — a slot already past today counts as spent.
        if (now.hour, now.minute) >= (self.hour, self.minute):
            self._last_day = now.date().isoformat()

    def _start(self) -> None:
        cmd = _command()
        self._proc = self._popen(cmd)
        print(f"scheduler: started {' '.join(cmd)} (pid {self._proc.pid})", flush=True)

    def _consume_sentinel(self) -> bool:
        """Consume the web UI's RUN NOW request; True when it started a run."""
        if self.sentinel is None or not self.sentinel.exists():
            return False
        # Deleted before anything else so one button press is one request:
        # a skip must not leave the file behind to re-fire every minute.
        self.sentinel.unlink(missing_ok=True)
        if self._proc is not None:
            print("scheduler: run requested but a run is still alive — skipping", flush=True)
            return False
        self._start()
        return True

    def tick(self, now: datetime) -> bool:
        """One per-minute check of both triggers; True when a run started."""
        if self._proc is not None and self._proc.poll() is not None:
            print(f"scheduler: nightly run exited with code {self._proc.returncode}", flush=True)
            self._proc = None
        # Sentinel before clock: RUN NOW must not wait on slot bookkeeping,
        # and a sentinel-started run makes a slot landing the same minute
        # hit the alive-guard below (skip, never queue).
        started = self._consume_sentinel()
        # >= instead of == so sleep drift across the target minute cannot
        # silently lose a day; _last_day keeps it to one fire per day.
        if (now.hour, now.minute) < (self.hour, self.minute):
            return started
        day = now.date().isoformat()
        if self._last_day == day:
            return started
        self._last_day = day
        if self._proc is not None:
            print("scheduler: previous nightly run still alive — skipping today's", flush=True)
            return started
        self._start()
        return True

    def run_forever(
        self,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.prime(clock())
        while True:
            sleep(60)
            self.tick(clock())


def main(cfg: Config) -> None:
    """`gustarr schedule`: a single-purpose foreground process — sleep,
    check the clock slot and the RUN NOW sentinel, fire the pipeline,
    repeat. Scheduling deliberately does NOT live inside the web
    process: one process, one job. Container users run this as a second
    service from the same image; systemd/cron users don't run it at all
    (their path unit consumes the sentinel instead)."""
    at = (cfg.raw.get("scheduler") or {}).get("nightly")
    if not at:
        raise SystemExit("gustarr schedule: [scheduler] nightly = \"HH:MM\" is not configured")
    print(f"scheduler: nightly pipeline at {at} local time", flush=True)
    Scheduler(str(at), sentinel=Path(cfg.data_dir) / "run-requested").run_forever()
