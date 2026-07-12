"""Built-in nightly scheduler for container deployments.

Off by default. When ``[scheduler] nightly = "HH:MM"`` (local time) is
set, ``gustarr web`` starts one daemon thread that checks the clock once
a minute and launches ``gustarr run nightly`` as a *subprocess* — the
pipeline must never block the web event loop, and a pipeline crash must
never take the UI down. One fire per day; a slot that comes up while the
previous run is still alive is skipped, not queued. NixOS/systemd users
should leave this unset and keep their timers — systemd is the better
privilege boundary and survives web-process restarts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime
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

    def __init__(self, at: str, popen: Callable[[list[str]], Any] = subprocess.Popen):
        self.hour, self.minute = _parse_hhmm(at)
        self._popen = popen
        self._proc: Any = None
        self._last_day: str | None = None

    def prime(self, now: datetime) -> None:
        # Booting the web UI at noon must not instantly fire a pipeline
        # whose slot was 04:30 — a slot already past today counts as spent.
        if (now.hour, now.minute) >= (self.hour, self.minute):
            self._last_day = now.date().isoformat()

    def tick(self, now: datetime) -> bool:
        """One clock check; returns True when a run was started."""
        if self._proc is not None and self._proc.poll() is not None:
            print(f"scheduler: nightly run exited with code {self._proc.returncode}", flush=True)
            self._proc = None
        # >= instead of == so sleep drift across the target minute cannot
        # silently lose a day; _last_day keeps it to one fire per day.
        if (now.hour, now.minute) < (self.hour, self.minute):
            return False
        day = now.date().isoformat()
        if self._last_day == day:
            return False
        self._last_day = day
        if self._proc is not None:
            print("scheduler: previous nightly run still alive — skipping today's", flush=True)
            return False
        cmd = _command()
        self._proc = self._popen(cmd)
        print(f"scheduler: started {' '.join(cmd)} (pid {self._proc.pid})", flush=True)
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
    fire the pipeline at the configured local time, repeat. Scheduling
    deliberately does NOT live inside the web process: one process, one
    job. Container users run this as a second service from the same
    image; systemd/cron users don't run it at all."""
    at = (cfg.raw.get("scheduler") or {}).get("nightly")
    if not at:
        raise SystemExit("gustarr schedule: [scheduler] nightly = \"HH:MM\" is not configured")
    print(f"scheduler: nightly pipeline at {at} local time", flush=True)
    Scheduler(str(at)).run_forever()
