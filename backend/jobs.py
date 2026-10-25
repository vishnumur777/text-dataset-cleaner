"""Background jobs for slow dataset loads.

Downloading a dataset file and parsing it can take a minute, which is too long
for one HTTP request to sit on: the browser cannot draw a progress bar while it
waits, and a proxy may time the request out. So the work runs in a daemon
thread and the browser polls a small JSON snapshot instead.

The job object is the single source of truth for progress: `done`/`total` are
bytes while downloading, and `phase` tells the UI which sentence to show.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .dataio import DataError
from .sources import SourceError
from .store import SessionError

log = logging.getLogger("tdc.jobs")

MAX_JOBS = 40
KEEP_FINISHED_S = 30 * 60

Work = Callable[["Job"], dict[str, Any]]


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"          # running | done | error
    phase: str = "starting"          # starting | download | extract | parse | ready
    message: str = "Starting…"
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    speed_bps: float = 0.0
    _speed_done: int = 0
    _speed_time: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------ state
    def update(
        self,
        *,
        phase: str | None = None,
        message: str | None = None,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        with self.lock:
            if phase is not None:
                self.phase = phase
            if message is not None:
                self.message = message
            if done is not None:
                self.done = max(int(done), 0)
            if total is not None:
                self.total = max(int(total), 0)

    def progress(self, done: int, total: int) -> None:
        """Callback handed to the download helpers. Also tracks throughput."""
        now = time.time()
        with self.lock:
            moved = done - self._speed_done
            elapsed = now - self._speed_time
            if moved > 0 and elapsed >= 0.15:
                sample = moved / elapsed
                # Smoothed so a slow first chunk does not define the estimate.
                self.speed_bps = sample if self.speed_bps <= 0 else 0.65 * self.speed_bps + 0.35 * sample
                self._speed_done = done
                self._speed_time = now
            self.update(done=done, total=total)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            percent = None
            if self.total > 0:
                percent = min(100, int(round(self.done / self.total * 100)))
            remaining = max(self.total - self.done, 0)
            eta = None
            if self.speed_bps > 0 and remaining > 0 and self.phase == "download":
                eta = round(remaining / self.speed_bps, 1)
            elapsed = (self.finished_at or time.time()) - self.created_at
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "phase": self.phase,
                "message": self.message,
                "done": self.done,
                "total": self.total,
                "percent": percent,
                "speed_bps": round(self.speed_bps, 1),
                "eta_s": eta,
                "elapsed_s": round(elapsed, 1),
                "result": self.result,
                "error": self.error,
            }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- public
    def start(self, kind: str, work: Work) -> dict[str, Any]:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._trim()
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run,
            args=(job, work),
            name=f"load-{kind}-{job.id}",
            daemon=True,
        ).start()
        return job.snapshot()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.snapshot() if job else None

    # ----------------------------------------------------------------- worker
    def _run(self, job: Job, work: Work) -> None:
        try:
            result = work(job)
            with job.lock:
                job.result = result
                job.status = "done"
                job.phase = "ready"
                job.message = "Ready to clean."
                if job.total:
                    job.done = job.total
                job.finished_at = time.time()
        except SourceError as exc:
            self._fail(job, exc.code, exc.message, exc.hint)
        except DataError as exc:
            self._fail(job, exc.code, exc.message, exc.hint)
        except SessionError as exc:
            self._fail(job, exc.code, exc.message, exc.hint)
        except Exception as exc:  # noqa: BLE001
            log.error("load job %s failed: %s\n%s", job.id, exc, traceback.format_exc())
            self._fail(
                job,
                "internal_error",
                "The dataset could not be loaded.",
                "Try again; if it keeps happening, check the server log and free disk space.",
            )

    def _fail(self, job: Job, code: str, message: str, hint: str = "") -> None:
        with job.lock:
            job.status = "error"
            job.phase = "error"
            job.message = message
            job.error = {"code": code, "message": message, "hint": hint}
            job.finished_at = time.time()

    def _trim(self) -> None:
        cutoff = time.time() - KEEP_FINISHED_S
        for job_id, job in list(self._jobs.items()):
            if job.status != "running" and (job.finished_at or 0) < cutoff:
                self._jobs.pop(job_id, None)
        while len(self._jobs) > MAX_JOBS:
            oldest = min(self._jobs.values(), key=lambda j: j.created_at)
            self._jobs.pop(oldest.id, None)


jobs = JobManager()
