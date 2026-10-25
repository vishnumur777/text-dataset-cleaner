"""One-shot pre-cleaning for a whole column.

Cleaning a row on demand costs ~7 s of model time, so walking a 200-row column
with the play button is a long wait. This module pays that cost once, up front,
and makes every row instant afterwards:

    rules pass for every row        (milliseconds, no model)
      -> skip rows that are already clean
      -> de-duplicate identical cells (repeated text costs one call)
      -> run the model concurrently against llama.cpp's slots
      -> store each result on the session, so GET /row returns it immediately

The play button still exists for a single row: it re-runs just that cell.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from . import llm
from .cleaner import BoundaryModel, _split_punct, analyze, normalize, preserves

log = logging.getLogger("tdc.batch")

DEFAULT_WORKERS = int(os.environ.get("TDC_PRECLEAN_WORKERS", "3"))
MAX_WORKERS = 4
SKIP_CLEAN_ROWS = os.environ.get("TDC_PRECLEAN_SKIP_CLEAN", "1") != "0"


@dataclass
class Job:
    session_id: str
    total: int
    mode: str = "llm"
    status: str = "running"          # running | done | cancelled | error
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    scanned: int = 0                 # rows examined
    llm_total: int = 0               # rows that actually need the model
    llm_done: int = 0
    reused: int = 0                  # cells answered by a duplicate
    skipped: int = 0                 # rows already clean after the rules pass
    failed: int = 0
    model: str = ""
    provider: str = ""
    warning: str = ""
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------ state
    @property
    def eta_s(self) -> float | None:
        with self.lock:
            done = self.llm_done
            remaining = max(self.llm_total - done, 0)
            if self.status != "running" or done == 0 or remaining == 0:
                return 0.0 if remaining == 0 and done else None
            # The first couple of calls include ramp-up and the rules scan, so
            # their average is noise; show "estimating" until three are done.
            if done < 3:
                return None
            per_row = (time.time() - self.started_at) / done
        return round(per_row * remaining, 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = (self.finished_at or time.time()) - self.started_at
            return {
                "session_id": self.session_id,
                "status": self.status,
                "mode": self.mode,
                "total": self.total,
                "scanned": self.scanned,
                "llm_total": self.llm_total,
                "llm_done": self.llm_done,
                "reused": self.reused,
                "skipped": self.skipped,
                "failed": self.failed,
                "model": self.model,
                "provider": self.provider,
                "warning": self.warning,
                "error": self.error,
                "elapsed_s": round(elapsed, 1),
                "eta_s": self.eta_s,
            }


class BatchManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ public
    def start(self, session, mode: str = "llm", workers: int | None = None) -> dict[str, Any]:
        with self._lock:
            existing = self._jobs.get(session.id)
            if existing and existing.status == "running":
                return existing.snapshot()
            workers = max(1, min(workers or DEFAULT_WORKERS, MAX_WORKERS))
            job = Job(session_id=session.id, total=len(session.df), mode=mode)
            self._jobs[session.id] = job

        thread = threading.Thread(
            target=self._work,
            args=(job, session, mode, workers),
            name=f"preclean-{session.id}",
            daemon=True,
        )
        thread.start()
        return job.snapshot()

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(session_id)
        return job.snapshot() if job else {"status": "idle", "session_id": session_id, "total": 0}

    def cancel(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(session_id)
        if not job:
            return {"status": "idle", "session_id": session_id}
        job.cancel.set()
        return job.snapshot()

    # ------------------------------------------------------------------ worker
    def _work(self, job: Job, session, mode: str, workers: int) -> None:
        try:
            model = session.model()
            indices = list(range(len(session.df)))

            # Rows the user already saved keep their saved text; do not overwrite.
            with session.lock:
                saved = set(session.edits)

            rule_results: dict[int, dict[str, Any]] = {}
            # One model call per distinct cell. The model is given the
            # rule-cleaned text, never the raw cell, so it polishes spelling
            # instead of re-deciding word boundaries.
            pending: dict[str, dict[str, Any]] = {}

            for row in indices:
                if job.cancel.is_set():
                    break
                if row in saved:
                    continue
                original = session.original(row)
                analysis = analyze(original, model)
                rule_results[row] = analysis
                with job.lock:
                    job.scanned += 1

                if mode != "llm":
                    continue
                if SKIP_CLEAN_ROWS and not _needs_model(original, analysis, model):
                    job.skipped += 1
                    continue
                key = normalize(original)
                entry = pending.setdefault(key, {"text": analysis["cleaned"], "rows": []})
                entry["rows"].append(row)

            with job.lock:
                job.llm_total = len(pending) if mode == "llm" else 0

            if job.cancel.is_set():
                for row, analysis in rule_results.items():
                    self._store(session, row, analysis, engine="rules")
                self._finish(job, "cancelled")
                return

            if mode == "llm" and pending:
                provider = llm.detect()
                for row in rule_results:
                    self._store(session, row, rule_results[row], engine="rules")

                if not provider.get("ok"):
                    with job.lock:
                        job.warning = (
                            "No local model was running, so every row was cleaned with the rule engine. "
                            "Start llama-server or Ollama and run the pass again to add Gemma's spelling fixes."
                        )
                        job.failed = job.llm_total
                        job.llm_done = job.llm_total
                    self._finish(job, "done")
                    return

                with job.lock:
                    job.model = provider.get("model", "")
                    job.provider = provider.get("provider", "")

                def run_one(text: str) -> dict[str, Any]:
                    return llm.clean_with_llm(text, provider, known=model.known)

                pool = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {
                        pool.submit(run_one, entry["text"]): entry
                        for entry in pending.values()
                    }
                    remaining = set(futures)
                    while remaining:
                        # Wake up often enough that Stop feels immediate: a
                        # model call takes seconds, so this is a poll cap.
                        done, remaining = wait(
                            remaining, timeout=0.4, return_when=FIRST_COMPLETED
                        )
                        if job.cancel.is_set():
                            break
                        for future in done:
                            entry = futures[future]
                            rows = entry["rows"]
                            try:
                                result = future.result()
                            except Exception as exc:  # noqa: BLE001
                                log.warning("preclean model failure: %s", exc)
                                result = {"accepted": False, "reason": "The model call failed."}

                            accepted = bool(result.get("accepted"))
                            if accepted:
                                if result.get("boundary_reverted") and not job.warning:
                                    job.warning = (
                                        "The model tried to change word spacing, so the "
                                        "rule-engine boundaries were kept and only its "
                                        "spelling fixes were applied."
                                    )
                                for row in rows:
                                    base = rule_results[row]
                                    if not preserves(base["fixes"], result["cleaned"]):
                                        # The model undid a boundary repair; the rules win.
                                        accepted = False
                                        break

                            if accepted:
                                for row in rows:
                                    base = rule_results[row]
                                    merged = {
                                        "original": base["original"],
                                        "cleaned": result["cleaned"],
                                        "fixes": (base["fixes"] + result.get("fixes", []))[:20],
                                        "changed": result["cleaned"] != normalize(base["original"]),
                                        "engine": "llm",
                                    }
                                    self._store(session, row, merged, engine="llm")
                                with job.lock:
                                    job.llm_done += 1
                                    job.reused += max(len(rows) - 1, 0)
                            else:
                                with job.lock:
                                    job.failed += 1
                                    job.llm_done += 1
                                if not job.warning and result.get("reason"):
                                    job.warning = str(result["reason"])
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            else:
                # Rules-only pass: write the analysis for every row.
                for row, analysis in rule_results.items():
                    self._store(session, row, analysis, engine="rules")

            if job.cancel.is_set():
                self._finish(job, "cancelled")
            else:
                self._finish(job, "done")

        except Exception as exc:  # noqa: BLE001
            log.exception("preclean job failed")
            with job.lock:
                job.error = f"{type(exc).__name__}: {exc}"[:300]
            self._finish(job, "error")
        finally:
            try:
                session.persist()
            except Exception:  # noqa: BLE001
                pass

    def _store(self, session, row: int, data: dict[str, Any], engine: str) -> None:
        payload = dict(data)
        payload["engine"] = engine
        session.remember(row, payload)

    def _finish(self, job: Job, status: str) -> None:
        with job.lock:
            job.status = status
            job.finished_at = time.time()


def _needs_model(original: str, analysis: dict[str, Any], model: BoundaryModel) -> bool:
    """Cheap gate: does this row deserve a model call?

    A row whose rules pass found nothing is only sent to the model when it still
    contains a token that looks misspelled — an unknown Latin word, or a Tamil
    token the dataset has never seen. This is what makes a mostly-clean column
    nearly free to pre-clean.
    """
    if analysis.get("fixes"):
        return True
    for token in normalize(original).split():
        core, _ = _split_punct(token)
        core = core.strip()
        if len(core) < 3:
            continue
        if core in model.vocab:
            continue
        if all(ch.isascii() and ch.isalpha() for ch in core):
            if core.lower() in model.english:
                continue
            return True
        if any("\u0b80" <= ch <= "\u0bff" for ch in core):
            return True
    return False


batches = BatchManager()
