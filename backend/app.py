"""FastAPI application for Text Dataset Cleaner.

Design notes
------------
* One origin in production: the built React app is served by this same server,
  so there is no cross-origin request and therefore no CORS failure to debug.
  During `npm run dev`, Vite proxies /api to this server, which avoids CORS too.
* Every failure becomes `{error: {code, message, hint}}` with text a normal
  person can act on; tracebacks stay in the server log.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, sources
from .batch import batches
from .cleaner import BoundaryModel, preserves
from .dataio import DataError, WRITABLE, guess_text_column, read_table, sniff_bytes, write_table
from .jobs import jobs
from .sources import SourceError
from .store import SessionError, store

log = logging.getLogger("tdc")
logging.basicConfig(level=os.environ.get("TDC_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

APP_VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
EXPORT_DIR = Path(".exports").resolve()
EXPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Text Dataset Cleaner", version=APP_VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get(
            "TDC_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if o.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- errors


def _error(status: int, code: str, message: str, hint: str = "") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "hint": hint}})


@app.exception_handler(SourceError)
async def _source_error(_: Request, exc: SourceError):
    return _error(400, exc.code, exc.message, exc.hint)


@app.exception_handler(DataError)
async def _data_error(_: Request, exc: DataError):
    return _error(400, exc.code, exc.message, exc.hint)


@app.exception_handler(SessionError)
async def _session_error(_: Request, exc: SessionError):
    return _error(404, exc.code, exc.message, exc.hint)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.error("unhandled error: %s\n%s", exc, traceback.format_exc())
    return _error(
        500,
        "internal_error",
        "Something went wrong on the server while handling that request.",
        "Try again; if it keeps happening, restart the server and check its log.",
    )


# ---------------------------------------------------------------------- models


class HfOpen(BaseModel):
    repo: str
    path: str
    revision: str = "main"


class KaggleOpen(BaseModel):
    ref: str
    path: str


class ConnectKaggle(BaseModel):
    username: str
    key: str


class ColumnChoice(BaseModel):
    column: str


class CleanRequest(BaseModel):
    index: int = Field(ge=0)
    mode: str = "rules"  # rules | llm


class SaveRequest(BaseModel):
    index: int = Field(ge=0)
    text: str


class ExportRequest(BaseModel):
    format: str | None = None


class PrecleanRequest(BaseModel):
    mode: str = "llm"          # llm | rules
    workers: int | None = None  # model calls in flight (llama.cpp slots)


class PublishRequest(BaseModel):
    target: str                        # huggingface | kaggle
    ref: str | None = None             # repo / dataset to write to
    format: str | None = None          # csv, parquet, zip, ...
    message: str | None = None


# --------------------------------------------------------------------- helpers


def _session_from(loaded, source: dict[str, Any]) -> dict[str, Any]:
    """Create a session from a parsed table and return the file-picker payload."""
    col = guess_text_column(loaded.df.columns, loaded.df)
    session = store.create(
        loaded.df,
        loaded.fmt,
        source,
        truncated=loaded.truncated,
        total_rows=loaded.total_rows,
    )
    session.text_column = col
    session.persist()
    return {"session": session.summary(), "suggested_column": col, "format": loaded.fmt}


def _open_hf(body: "HfOpen", job=None) -> dict[str, Any]:
    name = Path(body.path).name
    if job:
        job.update(phase="download", message=f"Downloading {name}…", done=0, total=0)
    local = sources.hf_download(
        body.repo,
        body.path,
        body.revision,
        on_progress=job.progress if job else None,
    )
    if job:
        job.update(phase="parse", message=f"Reading {name}…")
    loaded = read_table(local)
    return _session_from(
        loaded,
        {"kind": "huggingface", "label": body.repo, "file": body.path, "repo": body.repo},
    )


def _open_kaggle(body: "KaggleOpen", job=None) -> dict[str, Any]:
    name = Path(body.path).name
    if job:
        job.update(phase="download", message=f"Downloading {name}…", done=0, total=0)
    local = sources.kaggle_download(body.ref, body.path, on_progress=job.progress if job else None)
    if job:
        job.update(phase="parse", message=f"Reading {name}…")
    loaded = read_table(local)
    return _session_from(
        loaded,
        {"kind": "kaggle", "label": body.ref, "file": body.path, "repo": body.ref},
    )


# ---------------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> dict[str, Any]:
    provider = llm.detect()
    return {
        "ok": True,
        "version": APP_VERSION,
        "llm": {
            "available": provider.get("ok", False),
            "provider": provider.get("provider"),
            "model": provider.get("model"),
        },
        "sources": sources.sources_status(),
    }


@app.get("/api/hf/search")
def hf_search(q: str, limit: int = 20):
    return {"results": sources.hf_search(q, min(max(limit, 1), 50))}


@app.get("/api/hf/files")
def hf_files(repo: str, revision: str = "main"):
    return {"repo": repo, "files": sources.hf_files(repo, revision)}


@app.post("/api/hf/open")
def hf_open(body: HfOpen):
    return _open_hf(body)


@app.post("/api/hf/open-job")
async def hf_open_job(body: HfOpen):
    """Start a download+parse job; poll GET /api/jobs/{id} for progress."""
    return {"job": jobs.start("huggingface", lambda job: _open_hf(body, job))}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    snapshot = jobs.get(job_id)
    if snapshot is None:
        return _error(404, "job_missing", "That loading job is no longer available.", "Start the download again.")
    return snapshot


@app.get("/api/kaggle/search")
def kaggle_search(q: str, limit: int = 20):
    return {"results": sources.kaggle_search(q, min(max(limit, 1), 50))}


@app.get("/api/kaggle/files")
def kaggle_files(ref: str):
    return {"ref": ref, "files": sources.kaggle_files(ref)}


@app.post("/api/kaggle/connect")
def kaggle_connect(body: ConnectKaggle):
    return sources.kaggle_connect(body.username, body.key)


@app.post("/api/kaggle/open")
def kaggle_open(body: KaggleOpen):
    return _open_kaggle(body)


@app.post("/api/kaggle/open-job")
async def kaggle_open_job(body: KaggleOpen):
    return {"job": jobs.start("kaggle", lambda job: _open_kaggle(body, job))}


SAMPLES = ROOT / "samples"


@app.get("/api/samples")
def list_samples():
    out = []
    for path in sorted(SAMPLES.glob("*")):
        if path.is_file():
            out.append({"path": path.name, "size": path.stat().st_size})
    return {"samples": out}


@app.post("/api/sample/open")
def open_sample(name: str):
    path = SAMPLES / Path(name).name
    if not path.exists():
        raise DataError("That sample no longer exists.", "Open the file picker and upload your own file.", "missing_sample")
    loaded = read_table(str(path))
    col = guess_text_column(loaded.df.columns, loaded.df)
    session = store.create(
        loaded.df,
        loaded.fmt,
        {"kind": "sample", "label": f"sample/{path.name}", "file": path.name},
        truncated=loaded.truncated,
        total_rows=loaded.total_rows,
    )
    session.text_column = col
    session.persist()
    return {"session": session.summary(), "suggested_column": col, "format": loaded.fmt}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise DataError("No file was received.", "Pick a dataset file and try again.", "no_file")
    suffix = Path(file.filename).suffix.lower()
    if not suffix:
        raise DataError("That file has no extension.", "Rename it with .csv, .json or .parquet.", "unknown_type")

    tmp_dir = Path(tempfile.mkdtemp(prefix="tdc-upload-"))
    dest = tmp_dir / f"upload{suffix}"
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    if dest.stat().st_size == 0:
        raise DataError("The uploaded file is empty.", "Check the file on disk and upload it again.", "empty_file")

    loaded = read_table(str(dest))
    col = guess_text_column(loaded.df.columns, loaded.df)
    session = store.create(
        loaded.df,
        loaded.fmt or sniff_bytes(file.filename),
        {"kind": "upload", "label": file.filename, "file": file.filename},
        truncated=loaded.truncated,
        total_rows=loaded.total_rows,
    )
    session.text_column = col
    session.persist()
    return {"session": session.summary(), "suggested_column": col, "format": loaded.fmt}


@app.get("/api/session/{session_id}")
def session_info(session_id: str):
    return store.get(session_id).summary()


@app.post("/api/session/{session_id}/column")
def set_column(session_id: str, body: ColumnChoice):
    session = store.get(session_id)
    if body.column not in session.df.columns:
        raise DataError(
            "That column is not in this dataset.",
            "Reload the file and pick a column from the list.",
            "unknown_column",
        )
    session.text_column = body.column
    session.suggestions.clear()
    session.persist()
    return session.summary()


def _row_payload(session, index: int, model: BoundaryModel | None = None) -> dict[str, Any]:
    original = session.original(index)
    suggestion = session.suggest(index, model)
    saved = session.edits.get(index)
    return {
        "index": index,
        "total": int(len(session.df)),
        "column": session.text_column,
        "original": original,
        "suggested": suggestion["cleaned"],
        "fixes": suggestion["fixes"],
        "engine": suggestion.get("engine", "rules"),
        "precleaned": bool(suggestion.get("engine")),
        "saved": saved,
        "edited": saved is not None and saved != original,
    }


@app.get("/api/session/{session_id}/row")
def get_row(session_id: str, index: int = 0):
    session = store.get(session_id)
    return _row_payload(session, max(0, min(index, len(session.df) - 1)))


@app.get("/api/session/{session_id}/rows")
def get_rows(session_id: str, start: int = 0, limit: int = 20):
    session = store.get(session_id)
    limit = max(1, min(limit, 100))
    model = session.model()
    rows = [
        _row_payload(session, i, model)
        for i in range(max(0, start), min(len(session.df), start + limit))
    ]
    return {"rows": rows, "start": start, "total": int(len(session.df))}


@app.post("/api/session/{session_id}/clean")
def clean_row(session_id: str, body: CleanRequest):
    session = store.get(session_id)
    original = session.original(body.index)
    suggestion = session.suggest(body.index)

    if body.mode != "llm":
        return {
            "engine": "rules",
            "index": body.index,
            "original": original,
            "cleaned": suggestion["cleaned"],
            "fixes": suggestion["fixes"],
            "warning": "",
            "elapsed_s": 0,
        }

    provider = llm.detect()
    if not provider.get("ok"):
        return {
            "engine": "rules",
            "index": body.index,
            "original": original,
            "cleaned": suggestion["cleaned"],
            "fixes": suggestion["fixes"],
            "warning": "No local model is running, so the rule engine handled this row. Start llama-server or Ollama to use Gemma.",
            "elapsed_s": 0,
        }

    try:
        # Give the model the already rule-cleaned text: the deterministic layer
        # handles boundaries, the model only polishes spelling. `known` lets a
        # lexically-confirmed join through the boundary guard.
        result = llm.clean_with_llm(suggestion["cleaned"], provider, known=session.model().known)
    except RuntimeError as exc:
        return {
            "engine": "rules",
            "index": body.index,
            "original": original,
            "cleaned": suggestion["cleaned"],
            "fixes": suggestion["fixes"],
            "warning": str(exc),
            "elapsed_s": 0,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("llm failure: %s", exc)
        return {
            "engine": "rules",
            "index": body.index,
            "original": original,
            "cleaned": suggestion["cleaned"],
            "fixes": suggestion["fixes"],
            "warning": "The model call failed, so the rule engine handled this row.",
            "elapsed_s": 0,
        }

    accepted = bool(result["accepted"]) and preserves(suggestion["fixes"], result["cleaned"])
    reason = ""
    if not accepted:
        reason = result.get("reason") or (
            "The model undid a spacing fix, so this row kept the rule-engine result."
        )
    elif result.get("boundary_reverted"):
        reason = (
            "The model changed word spacing, so the rule-engine boundaries were kept "
            "and only its spelling fixes were applied."
        )
    cleaned = result["cleaned"] if accepted else suggestion["cleaned"]
    fixes = suggestion["fixes"] + (result["fixes"] if accepted else [])
    return {
        "engine": "llm" if accepted else "rules",
        "index": body.index,
        "original": original,
        "cleaned": cleaned,
        "fixes": fixes[:20],
        "warning": reason,
        "model": result.get("model", ""),
        "elapsed_s": result.get("elapsed_s", 0),
    }


@app.post("/api/session/{session_id}/preclean")
def preclean(session_id: str, body: PrecleanRequest):
    """Clean every row in one pass, then serve each row from the cache.

    Returns immediately with the job state; poll GET .../preclean for progress.
    """
    session = store.get(session_id)
    if session.text_column is None:
        raise SessionError(
            "Choose a text column before cleaning the column.",
            "Reload the dataset from the file picker and pick the column.",
            "no_column",
        )
    return batches.start(session, body.mode, body.workers)


@app.get("/api/session/{session_id}/preclean")
def preclean_status(session_id: str):
    store.get(session_id)
    return batches.status(session_id)


@app.post("/api/session/{session_id}/preclean/cancel")
def preclean_cancel(session_id: str):
    store.get(session_id)
    return batches.cancel(session_id)


@app.post("/api/session/{session_id}/save")
def save_row(session_id: str, body: SaveRequest):
    session = store.get(session_id)
    session.save(body.index, body.text)
    return {
        "ok": True,
        "index": body.index,
        "saved": len(session.edits),
        "next": min(body.index + 1, len(session.df) - 1),
        "summary": session.summary(),
    }


@app.post("/api/session/{session_id}/export")
def export(session_id: str, body: ExportRequest):
    session = store.get(session_id)
    fmt = (body.format or session.fmt or "csv").lstrip(".").lower()
    df = session.output_frame()
    stem = Path(str(session.source.get("file", "dataset"))).stem or "dataset"
    out = EXPORT_DIR / f"{stem}.cleaned.{fmt}"
    write_table(
        df,
        fmt,
        str(out),
        text_column=session.text_column,
        stem=stem,
        source_format=session.fmt,
        meta=session.summary(),
    )
    return FileResponse(
        path=str(out),
        filename=out.name,
        media_type="application/octet-stream",
        headers={"X-Edited-Rows": str(len(session.edits))},
    )


@app.post("/api/session/{session_id}/publish")
def publish(session_id: str, body: PublishRequest):
    """Write the cleaned rows back to Hugging Face or a Kaggle companion dataset."""
    session = store.get(session_id)
    if (session.fmt or "").lower() == "txt":
        raise SourceError(
            "Text-file datasets are published by hand.",
            "Download the zip export and upload it yourself; a .txt source has no repository to write back to.",
            "manual_publish",
        )

    target = (body.target or "").strip().lower()
    if target not in {"huggingface", "kaggle"}:
        raise SourceError(
            "Choose where to publish first.",
            "Publishing is available for Hugging Face and Kaggle datasets.",
            "bad_target",
        )

    fmt = (body.format or session.fmt or "csv").lstrip(".").lower()
    if f".{fmt}" not in WRITABLE:
        raise DataError(
            f"'{fmt}' cannot be published.",
            "Pick one of the download formats first.",
            "unsupported_format",
        )

    df = session.output_frame()
    stem = Path(str(session.source.get("file", "dataset"))).stem or "dataset"
    filename = f"{stem}.cleaned.{fmt}"
    message = (body.message or "Cleaned with text-dataset-cleaner").strip()[:240]

    with tempfile.TemporaryDirectory(prefix="tdc-publish-") as tmp_dir:
        local = Path(tmp_dir) / filename
        write_table(
            df,
            fmt,
            str(local),
            text_column=session.text_column,
            stem=stem,
            source_format=session.fmt,
            meta=session.summary(),
        )
        if target == "huggingface":
            repo = (body.ref or session.source.get("repo") or "").strip()
            if not repo:
                raise SourceError(
                    "This dataset did not come from a Hugging Face repository.",
                    "Use Download, then upload the file with your own tooling.",
                    "no_repo",
                )
            result = sources.hf_upload(
                repo,
                f"cleaned_datasets/{filename}",
                str(local),
                message=message,
            )
        else:
            source_ref = str(session.source.get("repo") or "")
            ref = (body.ref or "").strip()
            if not ref:
                if not source_ref:
                    raise SourceError(
                        "This dataset did not come from a Kaggle dataset.",
                        "Use Download, then upload the file with your own tooling.",
                        "no_repo",
                    )
                ref = sources.kaggle_companion_ref(source_ref)
            result = sources.kaggle_publish(ref, str(local), message=message)

    return {"ok": True, **result}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    session = store.get(session_id)
    for child in session.dir.glob("*"):
        child.unlink(missing_ok=True)
    try:
        session.dir.rmdir()
    except OSError:
        pass
    return {"ok": True}


# --------------------------------------------------------------- static app

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return _error(404, "not_found", "That API route does not exist.", "")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=os.environ.get("TDC_HOST", "127.0.0.1"),
        port=int(os.environ.get("TDC_PORT", "8000")),
        reload=bool(os.environ.get("TDC_RELOAD")),
    )


if __name__ == "__main__":
    main()
