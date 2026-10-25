"""Session store: one loaded dataset, its cleaned output, and its edit log.

Kept in memory for speed and mirrored to disk so a server restart does not lose
a user's work (the UI shows a clear message if a session really is gone).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaner import BoundaryModel, analyze

SESSIONS_DIR = Path(".sessions").resolve()
MAX_SESSIONS = 12
SESSION_TTL_HOURS = 48


class SessionError(Exception):
    def __init__(self, message: str, hint: str = "", code: str = "session_error"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


def _load_suggestions(session: "Session", path: Path) -> None:
    """Replay the pre-clean log; the last write for a row wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            session.suggestions[int(item["row"])] = item["suggestion"]
        except Exception:
            continue


@dataclass
class Session:
    id: str
    df: pd.DataFrame
    fmt: str
    source: dict[str, Any]
    text_column: str | None = None
    truncated: bool = False
    total_rows: int = 0
    edits: dict[int, str] = field(default_factory=dict)
    suggestions: dict[int, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)
    _model_cache: tuple[str | None, "BoundaryModel"] | None = None

    # ---------------------------------------------------------------- helpers
    @property
    def dir(self) -> Path:
        return SESSIONS_DIR / self.id

    def model(self) -> BoundaryModel:
        # The lexicon is learned from the original column, which never changes
        # while a session is open, so one model is built and reused. Rebuilding
        # it on every row (the clean route asks for it) was pure waste.
        with self.lock:
            cached = self._model_cache
            if cached is not None and cached[0] == self.text_column:
                return cached[1]
            column = self.text_column
            texts = self.df[column].tolist() if column else []
        model = BoundaryModel(texts)
        with self.lock:
            self._model_cache = (column, model)
        return model

    def original(self, row: int) -> str:
        if self.text_column is None:
            raise SessionError("Pick a text column first.", "Go back to file selection and choose the column to clean.", "no_column")
        if row < 0 or row >= len(self.df):
            raise SessionError("That row does not exist.", f"There are {len(self.df)} rows.", "row_out_of_range")
        return str(self.df.iloc[row][self.text_column])

    def suggest(self, row: int, model: BoundaryModel | None = None) -> dict[str, Any]:
        with self.lock:
            if row in self.suggestions:
                return self.suggestions[row]
        result = analyze(self.original(row), model or self.model())
        with self.lock:
            self.suggestions[row] = result
        return result

    def save(self, row: int, text: str) -> None:
        self.original(row)  # validates
        with self.lock:
            self.edits[row] = text
        self._append_edit(row, text)

    def remember(self, row: int, suggestion: dict[str, Any]) -> None:
        """Store a pre-computed suggestion (from the rules pass or the model)."""
        with self.lock:
            self.suggestions[row] = suggestion
        self._append_suggestion(row, suggestion)

    def precleaned(self) -> int:
        with self.lock:
            return sum(1 for s in self.suggestions.values() if s.get("engine"))

    def output_frame(self) -> pd.DataFrame:
        with self.lock:
            df = self.df.copy()
            if self.text_column is None:
                return df
            for row, text in self.edits.items():
                if 0 <= row < len(df):
                    df.iat[row, df.columns.get_loc(self.text_column)] = text
            return df

    def summary(self) -> dict[str, Any]:
        with self.lock:
            edits = dict(self.edits)
        changed = sum(
            1
            for row, text in edits.items()
            if self.text_column and text != str(self.df.iloc[row][self.text_column])
        )
        return {
            "id": self.id,
            "source": self.source,
            "format": self.fmt,
            "rows": int(len(self.df)),
            "total_rows": self.total_rows or int(len(self.df)),
            "truncated": self.truncated,
            "columns": [str(c) for c in self.df.columns],
            "text_column": self.text_column,
            "saved": len(edits),
            "changed": changed,
            "precleaned": self.precleaned(),
        }

    # ------------------------------------------------------------- durability
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.df.to_parquet(self.dir / "original.parquet", index=False)
            (self.dir / "meta.json").write_text(
                json.dumps(
                    {
                        "id": self.id,
                        "fmt": self.fmt,
                        "source": self.source,
                        "text_column": self.text_column,
                        "truncated": self.truncated,
                        "total_rows": self.total_rows,
                        "created_at": self.created_at,
                    },
                    ensure_ascii=False,
                )
            )

    def _append_edit(self, row: int, text: str) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "edits.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"row": row, "text": text}, ensure_ascii=False) + "\n")
        except Exception:
            pass  # durability is best-effort; memory is authoritative

    def _append_suggestion(self, row: int, suggestion: dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "suggestions.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"row": row, "suggestion": suggestion}, ensure_ascii=False) + "\n")
        except Exception:
            pass


class Store:
    def __init__(self) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.RLock()
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        for meta in sorted(SESSIONS_DIR.glob("*/meta.json")):
            try:
                data = json.loads(meta.read_text())
                parquet = meta.parent / "original.parquet"
                if not parquet.exists():
                    continue
                df = pd.read_parquet(parquet)
                session = Session(
                    id=data["id"],
                    df=df,
                    fmt=data.get("fmt", "csv"),
                    source=data.get("source", {}),
                    text_column=data.get("text_column"),
                    truncated=data.get("truncated", False),
                    total_rows=data.get("total_rows", len(df)),
                    created_at=data.get("created_at", time.time()),
                )
                edits = meta.parent / "edits.jsonl"
                if edits.exists():
                    for line in edits.read_text(encoding="utf-8").splitlines():
                        try:
                            item = json.loads(line)
                            session.edits[int(item["row"])] = item["text"]
                        except Exception:
                            continue
                _load_suggestions(session, meta.parent / "suggestions.jsonl")
                self._sessions[session.id] = session
            except Exception:
                continue
        self._trim()

    def create(self, df: pd.DataFrame, fmt: str, source: dict[str, Any], truncated: bool = False, total_rows: int = 0) -> Session:
        session = Session(
            id=uuid.uuid4().hex[:12],
            df=df,
            fmt=fmt,
            source=source,
            truncated=truncated,
            total_rows=total_rows or len(df),
        )
        with self._lock:
            self._sessions[session.id] = session
            self._trim()
        session.persist()
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._load_from_disk(session_id)
                if session is None:
                    raise SessionError(
                        "That editing session has expired.",
                        "Load the dataset again from the file picker; saved rows are kept on disk.",
                        "session_missing",
                    )
                # A session evicted from memory is still on disk: put it back.
                self._sessions[session_id] = session
                self._trim()
            self._sessions.move_to_end(session_id)
            return session

    def _trim(self) -> None:
        with self._lock:
            while len(self._sessions) > MAX_SESSIONS:
                _, old = self._sessions.popitem(last=False)
                try:
                    old.persist()
                except Exception:
                    pass

    def _load_from_disk(self, session_id: str) -> Session | None:
        meta = SESSIONS_DIR / session_id / "meta.json"
        parquet = SESSIONS_DIR / session_id / "original.parquet"
        if not (meta.exists() and parquet.exists()):
            return None
        try:
            data = json.loads(meta.read_text())
            session = Session(
                id=session_id,
                df=pd.read_parquet(parquet),
                fmt=data.get("fmt", "csv"),
                source=data.get("source", {}),
                text_column=data.get("text_column"),
                truncated=data.get("truncated", False),
                total_rows=data.get("total_rows", 0),
            )
            edits = meta.parent / "edits.jsonl"
            if edits.exists():
                for line in edits.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                        session.edits[int(item["row"])] = item["text"]
                    except Exception:
                        continue
            _load_suggestions(session, meta.parent / "suggestions.jsonl")
            return session
        except Exception:
            return None

    def purge_old(self) -> int:
        cutoff = time.time() - SESSION_TTL_HOURS * 3600
        removed = 0
        for path in SESSIONS_DIR.glob("*/meta.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("created_at", 0) < cutoff:
                    for child in path.parent.iterdir():
                        child.unlink(missing_ok=True)
                    path.parent.rmdir()
                    removed += 1
            except Exception:
                continue
        return removed


store = Store()
