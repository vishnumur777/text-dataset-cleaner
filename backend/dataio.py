"""Dataset reading/writing helpers.

Only formats that pandas (or a thin optional shim) can round-trip are accepted.
Every failure is converted into a `DataError` with a message written for humans.

`.txt` is a first-class dataset format here: one non-empty line is one row.
Exporting that dataset writes the cleaned text back one line per row, and the
`zip` export bundles that text file with a manifest (and, for tabular sources,
a copy of the cleaned table in its original format).
"""
from __future__ import annotations

import json
import csv
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

MAX_ROWS = int(os.environ.get("TDC_MAX_ROWS", "200000"))

READABLE = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".avro", ".xlsx", ".xls", ".txt"}
WRITABLE = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".avro", ".xlsx", ".txt", ".zip"}
# Formats that can be written inside the zip bundle (zip itself cannot nest).
BUNDLE_INNER = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".avro", ".xlsx", ".txt"}

TEXT_FILE_COLUMN = "text"
TOOL = "text-dataset-cleaner"

# Words that usually mean "this column holds the long text we want to clean".
TEXT_HINTS = (
    "text", "sentence", "content", "body", "review", "comment", "description",
    "transcript", "document", "paragraph", "prompt", "output", "input", "tamil",
    "ta", "title", "lyrics", "article", "tweet", "caption",
)


class DataError(Exception):
    """User-facing data problem."""

    def __init__(self, message: str, hint: str = "", code: str = "dataset_error"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


@dataclass
class Loaded:
    df: pd.DataFrame
    fmt: str
    truncated: bool = False
    total_rows: int = 0


def _read_csv(path: str, sep: str | None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python" if sep is None else "c",
        on_bad_lines="skip",
    )


def read_text_file(path: str) -> str:
    """Decode a text file, preferring UTF-8 and falling back to a lossy read."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# Delimiters a CSV may plausibly use. Sniffing is restricted to these because
# pandas' own sniffer will happily pick a letter: a single-column CSV with no
# commas was read with "t" as the separator, turning one "text" column into
# "Unnamed: 0", "ex", "Unnamed: 2".
CSV_DELIMITERS = (",", ";", "\t", "|")


def _sniff_delimiter(path: str) -> str:
    sample = read_text_file(path)[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(CSV_DELIMITERS)).delimiter
    except Exception:
        return ","


def read_table(path: str) -> Loaded:
    """Read a dataset file into a DataFrame of strings."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in READABLE:
        raise DataError(
            f"'{ext or 'unknown'}' files are not supported for import.",
            f"Supported: {', '.join(sorted(READABLE))}.",
            "unsupported_format",
        )

    try:
        if ext == ".csv":
            sep = _sniff_delimiter(path)
            try:
                df = _read_csv(path, sep)
            except Exception:
                df = _read_csv(path, ",")
            if df.shape[1] == 1 and sep != ",":
                df = _read_csv(path, ",")
        elif ext == ".tsv":
            df = _read_csv(path, "\t")
        elif ext == ".txt":
            lines = [line for line in read_text_file(path).splitlines() if line.strip()]
            if not lines:
                raise DataError(
                    "That text file has no readable lines.",
                    "A .txt dataset is one sample per line; check that the file is not empty.",
                    "empty_text_file",
                )
            df = pd.DataFrame({TEXT_FILE_COLUMN: lines})
        elif ext in (".json",):
            try:
                df = pd.read_json(path, dtype=str)
                if isinstance(df, pd.Series):
                    df = df.to_frame()
            except Exception:
                df = pd.read_json(path, lines=True, dtype=str)
        elif ext in (".jsonl", ".ndjson"):
            df = pd.read_json(path, lines=True, dtype=str)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".avro":
            try:
                import fastavro  # type: ignore
            except ImportError:
                raise DataError(
                    "Avro support is not installed on this server.",
                    "Install it with: pip install fastavro",
                    "missing_dependency",
                )
            with open(path, "rb") as fh:
                records = list(fastavro.reader(fh))
            df = pd.DataFrame.from_records(records)
        elif ext in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(path, dtype=str)
            except ImportError:
                raise DataError(
                    "Excel support is not installed on this server.",
                    "Install it with: pip install openpyxl",
                    "missing_dependency",
                )
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"We could not parse this {ext} file.",
            f"The file may be corrupt or use an unusual encoding ({type(exc).__name__}).",
            "parse_error",
        ) from exc

    if df is None or len(df.columns) == 0:
        raise DataError("The file has no columns.", "Check that it is a valid dataset file.")

    df = df.fillna("")
    df = df.astype(str)

    truncated = False
    total = len(df)
    if total > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()
        truncated = True

    df = df.reset_index(drop=True)
    return Loaded(df=df, fmt=ext.lstrip("."), truncated=truncated, total_rows=total)


def guess_text_column(columns: Iterable[str], df: pd.DataFrame | None = None) -> str | None:
    """Pick the column that most likely holds the text to clean.

    The hint list only matches English header words ("text", "content", ...), so
    a dataset whose headers are in another language (a Tamil news corpus, say)
    falls through. When that happens we fall back to shape rather than names:
    the free-text column is almost always the longest one by a wide margin.
    """
    cols = list(columns)
    lowered = {c: c.lower() for c in cols}
    for hint in TEXT_HINTS:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in TEXT_HINTS:
        for col, low in lowered.items():
            if hint in low:
                return col
    return _guess_by_length(cols, df)


def _guess_by_length(cols: list[str], df: pd.DataFrame | None) -> str | None:
    """The longest average cell wins, but only when it clearly stands out."""
    if df is None or len(cols) < 2:
        return None
    means: list[tuple[float, str]] = []
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        if series.empty:
            continue
        means.append((float(series.str.len().mean()), col))
    if not means:
        return None
    means.sort(reverse=True)
    best_len, best_col = means[0]
    runner_up = means[1][0] if len(means) > 1 else 0.0
    # A real text column is long in absolute terms and much longer than the
    # metadata beside it; anything less is a guess we should not make.
    if best_len >= 120 and best_len >= max(2.0 * runner_up, 40.0):
        return best_col
    return None


def _text_series(df: pd.DataFrame, text_column: str | None) -> pd.Series:
    if text_column and text_column in df.columns:
        return df[text_column]
    if TEXT_FILE_COLUMN in df.columns:
        return df[TEXT_FILE_COLUMN]
    if len(df.columns) == 1:
        return df.iloc[:, 0]
    raise DataError(
        "This dataset has more than one column, so it cannot be written as plain text.",
        "Pick a text column, or export as CSV/JSON instead.",
        "no_text_column",
    )


def _text_payload(df: pd.DataFrame, text_column: str | None) -> str:
    series = _text_series(df, text_column)
    # A text dataset is line-oriented, so a newline inside a cell would silently
    # split one row into two on the next import. Flatten it instead.
    values = [
        str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        for value in series
    ]
    return "\n".join(values) + ("\n" if values else "")


def _manifest(
    df: pd.DataFrame,
    *,
    text_column: str | None,
    source_format: str | None,
    files: Sequence[str],
    meta: dict[str, Any] | None,
) -> dict[str, Any]:
    info = dict(meta or {})
    return {
        "tool": TOOL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "text_column": text_column,
        "source_format": source_format,
        "files": list(files),
        "source": info.get("source", {}),
        "saved_rows": info.get("saved"),
        "changed_rows": info.get("changed"),
        "precleaned_rows": info.get("precleaned"),
        "note": (
            "One line per row in the .txt file. Newlines inside a cell are flattened "
            "to spaces so the file can be imported again without changing the row count."
        ),
    }


def _write_zip(
    df: pd.DataFrame,
    path: str,
    *,
    text_column: str | None,
    stem: str,
    source_format: str | None,
    meta: dict[str, Any] | None,
) -> None:
    txt_name = f"{stem}.cleaned.txt"
    inner: str | None = None
    inner_name = ""
    with tempfile.TemporaryDirectory(prefix="tdc-zip-") as tmp_dir:
        tmp = Path(tmp_dir) / f"{stem}.cleaned.{source_format}"
        if (
            source_format
            and f".{source_format.lower()}" in BUNDLE_INNER
            and source_format.lower() != "txt"
        ):
            write_table(
                df,
                source_format,
                str(tmp),
                text_column=text_column,
                stem=stem,
                source_format=source_format,
            )
            inner = str(tmp)
            inner_name = tmp.name

        files = [txt_name, "manifest.json"] + ([inner_name] if inner else [])
        manifest = _manifest(
            df,
            text_column=text_column,
            source_format=source_format,
            files=files,
            meta=meta,
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(txt_name, _text_payload(df, text_column))
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            if inner:
                archive.write(inner, inner_name)


def write_table(
    df: pd.DataFrame,
    fmt: str,
    path: str,
    *,
    text_column: str | None = None,
    stem: str | None = None,
    source_format: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    fmt = fmt.lower().lstrip(".")
    if fmt not in {f.lstrip(".") for f in WRITABLE}:
        raise DataError(
            f"'{fmt}' is not a supported export format.",
            f"Supported: {', '.join(sorted(f.lstrip('.') for f in WRITABLE))}.",
            "unsupported_format",
        )
    stem = stem or Path(path).stem or "dataset"
    try:
        if fmt == "csv":
            df.to_csv(path, index=False, encoding="utf-8")
        elif fmt == "tsv":
            df.to_csv(path, index=False, sep="\t", encoding="utf-8")
        elif fmt == "json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        elif fmt == "jsonl":
            df.to_json(path, orient="records", force_ascii=False, lines=True)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt in ("xlsx", "xls"):
            df.to_excel(path, index=False)
        elif fmt == "txt":
            Path(path).write_text(_text_payload(df, text_column), encoding="utf-8")
        elif fmt == "zip":
            _write_zip(
                df,
                path,
                text_column=text_column,
                stem=stem,
                source_format=source_format,
                meta=meta,
            )
        elif fmt == "avro":
            import fastavro  # type: ignore

            records = json.loads(df.to_json(orient="records", force_ascii=False))
            for rec in records:
                for k, v in list(rec.items()):
                    if v is None:
                        rec[k] = ""
            cleaned = [
                {str(k): "" if v is None else (v if isinstance(v, (str, int, float, bool)) else str(v))
                 for k, v in rec.items()}
                for rec in records
            ]
            schema = {
                "type": "record",
                "name": "row",
                "fields": [{"name": str(c), "type": ["null", "string"]} for c in df.columns],
            }
            with open(path, "wb") as fh:
                fastavro.writer(fh, fastavro.parse_schema(schema), cleaned)
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"Export to {fmt} failed.",
            f"{type(exc).__name__}: {exc}"[:300],
            "export_error",
        ) from exc


def sniff_bytes(name: str) -> str:
    return os.path.splitext(name)[1].lower().lstrip(".")
