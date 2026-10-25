#!/usr/bin/env python3
"""Smoke test for the offline core, plus the live API when it is running.

    .venv/bin/python scripts/smoke_test.py          # core only
    .venv/bin/python scripts/smoke_test.py --api    # core + running server

Exits non-zero on the first failure so it can gate a hand-off or a CI job.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.cleaner import BoundaryModel, analyze, preserves  # noqa: E402
from backend.dataio import guess_text_column, read_table, write_table  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{f' — {detail}' if detail else ''}")
        FAILURES.append(name)


def check_repairs() -> None:
    print("boundary repair")
    sample = Path("samples/messy_tamil.csv")
    if not sample.exists():
        print("  skip (run scripts/make_sample.py first)")
        return
    df = pd.read_csv(sample, dtype=str, keep_default_na=False)
    model = BoundaryModel(df["text"])

    cases = {
        "down loa ded the dataset.": "downloaded the dataset.",
        "homescreenwallpaper. We test it today.": "home screen wallpaper. We test it today.",
        "Spellingcorrection tool.": "Spelling correction tool.",
        "வ ண க் க ம் நன்றி.": "வணக்கம் நன்றி.",
    }
    for before, expected in cases.items():
        got = analyze(before, model)["cleaned"]
        check(f"{before!r} -> {expected!r}", got == expected, f"got {got!r}")

    idempotent = analyze(analyze("home screen wallpaper. Ok.", model)["cleaned"], model)["cleaned"]
    check("cleaning is idempotent", idempotent == "home screen wallpaper. Ok.", idempotent)

    # Words that merely look like compounds must survive untouched.
    for word in ("microservices", "understanding", "The quick brown fox jumps over the lazy dog."):
        got = analyze(word, model)["cleaned"]
        check(f"leaves {word!r} alone", got == word, got)

    rule_fixes = [{"kind": "split", "before": "Spellingcorrection", "after": "Spelling correction"}]
    check("accepts a preserved fix", preserves(rule_fixes, "Spelling correction tool."))
    check("rejects an undone fix", not preserves(rule_fixes, "Spellingcorrection tool."))


def check_roundtrip() -> None:
    print("format round-trip")
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["வணக்கம் நன்றி.", "home screen wallpaper."],
            "lang": ["ta", "en"],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("csv", "tsv", "json", "jsonl", "parquet", "avro", "xlsx"):
            path = str(Path(tmp) / f"data.{fmt}")
            try:
                write_table(df, fmt, path)
                loaded = read_table(path)
            except Exception as exc:  # noqa: BLE001 - report, do not traceback
                check(f"{fmt} round-trip", False, f"{type(exc).__name__}: {exc}")
                continue
            same = list(loaded.df.columns) == list(df.columns) and list(loaded.df["text"]) == list(df["text"])
            check(f"{fmt} round-trip", same)

        # .txt is line-oriented: one non-empty line per row, no header.
        txt_path = str(Path(tmp) / "data.txt")
        write_table(df, "txt", txt_path, text_column="text")
        loaded = read_table(txt_path)
        check(
            "txt round-trip",
            list(loaded.df["text"]) == list(df["text"]),
            str(list(loaded.df["text"]))[:120],
        )

        # .zip bundles the cleaned text, a manifest, and the table in its source format.
        zip_path = str(Path(tmp) / "data.zip")
        write_table(
            df,
            "zip",
            zip_path,
            text_column="text",
            stem="data",
            source_format="csv",
            meta={"saved": 2},
        )
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            bundled = archive.read("data.cleaned.txt").decode("utf-8").splitlines()
        check(
            "zip bundle contents",
            {"data.cleaned.txt", "manifest.json", "data.cleaned.csv"} <= names,
            str(sorted(names)),
        )
        check("zip manifest rows", manifest.get("rows") == 2, str(manifest.get("rows")))
        check("zip text is one row per line", bundled == list(df["text"]), str(bundled))


def check_column_guess() -> None:
    print("column detection")
    check("guesses 'text'", guess_text_column(["id", "text", "lang"]) == "text")
    check("guesses 'sentence'", guess_text_column(["label", "sentence"]) == "sentence")
    check("returns None when unsure", guess_text_column(["a", "b"]) is None)

    # Headers written in another language miss the English hint list, so the
    # column that is far longer than everything beside it is the text.
    df = pd.DataFrame(
        {
            "வெளியிட்ட தேதி": ["2026-10-15", "2026-10-16"],
            "தலைப்பு": ["செய்தி ஒன்று", "செய்தி இரண்டு"],
            "உள்ளடக்கம்": ["நீண்ட உரை " * 40, "மற்றொரு நீண்ட உரை " * 40],
        }
    )
    check("falls back to the longest column", guess_text_column(list(df.columns), df) == "உள்ளடக்கம்")
    check("no fallback without a frame", guess_text_column(["a", "b"]) is None)


def check_api(base: str) -> None:
    print(f"live API at {base}")
    import httpx

    with httpx.Client(base_url=base, timeout=30) as client:
        health = client.get("/api/health")
        check("GET /api/health", health.status_code == 200, health.text[:120])
        if health.status_code == 200:
            body = health.json()
            print(f"       model: {body['llm']['provider'] or 'rules only'} · {body['llm']['model']}")

        opened = client.post("/api/sample/open", params={"name": "messy_tamil.csv"})
        check("POST /api/sample/open", opened.status_code == 200, opened.text[:120])
        if opened.status_code != 200:
            return
        session = opened.json()["session"]["id"]
        check("sample reports rows", opened.json()["session"]["rows"] > 0)

        row = client.get(f"/api/session/{session}/row", params={"index": 0})
        check("GET row", row.status_code == 200)
        cleaned = client.post(f"/api/session/{session}/clean", json={"index": 0, "mode": "rules"})
        check("POST clean (rules)", cleaned.status_code == 200)

        # The one-pass job, in rules mode so the test stays fast and offline-safe.
        started = client.post(f"/api/session/{session}/preclean", json={"mode": "rules"})
        check("POST preclean starts", started.status_code == 200 and started.json()["status"] in ("running", "done"))
        job = started.json()
        for _ in range(120):
            if job.get("status") != "running":
                break
            time.sleep(0.25)
            job = client.get(f"/api/session/{session}/preclean").json()
        check("preclean reaches done", job.get("status") == "done", str(job)[:160])
        check("preclean covered the column", job.get("scanned", 0) >= 1)
        precleaned = client.get(f"/api/session/{session}/row", params={"index": 1}).json()
        check("rows expose a pre-cleaned suggestion", precleaned.get("precleaned") is True)
        check("summary counts pre-cleaned rows", client.get(f"/api/session/{session}").json()["precleaned"] >= 1)

        saved = client.post(
            f"/api/session/{session}/save",
            json={"index": 0, "text": cleaned.json().get("cleaned", "")},
        )
        check("POST save", saved.status_code == 200 and saved.json().get("saved") == 1)

        export = client.post(f"/api/session/{session}/export", json={"format": "csv"})
        check("POST export", export.status_code == 200 and len(export.content) > 0)

        # Text datasets are a first-class import/export path.
        upload = client.post(
            "/api/upload",
            files={"file": ("mini.txt", io.BytesIO("first line\n\nsecond line\n".encode()), "text/plain")},
        )
        check("POST upload .txt", upload.status_code == 200, upload.text[:120])
        if upload.status_code == 200:
            text_session = upload.json()["session"]["id"]
            check("txt upload reports rows", upload.json()["session"]["rows"] == 2)
            as_txt = client.post(f"/api/session/{text_session}/export", json={"format": "txt"})
            check("POST export txt", as_txt.status_code == 200 and as_txt.text.count("\n") == 2)
            as_zip = client.post(f"/api/session/{text_session}/export", json={"format": "zip"})
            if as_zip.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(as_zip.content)) as archive:
                    names = set(archive.namelist())
                check("POST export zip bundle", {"mini.cleaned.txt", "manifest.json"} <= names, str(names))
            else:
                check("POST export zip bundle", False, as_zip.text[:120])
            blocked = client.post(f"/api/session/{text_session}/publish", json={"target": "huggingface"})
            check(
                "txt publish asks for a manual upload",
                blocked.status_code == 400 and blocked.json()["error"]["code"] == "manual_publish",
                blocked.text[:120],
            )

        missing = client.get("/api/jobs/does-not-exist")
        check(
            "missing load job is friendly",
            missing.status_code == 404 and "message" in missing.json().get("error", {}),
            missing.text[:120],
        )

        # A model pass must never break word boundaries: the rules join the
        # stray split and the guard stops the model re-breaking it.
        boundary_csv = io.BytesIO(
            b"text\nI down loaded the file today\nI downloaded the file today\n"
        )
        bup = client.post(
            "/api/upload", files={"file": ("boundaries.csv", boundary_csv, "text/csv")}
        )
        check("POST upload boundary csv", bup.status_code == 200, bup.text[:120])
        if bup.status_code == 200:
            bsession = bup.json()["session"]["id"]
            client.post(f"/api/session/{bsession}/column", json={"column": "text"})
            for index in (0, 1):
                result = client.post(
                    f"/api/session/{bsession}/clean", json={"index": index, "mode": "llm"}
                )
                text = result.json().get("cleaned", "")
                check(
                    f"row {index} keeps the word joined",
                    result.status_code == 200 and "downloaded" in text and "down loaded" not in text,
                    text[:80],
                )

        spa = client.get("/file-select")
        check("SPA route serves the app", spa.status_code == 200 and 'id="root"' in spa.text)

        friendly = client.get("/api/kaggle/search", params={"q": "tamil"})
        body = friendly.json()
        check(
            "errors are user-facing",
            "error" in body and "message" in body["error"] and "hint" in body["error"],
            friendly.text[:120],
        )


def check_boundaries() -> None:
    """The two failure modes that matter: never split a real word, and join a
    stray split even when the column also uses both halves separately."""
    print("boundary decisions")
    from backend.cleaner import BoundaryModel, analyze, freeze_boundaries

    rows = [
        "I downloaded the file yesterday",
        "down loaded the file today",
        "he downloaded it again",
        "down loaded again",
        "dow nloaded once more",
        "stopwatch and DeepSeek notes",
        "மசோதாவால் நன்றி",
        "ம சோதாவால் நன்றி",
        "download the dataset",
        "download the file",
    ]
    model = BoundaryModel(rows)

    def cleaned(text: str) -> str:
        return analyze(text, model)["cleaned"]

    check("joins a stray split", cleaned("down loaded the file today") == "downloaded the file today")
    check("joins a broken fragment", cleaned("dow nloaded once more") == "downloaded once more")
    check("joins tamil stray space", cleaned("ம சோதாவால் நன்றி") == "மசோதாவால் நன்றி")
    check("leaves 'downloaded' whole", cleaned("I downloaded the file yesterday") == "I downloaded the file yesterday")
    check("leaves 'stopwatch' whole", "stopwatch" in cleaned("stopwatch and DeepSeek notes"))
    check("leaves 'DeepSeek' whole", "DeepSeek" in cleaned("stopwatch and DeepSeek notes"))
    check("leaves tamil word whole", cleaned("மசோதாவால் நன்றி") == "மசோதாவால் நன்றி")

    # The model's own output goes through the same guard: spelling survives,
    # invented spaces do not.
    check(
        "reverts an invented split",
        freeze_boundaries("downloaded the file", "down loaded the file")[0] == "downloaded the file",
    )
    check(
        "reverts a camel-case split",
        freeze_boundaries("DeepSeek is fast", "Deep Seek is fast")[0] == "DeepSeek is fast",
    )
    check(
        "reverts a tamil split",
        freeze_boundaries("மசோதாவால் நன்றி", "மசோதா வால் நன்றி")[0] == "மசோதாவால் நன்றி",
    )
    check(
        "keeps a spelling edit inside the word",
        freeze_boundaries("this is speling", "this is spelling")[0] == "this is spelling",
    )
    check(
        "reverts a dropped word",
        freeze_boundaries("hello brave world", "hello world")[0] == "hello brave world",
    )
    check(
        "accepts a join the lexicon confirms",
        freeze_boundaries("down loaded the file", "downloaded the file", known=model.known)[0]
        == "downloaded the file",
    )
    check(
        "reverts a typed newline",
        freeze_boundaries("ends 15 here", "ends 15\\n here")[0] == "ends 15 here",
    )

    # Tamil is agglutinative, so a stem with its case ending looks exactly like
    # two known words glued together. On a small column that made the engine
    # split the correct "பெண்களின்" into "பெண் களின்"; both halves being known is
    # normal morphology, not evidence of a glue error.
    tamil_rows = [
        "பெண் ஒரு சொல் இங்கே",
        "பெண் மற்றொரு சொல் அங்கே",
        "களின் வேறொரு சொல் இங்கே",
        "களின் மற்றொரு சொல் அங்கே",
        "பெண்களின் உரிமைகள் பாதுகாக்கப்பட வேண்டும்",
    ]
    tamil_model = BoundaryModel(tamil_rows)
    check(
        "does not split an inflected tamil word",
        analyze("பெண்களின் உரிமைகள்", tamil_model)["cleaned"] == "பெண்களின் உரிமைகள்",
    )


def check_job_shape() -> None:
    """The load job is what the progress bar reads, so its shape is a contract."""
    print("load job snapshot")
    from backend.jobs import Job

    snapshot = Job(id="test", kind="huggingface").snapshot()
    for key in ("status", "phase", "message", "done", "total", "percent", "speed_bps", "eta_s"):
        check(f"snapshot has {key}", key in snapshot, str(sorted(snapshot)))
    job = Job(id="test", kind="huggingface")
    job.update(phase="download", done=512, total=1024)
    check("percent is byte-accurate", job.snapshot()["percent"] == 50, str(job.snapshot()["percent"]))


def check_parser() -> None:
    """Small models write almost-JSON: curly quotes, literal escapes, no end."""
    print("model output parsing")
    from backend.llm import _extract_json

    check(
        "reads a plain object",
        (_extract_json('{"cleaned": "I downloaded the file"}') or {}).get("cleaned")
        == "I downloaded the file",
    )
    check(
        "reads a fenced object",
        (_extract_json('```json\n{"cleaned": "spelling error"}\n```') or {}).get("cleaned")
        == "spelling error",
    )
    # The model copied the article's own quotation mark as the JSON delimiter.
    check(
        "accepts a curly closing quote",
        (_extract_json('{"cleaned": "நன்றி கூறியுள்ளனர்.”}') or {}).get("cleaned") == "நன்றி கூறியுள்ளனர்.",
    )
    # finish_reason=stop, but the string was never closed.
    check(
        "accepts an unterminated value",
        (_extract_json('```json\n{"cleaned": "வணக்கம் நன்றி') or {}).get("cleaned") == "வணக்கம் நன்றி",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="also test a running server")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server URL")
    args = parser.parse_args()

    check_repairs()
    check_boundaries()
    check_parser()
    check_job_shape()
    check_roundtrip()
    check_column_guess()
    if args.api:
        try:
            check_api(args.base)
        except Exception as exc:  # noqa: BLE001
            check("API reachable", False, f"start the server with ./run.sh ({type(exc).__name__})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
