"""Dataset sources: Hugging Face Hub and Kaggle.

Every network call is wrapped so the UI can show a friendly, actionable message
instead of a stack trace. Credentials are read from the environment or from
files on disk and are never returned to the browser.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .dataio import READABLE

HF_API = "https://huggingface.co/api"
HF_TIMEOUT = float(os.environ.get("TDC_HTTP_TIMEOUT", "30"))
DOWNLOAD_TIMEOUT = float(os.environ.get("TDC_DOWNLOAD_TIMEOUT", "600"))
CACHE_DIR = Path(os.environ.get("TDC_CACHE", ".cache/datasets")).resolve()
CONFIG_DIR = Path(os.environ.get("TDC_CONFIG", os.path.expanduser("~/.config/text-dataset-cleaner")))


class SourceError(Exception):
    def __init__(self, message: str, hint: str = "", code: str = "source_error"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


# A progress callback receives (bytes_done, bytes_total). A total of 0 means the
# server did not send a Content-Length, so the UI shows an indeterminate bar.
Progress = Callable[[int, int], None]

# One download per destination at a time. Two browser tabs asking for the same
# file must not write the same temporary path: that is what produced the
# "No such file or directory: ...part" error when two requests raced.
_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _download_lock(key: str) -> threading.Lock:
    with _download_locks_guard:
        lock = _download_locks.get(key)
        if lock is None:
            lock = _download_locks[key] = threading.Lock()
        return lock


def _report(on_progress: Progress | None, done: int, total: int) -> None:
    if on_progress is None:
        return
    try:
        on_progress(max(done, 0), max(total, 0))
    except Exception:  # a broken UI callback must never fail a download
        pass


def _cached(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _client(timeout: float = HF_TIMEOUT) -> httpx.Client:
    headers = {"User-Agent": "text-dataset-cleaner/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return httpx.Client(timeout=timeout, headers=headers, follow_redirects=True), token


# --------------------------------------------------------------------------- HF


def hf_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    client, token = _client()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = client.get(
            f"{HF_API}/datasets",
            params={"search": query, "limit": limit, "sort": "downloads", "direction": -1, "full": "true"},
            headers=headers,
        )
        if r.status_code == 401:
            raise SourceError(
                "Hugging Face rejected the request.",
                "Set HF_TOKEN if you are hitting a private or gated dataset.",
                "hf_unauthorized",
            )
        r.raise_for_status()
        raw = r.json()
    except SourceError:
        raise
    except httpx.TimeoutException as exc:
        raise SourceError("Hugging Face took too long to respond.", "Check your internet connection and retry.", "timeout") from exc
    except httpx.HTTPError as exc:
        raise SourceError("Could not reach Hugging Face.", f"{type(exc).__name__}", "network") from exc
    finally:
        client.close()

    return [
        {
            "id": item.get("id", ""),
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": [t for t in item.get("tags", []) if isinstance(t, str)][:8],
            "updated": item.get("lastModified", ""),
            "gated": bool(item.get("gated")),
            "private": bool(item.get("private")),
        }
        for item in raw
        if item.get("id")
    ]


def hf_files(repo_id: str, revision: str = "main") -> list[dict[str, Any]]:
    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise SourceError("That repository id does not look right.", "Use the form owner/dataset-name.", "bad_repo")
    client, token = _client()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = client.get(
            f"{HF_API}/datasets/{quote(repo_id, safe='/')}/tree/{quote(revision)}",
            params={"recursive": "true"},
            headers=headers,
        )
        if r.status_code == 404:
            raise SourceError(
                "Hugging Face could not find that dataset.",
                "It may have been renamed, deleted, or made private.",
                "hf_not_found",
            )
        if r.status_code in (401, 403):
            raise SourceError(
                "This dataset is gated or private.",
                "Add HF_TOKEN on the machine running the server to request access.",
                "hf_forbidden",
            )
        r.raise_for_status()
        entries = r.json()
    except SourceError:
        raise
    except httpx.HTTPError as exc:
        raise SourceError("Could not list files from Hugging Face.", f"{type(exc).__name__}", "network") from exc
    finally:
        client.close()

    files = []
    for item in entries:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        ext = os.path.splitext(path)[1].lower()
        files.append(
            {
                "path": path,
                "size": item.get("size", 0),
                "supported": ext in READABLE,
                "ext": ext.lstrip("."),
            }
        )
    files.sort(key=lambda f: (not f["supported"], f["path"]))
    return files


def hf_download(
    repo_id: str,
    path: str,
    revision: str = "main",
    on_progress: Progress | None = None,
) -> str:
    """Download one file from the Hub into the local cache.

    The temporary name is unique per request and guarded by a per-file lock, so
    concurrent downloads of the same file cannot delete each other's `.part`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = repo_id.replace("/", "__") + "__" + path.replace("/", "__")
    dest = CACHE_DIR / safe_name
    if _cached(dest):
        size = dest.stat().st_size
        _report(on_progress, size, size)
        return str(dest)

    url = f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}/resolve/{quote(revision)}/{quote(path)}"
    with _download_lock(f"hf:{safe_name}"):
        if _cached(dest):  # another request finished while we waited
            size = dest.stat().st_size
            _report(on_progress, size, size)
            return str(dest)

        client, token = _client(DOWNLOAD_TIMEOUT)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        tmp = CACHE_DIR / f".{safe_name}.{uuid.uuid4().hex[:8]}.part"
        try:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 404:
                    raise SourceError(
                        "That file is no longer available.",
                        "Refresh the file list and pick another file.",
                        "hf_not_found",
                    )
                if r.status_code in (401, 403):
                    raise SourceError(
                        "You do not have access to this file.",
                        "Accept the dataset terms on huggingface.co, or set HF_TOKEN.",
                        "hf_forbidden",
                    )
                r.raise_for_status()
                try:
                    total = int(r.headers.get("content-length") or 0)
                except ValueError:
                    total = 0
                # Publish the total before the first byte arrives so the bar is
                # determinate from the very first poll instead of sitting on an
                # ellipsis while the connection is set up.
                _report(on_progress, 0, total)
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1024 * 256):
                        fh.write(chunk)
                        done += len(chunk)
                        _report(on_progress, done, total)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise SourceError(
                    "Hugging Face returned an empty file.",
                    "Refresh the file list and pick another file.",
                    "hf_empty_file",
                )
            os.replace(tmp, dest)
        except SourceError:
            raise
        except httpx.HTTPError as exc:
            raise SourceError(
                "The download did not finish.",
                f"{type(exc).__name__}: retry in a moment.",
                "network",
            ) from exc
        except OSError as exc:
            raise SourceError(
                "The downloaded file could not be saved.",
                f"{type(exc).__name__}: check free disk space, then retry.",
                "cache_write_failed",
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)
            client.close()

    size = dest.stat().st_size
    _report(on_progress, size, size)
    return str(dest)


def hf_preview_rows(repo_id: str, limit: int = 100) -> tuple[list[str], list[dict[str, Any]]]:
    """Fallback for repositories that only publish arrow shards: use the
    dataset viewer API (datasets-server) to grab the first rows."""
    client, _ = _client()
    try:
        splits = client.get(f"https://datasets-server.huggingface.co/splits", params={"dataset": repo_id})
        splits.raise_for_status()
        items = splits.json().get("splits", [])
        if not items:
            raise SourceError("This dataset has no viewable split.", "Try another repository.", "hf_no_split")
        first = items[0]
        rows = client.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": repo_id,
                "config": first.get("config"),
                "split": first.get("split"),
                "offset": 0,
                "length": limit,
            },
        )
        rows.raise_for_status()
        data = rows.json()
        rows_out = [r.get("row", {}) for r in data.get("rows", [])]
        columns = data.get("features", []) and [f.get("name") for f in data["features"]]
        if not rows_out:
            raise SourceError("The dataset viewer returned no rows.", "Pick a file from the repository instead.", "hf_empty")
        return columns or list(rows_out[0].keys()), rows_out
    except SourceError:
        raise
    except httpx.HTTPError as exc:
        raise SourceError(
            "This repository has no directly readable files and the dataset viewer failed.",
            f"{type(exc).__name__}",
            "hf_viewer_error",
        ) from exc
    finally:
        client.close()


# ----------------------------------------------------------------------- Kaggle


def _kaggle_creds() -> tuple[str, str] | None:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    for candidate in (CONFIG_DIR / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"):
        try:
            data = json.loads(candidate.read_text())
            if data.get("username") and data.get("key"):
                return data["username"], data["key"]
        except Exception:
            continue
    return None


def kaggle_status() -> dict[str, Any]:
    creds = _kaggle_creds()
    return {"connected": bool(creds), "username": creds[0] if creds else "", "source": "file/env" if creds else ""}


def kaggle_connect(username: str, key: str) -> dict[str, Any]:
    username, key = (username or "").strip(), (key or "").strip()
    if not username or not key:
        raise SourceError("Enter both your Kaggle username and API key.", "Find them at kaggle.com/settings → API.", "bad_credentials")
    try:
        r = httpx.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"page": 1, "pageSize": 1},
            auth=(username, key),
            timeout=20,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise SourceError("Could not reach Kaggle.", "Check your internet connection.", "network") from exc
    if r.status_code in (401, 403):
        raise SourceError(
            "Kaggle rejected those credentials.",
            "Regenerate the API key at kaggle.com/settings and paste it again.",
            "kaggle_unauthorized",
        )
    if r.status_code >= 400:
        raise SourceError("Kaggle returned an unexpected response.", f"HTTP {r.status_code}", "kaggle_error")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}))
    os.chmod(path, 0o600)
    return {"connected": True, "username": username}


def _kaggle_auth() -> tuple[str, str]:
    creds = _kaggle_creds()
    if not creds:
        raise SourceError(
            "Kaggle is not connected yet.",
            "Open the Kaggle tab and add your username and API key, or place kaggle.json in ~/.kaggle/.",
            "kaggle_not_connected",
        )
    return creds


def kaggle_search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    auth = _kaggle_auth()
    try:
        r = httpx.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "page": 1, "pageSize": limit},
            auth=auth,
            timeout=30,
            follow_redirects=True,
        )
        if r.status_code in (401, 403):
            raise SourceError(
                "Kaggle rejected the stored credentials.",
                "Reconnect Kaggle with a fresh API key.",
                "kaggle_unauthorized",
            )
        r.raise_for_status()
        data = r.json()
    except SourceError:
        raise
    except httpx.HTTPError as exc:
        raise SourceError("Kaggle search failed.", f"{type(exc).__name__}", "network") from exc

    out = []
    for item in data:
        ref = item.get("ref") or f"{item.get('ownerName','')}/{item.get('datasetSlug','')}"
        out.append(
            {
                "id": ref,
                "title": item.get("title", ref),
                "subtitle": item.get("subtitle", ""),
                "size": item.get("totalBytes", 0),
                "downloads": item.get("downloadCount", 0),
                "updated": item.get("lastUpdated", ""),
            }
        )
    return out


def kaggle_files(ref: str) -> list[dict[str, Any]]:
    auth = _kaggle_auth()
    ref = (ref or "").strip().strip("/")
    if "/" not in ref:
        raise SourceError("Kaggle dataset id must look like owner/dataset.", "", "bad_repo")
    owner, slug = ref.split("/", 1)
    try:
        r = httpx.get(
            f"https://www.kaggle.com/api/v1/datasets/view/{quote(owner)}/{quote(slug)}",
            auth=auth,
            timeout=30,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as exc:
        raise SourceError("Could not read that Kaggle dataset.", f"{type(exc).__name__}", "kaggle_error") from exc

    files = []
    for f in data.get("files") or []:
        path = f.get("name", "")
        ext = os.path.splitext(path)[1].lower()
        files.append(
            {
                "path": path,
                "size": f.get("totalBytes", 0),
                "supported": ext in READABLE,
                "ext": ext.lstrip("."),
            }
        )
    if not files:
        raise SourceError(
            "Kaggle did not return a file list for this dataset.",
            "Download the dataset manually and use the Upload tab.",
            "kaggle_no_files",
        )
    files.sort(key=lambda x: (not x["supported"], x["path"]))
    return files


def kaggle_download(ref: str, path: str, on_progress: Progress | None = None) -> str:
    """Download one member of a Kaggle dataset archive.

    Kaggle always serves a zip of the whole dataset, so progress is measured on
    that archive; the member is extracted once the archive is complete.
    """
    auth = _kaggle_auth()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the full member path in the cache key: a dataset may hold
    # `train/data.csv` and `test/data.csv`, and both must not share a file.
    stamp = ref.replace("/", "__") + "__" + path.replace("/", "__")
    dest = CACHE_DIR / stamp
    if _cached(dest):
        size = dest.stat().st_size
        _report(on_progress, size, size)
        return str(dest)

    with _download_lock(f"kaggle:{stamp}"):
        if _cached(dest):
            size = dest.stat().st_size
            _report(on_progress, size, size)
            return str(dest)

        owner, slug = ref.split("/", 1)
        archive = CACHE_DIR / f".{stamp}.{uuid.uuid4().hex[:8]}.archive.part"
        member = CACHE_DIR / f".{stamp}.{uuid.uuid4().hex[:8]}.part"
        try:
            with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True, auth=auth) as client:
                with client.stream(
                    "GET",
                    f"https://www.kaggle.com/api/v1/datasets/download/{quote(owner)}/{quote(slug)}",
                ) as r:
                    if r.status_code in (401, 403):
                        raise SourceError(
                            "Kaggle rejected the stored credentials.",
                            "Reconnect Kaggle with a fresh API key.",
                            "kaggle_unauthorized",
                        )
                    r.raise_for_status()
                    try:
                        total = int(r.headers.get("content-length") or 0)
                    except ValueError:
                        total = 0
                    # Determinate from the first poll, before any byte lands.
                    _report(on_progress, 0, total)
                    done = 0
                    with open(archive, "wb") as out:
                        for chunk in r.iter_bytes(1024 * 256):
                            out.write(chunk)
                            done += len(chunk)
                            _report(on_progress, done, total)

            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as zf:
                    wanted = None
                    base = os.path.basename(path)
                    for name in zf.namelist():
                        if name == path or os.path.basename(name) == base:
                            wanted = name
                            break
                    if wanted is None:
                        raise SourceError(
                            "That file is not inside the dataset archive.",
                            "Refresh the file list and pick another file.",
                            "kaggle_missing_file",
                        )
                    with zf.open(wanted) as src, open(member, "wb") as out:
                        shutil.copyfileobj(src, out)
            else:
                shutil.copyfile(archive, member)
            if not member.exists() or member.stat().st_size == 0:
                raise SourceError(
                    "Kaggle returned an empty file.",
                    "Refresh the file list and pick another file.",
                    "kaggle_empty_file",
                )
            os.replace(member, dest)
        except SourceError:
            raise
        except httpx.HTTPError as exc:
            raise SourceError("Kaggle download failed.", f"{type(exc).__name__}", "network") from exc
        except (OSError, zipfile.BadZipFile) as exc:
            raise SourceError(
                "The Kaggle download could not be unpacked.",
                f"{type(exc).__name__}: check free disk space, then retry.",
                "kaggle_unpack_failed",
            ) from exc
        finally:
            archive.unlink(missing_ok=True)
            member.unlink(missing_ok=True)

    size = dest.stat().st_size
    _report(on_progress, size, size)
    return str(dest)


# --------------------------------------------------------------- publishing


def hf_upload(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    *,
    message: str = "Cleaned with text-dataset-cleaner",
    token: str | None = None,
) -> dict[str, Any]:
    """Commit one file into an existing dataset repository.

    `huggingface_hub` handles the commit protocol (including LFS) for us, so the
    same code works for a small CSV and a large parquet.
    """
    repo_id = (repo_id or "").strip()
    if "/" not in repo_id:
        raise SourceError("That Hugging Face repository id does not look right.", "Use the form owner/dataset-name.", "bad_repo")
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SourceError(
            "Publishing to Hugging Face needs a write token.",
            "Set HF_TOKEN on the machine running this server, then restart it. The token never reaches the browser.",
            "hf_no_token",
        )
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise SourceError(
            "Hugging Face publishing support is not installed on this server.",
            "Install it with: pip install huggingface_hub",
            "missing_dependency",
        ) from exc

    try:
        api = HfApi(token=token)
        info = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=message,
            create_repo=False,
        )
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        if status in (401, 403):
            raise SourceError(
                "That Hugging Face token cannot write to this dataset.",
                "Use a token with write access to the repository, or set HF_TOKEN to its owner's token.",
                "hf_forbidden",
            ) from exc
        if status == 404:
            raise SourceError(
                "Hugging Face could not find that dataset repository.",
                "Publishing writes into the repository the file came from; it must still exist and be yours.",
                "hf_not_found",
            ) from exc
        raise SourceError(
            "Hugging Face rejected the upload.",
            f"{type(exc).__name__}: {str(exc)[:200]}",
            "hf_upload_failed",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise SourceError(
            "The upload to Hugging Face did not finish.",
            f"{type(exc).__name__}: {str(exc)[:200]}",
            "hf_upload_failed",
        ) from exc

    return {
        "target": "huggingface",
        "ref": repo_id,
        "path": path_in_repo,
        "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
        "commit": getattr(info, "oid", "") or "",
        "message": message,
    }


_kaggle_publish_lock = threading.Lock()


def _kaggle_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def kaggle_companion_ref(ref: str) -> str:
    """Kaggle cannot append a folder to an existing dataset version.

    A new version always replaces the dataset's file list, so writing cleaned
    data into the source dataset would delete the original files. Instead the
    cleaned rows go to a companion dataset named `<slug>-cleaned`.
    """
    owner, _, slug = (ref or "").partition("/")
    owner = owner or (kaggle_status().get("username") or "")
    base = _kaggle_slug(slug or "dataset")[:42].strip("-") or "dataset"
    return f"{owner}/{base}-cleaned"


def kaggle_publish(
    ref: str,
    local_path: str,
    *,
    message: str = "Cleaned with text-dataset-cleaner",
    public: bool = False,
) -> dict[str, Any]:
    """Create (or version) a Kaggle dataset that holds the cleaned file."""
    username, key = _kaggle_auth()
    ref = (ref or "").strip().strip("/")
    if "/" not in ref:
        raise SourceError("Kaggle dataset id must look like owner/dataset.", "Use the form owner/dataset-name.", "bad_repo")
    owner, slug = ref.split("/", 1)
    if _kaggle_slug(slug) != slug or len(slug) < 6 or len(slug) > 50:
        raise SourceError(
            "Kaggle dataset names must be 6-50 characters of lowercase letters, numbers and dashes.",
            "Edit the target name in the publish box, for example owner/my-dataset-cleaned.",
            "kaggle_bad_slug",
        )
    if owner != username:
        raise SourceError(
            "Kaggle only lets you publish to datasets your account owns.",
            f"The connected account is {username}; change the target to {username}/<name>.",
            "kaggle_not_owner",
        )

    filename = Path(local_path).name
    title = f"{Path(filename).stem[:34]} cleaned"[:50]
    meta = {
        "title": title,
        "id": ref,
        "licenses": [{"name": "other"}],
        "subtitle": "Cleaned text dataset",
        "description": (
            "Cleaned rows produced with text-dataset-cleaner. "
            "Kaggle dataset versions replace the previous file list, so this is a "
            "companion dataset and the original files are untouched."
        ),
    }

    try:
        with _kaggle_publish_lock:
            previous = {k: os.environ.get(k) for k in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = username, key
            try:
                # Importing the Kaggle package authenticates a module-level API
                # object and prints CLI help when no token is configured; keep
                # that noise out of the server log.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    from kaggle.api.kaggle_api_extended import KaggleApi

                    api = KaggleApi()
                    api.authenticate()
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            with tempfile.TemporaryDirectory(prefix="tdc-kaggle-") as tmp_dir:
                folder = Path(tmp_dir)
                shutil.copyfile(local_path, folder / filename)
                (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

                exists = False
                try:
                    api.dataset_status(ref)
                    exists = True
                except Exception:  # noqa: BLE001 - 404 means "create it"
                    exists = False

                if exists:
                    result = api.dataset_create_version(
                        str(folder),
                        version_notes=message,
                        quiet=True,
                        dir_mode="skip",
                    )
                else:
                    result = api.dataset_create_new(
                        str(folder),
                        public=public,
                        quiet=True,
                        dir_mode="skip",
                    )
    except SourceError:
        raise
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - KaggleApi calls exit() on bad auth
        raise SourceError(
            "Kaggle rejected the publish request.",
            f"{type(exc).__name__}: {str(exc)[:200]}",
            "kaggle_publish_failed",
        ) from exc

    status = str(getattr(result, "status", "") or "").lower()
    error = str(getattr(result, "error", "") or "")
    if status and status != "ok":
        raise SourceError(
            "Kaggle did not accept the new dataset version.",
            error[:240] or "Open kaggle.com and check the dataset's status.",
            "kaggle_publish_failed",
        )
    if error:
        raise SourceError("Kaggle did not accept the new dataset version.", error[:240], "kaggle_publish_failed")

    return {
        "target": "kaggle",
        "ref": ref,
        "path": filename,
        "url": f"https://www.kaggle.com/datasets/{ref}",
        "status": status or "ok",
        "message": message,
    }


def sources_status() -> dict[str, Any]:
    hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return {
        "huggingface": {"public": True, "token": hf_token},
        "kaggle": kaggle_status(),
    }
