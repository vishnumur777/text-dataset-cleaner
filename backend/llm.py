"""LLM access for the two tasks this project needs:

1. spelling corrections
2. repairing unwanted / missing spaces inside words

Works with any OpenAI-compatible chat endpoint. Two local favourites are
auto-detected: a llama.cpp `llama-server` (recommended, runs Gemma 3 1B GGUF) and
Ollama. If neither is running the API degrades to the deterministic rule engine
instead of failing.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from typing import Any

import httpx

from .cleaner import freeze_boundaries

LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_TIMEOUT = float(os.environ.get("TDC_LLM_TIMEOUT", "120"))
MODEL_NAME = os.environ.get("TDC_MODEL", "")

SYSTEM = (
    "You are a meticulous spelling checker for Tamil and English text. "
    "You fix misspelled words and nothing else. "
    "You never add, remove, move or invent a space, and you never split one word "
    "into two or join two words into one; the existing word boundaries are correct "
    "as given. "
    "You return the whole text, never a summary, never a fragment, and you never "
    "repeat the input. "
    "You never translate, paraphrase, summarise, or add new information. "
    "You reply with a single JSON object and nothing else, using plain ASCII "
    "double quotes."
)

PROMPT = """Fix the spelling mistakes in the text below.

Rules:
- Change only letters that are misspelled.
- Never add, remove or move a space. "downloaded" must stay "downloaded";
  do not return "down loaded" or "dow nloaded".
- Never split a word into two words, and never join two words into one.
- Keep the language, punctuation, casing, numbers, code and URLs unchanged.
- Never translate, add or remove sentences.
- Return the full text, not a fragment, and do not repeat the input.

Examples of the reply format:
{{"cleaned": "I downloaded the document"}}
{{"cleaned": "this is speling eror"}}

Text to fix:
{text}

Reply with one JSON object and nothing else."""


def _http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(LLM_TIMEOUT, connect=3.0))


def detect() -> dict[str, Any]:
    """Return the best available provider without raising."""
    with _http() as client:
        try:
            r = client.get(f"{LLAMA_URL}/health", timeout=2.0)
            if r.status_code < 500:
                return {"provider": "llama.cpp", "url": LLAMA_URL, "model": _llama_model(), "ok": True}
        except Exception:
            pass
        try:
            r = client.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                preferred = MODEL_NAME or next(
                    (m for m in models if "gemma" in m.lower()), models[0] if models else ""
                )
                return {"provider": "ollama", "url": OLLAMA_URL, "model": preferred, "ok": bool(models)}
        except Exception:
            pass
    return {"provider": None, "url": "", "model": "", "ok": False}


def _count_tokens(text: str, provider: dict[str, Any]) -> int:
    """How many tokens the model needs to read or restate this text.

    ``max_tokens`` used to be derived with ``len(text) * 2 + 120`` characters,
    which pinned every cell over ~450 chars to the same 1024-token ceiling. A
    4,580-char Tamil cell needs far more than that just to say the text back, so
    the answer was truncated before the model ever reached ``"cleaned"``.
    """
    if provider.get("provider") == "llama.cpp" and text:
        try:
            with _http() as client:
                r = client.post(f"{provider['url']}/tokenize", json={"content": text}, timeout=20.0)
                r.raise_for_status()
                tokens = r.json().get("tokens")
                if isinstance(tokens, list) and tokens:
                    return len(tokens)
        except Exception:
            pass
    # Conservative fallback for Ollama or a failed probe: ~3 chars per token.
    return max(1, int(len(text) / 3.0) + 8)


def _context_size(provider: dict[str, Any], default: int = 4096) -> int:
    """The server's context window, so a cell that cannot fit is reported honestly."""
    if provider.get("provider") == "llama.cpp":
        try:
            with _http() as client:
                r = client.get(f"{provider['url']}/props", timeout=5.0)
                r.raise_for_status()
                props = r.json()
                n = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
                if n:
                    return int(n)
        except Exception:
            pass
    return default


def _llama_model() -> str:
    if MODEL_NAME:
        return MODEL_NAME
    try:
        with _http() as client:
            data = client.get(f"{LLAMA_URL}/v1/models", timeout=2.0).json()
        return (data.get("data") or [{}])[0].get("id", "") or "local"
    except Exception:
        return "local"


_QUOTES = "\"\u201c\u201d\u2018\u2019'"


def _unescape(value: str) -> str:
    """Turn a JSON string body back into text, even if its escapes are literal."""
    try:
        return json.loads(f'"{value}"')
    except Exception:
        # The model typed the escapes literally ("\n" instead of a newline).
        return (
            value.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
        )


def _scrape_cleaned(raw: str) -> str | None:
    """Last-resort read of the ``cleaned`` value.

    A 1B model writing Tamil does two things that break ``json.loads``:

    * it closes the JSON string with a curly quote copied from the text itself
      ("...னர்.”}), which an ASCII-only pattern will not match, and
    * it simply stops -- ``finish_reason`` is ``stop`` -- leaving the string
      unterminated (``{"cleaned": "விண்ணை ...``).

    The value is still perfectly good in both cases, so read up to the closing
    quote (of any flavour) if there is one, otherwise take the rest of the
    output. The boundary guard downstream restores anything the model dropped.
    """
    key = raw.find("cleaned")
    if key < 0:
        return None
    colon = raw.find(":", key)
    if colon < 0:
        return None
    start = colon + 1
    while start < len(raw) and raw[start] in " \t\r\n":
        start += 1
    if start < len(raw) and raw[start] in _QUOTES:
        start += 1
    end = re.search(r"[\"\u201c\u201d\u2018\u2019']\s*[,}\]]", raw[start:])
    if end is not None:
        value = raw[start : start + end.start()]
    else:
        value = re.sub(r"[\s`\]}]+$", "", raw[start:])
    if not value.strip():
        return None
    return _unescape(value)


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw, flags=re.S).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Small models often emit almost-JSON (stray brackets, a missing value).
    # Recover the fields we need instead of throwing the whole answer away.
    cleaned_match = re.search(r'"cleaned"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, flags=re.S)
    if not cleaned_match:
        cleaned = _scrape_cleaned(raw)
        if cleaned is None:
            return None
        fixes = []
        for m in re.finditer(
            r'"before"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"after"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, flags=re.S
        ):
            fixes.append({"before": m.group(1), "after": m.group(2), "kind": "space"})
        return {"cleaned": cleaned, "fixes": fixes, "recovered": True}
    try:
        cleaned = json.loads(f'"{cleaned_match.group(1)}"')
    except Exception:
        cleaned = cleaned_match.group(1)
    fixes = []
    for m in re.finditer(
        r'"before"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"after"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, flags=re.S
    ):
        try:
            before = json.loads(f'"{m.group(1)}"')
            after = json.loads(f'"{m.group(2)}"')
        except Exception:
            before, after = m.group(1), m.group(2)
        fixes.append({"before": before, "after": after, "kind": "space"})
    return {"cleaned": cleaned, "fixes": fixes, "recovered": True}


def _diff_fixes(before: str, after: str) -> list[dict[str, str]]:
    """Fallback: derive fixes when the model omits them."""
    fixes: list[dict[str, str]] = []
    sm = difflib.SequenceMatcher(None, before.split(), after.split())
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(before.split()[i1:i2]) or "(missing)"
        new = " ".join(after.split()[j1:j2]) or "(removed)"
        fixes.append({"before": old, "after": new, "kind": "space" if " " in old or " " in new else "spelling"})
    return fixes[:20]


def _validate(original: str, cleaned: str, fixes: Any) -> tuple[bool, str]:
    if not cleaned or not cleaned.strip():
        return False, "The model returned an empty result."
    if len(cleaned) > len(original) * 2.5 + 80 or len(cleaned) < len(original) * 0.45:
        return False, "The model rewrote too much text, so the change was discarded."
    low = cleaned.lower()
    for bad in ("```", "as an ai", "i cannot", "json object", "cleaned\":"):
        if bad in low:
            return False, "The model returned an unusable response, so the change was discarded."
    ratio = difflib.SequenceMatcher(None, original, cleaned).ratio()
    if ratio < 0.45:
        return False, "The model changed the meaning too much, so the change was discarded."
    if not isinstance(fixes, list):
        return True, ""
    return True, ""


def clean_with_llm(
    text: str,
    provider: dict[str, Any] | None = None,
    known: Any = None,
) -> dict[str, Any]:
    """Run the model on one cell. Raises RuntimeError with a friendly message.

    ``known`` is an optional ``BoundaryModel.known`` callback. It is only used to
    let a lexically-confirmed join through the boundary guard ("down loaded" ->
    "downloaded"); every other spacing edit the model makes is reverted.
    """
    provider = provider or detect()
    if not provider.get("ok"):
        raise RuntimeError(
            "No local model is running. Start llama-server (Gemma 3 1B) or Ollama, then try again."
        )

    started = time.time()
    prompt_text = PROMPT.format(text=text)
    # The model has to restate the whole cell, so the output budget is the length
    # of the text plus a small margin for the JSON wrapper, and it is bounded by
    # what is actually left in the context window.
    prompt_tokens = _count_tokens(SYSTEM + prompt_text, provider)
    echo_tokens = _count_tokens(text, provider) + 48
    n_ctx = _context_size(provider)
    room = n_ctx - prompt_tokens - 16
    if room < 64:
        return {
            "accepted": False,
            "cleaned": text,
            "fixes": [],
            "provider": provider.get("provider", ""),
            "model": provider.get("model", ""),
            "elapsed_s": round(time.time() - started, 2),
            "reason": (
                f"This cell needs about {prompt_tokens + echo_tokens} tokens, more than the "
                f"model's {n_ctx}-token window, so it was left to the rule engine. "
                "Split long cells or raise the server's context size (-c)."
            ),
            "boundary_reverted": 0,
        }
    out_tokens = max(64, min(echo_tokens, room))

    if provider["provider"] == "llama.cpp":
        payload = {
            "model": provider.get("model") or "local",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": out_tokens,
            "stream": False,
        }
        url = f"{provider['url']}/v1/chat/completions"
        with _http() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
    elif provider["provider"] == "ollama":
        payload = {
            "model": provider.get("model"),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": out_tokens},
        }
        url = f"{provider['url']}/api/chat"
        with _http() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            raw = r.json()["message"]["content"]
    else:
        raise RuntimeError("Unknown model provider.")

    elapsed = round(time.time() - started, 2)
    obj = _extract_json(raw) or {}
    cleaned = str(obj.get("cleaned", "")).strip()
    fixes = obj.get("fixes", [])
    ok, reason = _validate(text, cleaned, fixes)
    if not ok:
        return {
            "accepted": False,
            "cleaned": text,
            "fixes": [],
            "provider": provider["provider"],
            "model": provider.get("model", ""),
            "elapsed_s": elapsed,
            "reason": reason,
        }

    # Keep the spelling, drop the spacing: the rule engine owns word boundaries.
    cleaned, boundary_reverted = freeze_boundaries(text, cleaned, known=known)
    if not cleaned.strip():
        return {
            "accepted": False,
            "cleaned": text,
            "fixes": [],
            "provider": provider["provider"],
            "model": provider.get("model", ""),
            "elapsed_s": elapsed,
            "reason": "The model returned an empty result.",
            "boundary_reverted": 0,
        }

    if not isinstance(fixes, list) or not fixes:
        fixes = _diff_fixes(text, cleaned)
    return {
        "accepted": True,
        "cleaned": cleaned,
        "fixes": fixes[:20],
        "provider": provider["provider"],
        "model": provider.get("model", ""),
        "elapsed_s": elapsed,
        "reason": "",
        "boundary_reverted": boundary_reverted,
    }
