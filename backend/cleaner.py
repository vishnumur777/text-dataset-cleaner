"""Deterministic text repair: unicode hygiene + dataset-adaptive word boundaries.

The two reported problems are one problem seen from both ends:

  "dow nloa ded"          -> "downloaded"            (stray spaces inside a word)
  "homescreenwallpaper"   -> "home screen wallpaper"  (two words glued together)

Both are decided with a dynamic program over token boundaries, scored by a
lexicon learned from the dataset itself (plus an English word list for split
candidates). Tamil gets its lexicon from the column, so the model adapts to the
domain instead of relying on hand-written rules per language.
"""
from __future__ import annotations

import difflib
import functools
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
WS_RE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
PUNCT_BEFORE_RE = re.compile(r"\s+([,.;:!?%)\]}\u201d\u2019])")
PUNCT_AFTER_RE = re.compile(r"([(\[{\u201c\u2018])\s+")
WORD_RE = re.compile(r"^([\w\u0b80-\u0bff]+)(.*)$", re.UNICODE)
# A literal two-character escape the model may type into the text ("\n", "\r", "\t").
ESCAPE_RE = re.compile(r"\\[nrt]")

KEEP_BONUS = 0.4
MERGE_PENALTY = 1.2
SPLIT_PENALTY = -1.0   # split is preferred when both halves are common words
EN_COST = 9.0
OOV_BASE = 16.0
OOV_PER_CHAR = 0.5
MIN_WORD_FREQ = 1

_WORDS_FILE = Path(__file__).with_name("data") / "english_words.txt"


@functools.lru_cache(maxsize=1)
def english_words() -> frozenset[str]:
    try:
        raw = _WORDS_FILE.read_text(encoding="utf-8", errors="ignore").split()
        return frozenset(w.lower() for w in raw if len(w) >= 2)
    except Exception:
        return frozenset()


@dataclass
class Fix:
    kind: str          # space | split | spelling | chars
    before: str
    after: str
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """Hygiene only: never joins or splits words."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = WS_RE.sub(" ", text)
    text = PUNCT_BEFORE_RE.sub(r"\1", text)
    text = PUNCT_AFTER_RE.sub(r"\1", text)
    return text.strip()


def _split_punct(token: str) -> tuple[str, str]:
    match = WORD_RE.match(token)
    if not match:
        return token, ""
    return match.group(1), match.group(2)


def _is_tamil(word: str) -> bool:
    return any("\u0b80" <= ch <= "\u0bff" for ch in word)


def _is_latin(word: str) -> bool:
    return all(ch.isascii() and ch.isalpha() for ch in word) and bool(word)


def _has_inner_capital(word: str) -> bool:
    """True for camel/Pascal-case tokens: "DeepSeek", "JavaScript", "TamilQA".

    Those decompose into dictionary words but are one name, so they must never
    be split just because both halves exist in a word list.
    """
    return any(ch.isupper() for ch in word[1:]) and any(ch.islower() for ch in word)


class BoundaryModel:
    """Lexicon + character n-gram model learned from the user's own column."""

    def __init__(self, texts: Iterable[str], use_english: bool = True):
        texts = [str(t) for t in texts]
        freq: Counter[str] = Counter()
        # Adjacent-token pairs, used as evidence that a boundary is real: a
        # non-Latin split is only trusted when the column itself writes the two
        # parts next to each other somewhere else.
        pairs: Counter[tuple[str, str]] = Counter()
        for text in texts:
            cores = [_split_punct(tok)[0] for tok in normalize(text).split()]
            cores = [core for core in cores if core]
            for core in cores:
                freq[core] += 1
            for left, right in zip(cores, cores[1:]):
                pairs[(left, right)] += 1
        self.pairs = pairs
        self.freq = freq
        self.total = max(sum(freq.values()), 1)
        self.english = english_words() if use_english else frozenset()
        self.blocked = self._find_blocked(freq)
        self.vocab = {
            w
            for w, c in freq.items()
            if len(w) >= 2
            # A lone Latin token only counts as a word if a dictionary agrees;
            # otherwise "homescreenwallpaper" would teach the model it is one.
            and (
                c >= 2
                or (c == 1 and len(w) >= 6 and (not _is_latin(w) or w.lower() in self.english))
            )
            and w not in self.blocked
        }
        self._build_char_model()
        self._word_cost_cache: dict[str, float] = {}

    def _find_blocked(self, freq: Counter[str]) -> set[str]:
        """Tokens that are almost certainly two words glued together.

        A token is blocked when it can be split into parts that each appear at
        least as often as the glued form. That is what keeps corrupt rows from
        teaching the model that "downloadthe" is a word.
        """
        blocked: set[str] = set()
        for token, count in freq.items():
            latin = _is_latin(token)
            tamil = _is_tamil(token)
            if not (latin or tamil):
                continue
            if len(token) < (10 if latin else 9):
                continue
            min_part = 4 if latin else 3
            threshold = max(count * 2, 2)
            for i in range(min_part, len(token) - min_part + 1):
                left, right = token[:i], token[i:]
                if not (self._is_word_like(left) and self._is_word_like(right)):
                    continue
                # A word only blocks when the column uses *both* halves as words
                # clearly more often than the whole. A real compound such as
                # "downloaded" can coexist with "down" and "loaded", and a
                # dictionary word is never a glue error at all.
                if latin and token.lower() in self.english:
                    continue
                if freq.get(left, 0) < 2 or freq.get(right, 0) < 2:
                    continue
                if min(self._part_freq(left), self._part_freq(right)) >= threshold:
                    blocked.add(token)
                    break
        return blocked

    def _part_freq(self, part: str) -> int:
        count = self.freq.get(part, 0)
        if count:
            return count
        return 1 if (_is_latin(part) and len(part) >= 3 and part.lower() in self.english) else 0

    def _is_word_like(self, part: str) -> bool:
        if part in self.freq:
            return True
        return _is_latin(part) and len(part) >= 3 and part.lower() in self.english


    def _build_char_model(self) -> None:
        self.tri: Counter[str] = Counter()
        self.bi: Counter[str] = Counter()
        self.uni: Counter[str] = Counter()
        for word, count in self.freq.items():
            padded = f"\u0002{word}\u0003"
            for i in range(len(padded) - 2):
                self.tri[padded[i : i + 3]] += count
            for i in range(len(padded) - 1):
                self.bi[padded[i : i + 2]] += count
            for ch in padded:
                self.uni[ch] += count
        self.tri_total = max(sum(self.tri.values()), 1)
        self.bi_total = max(sum(self.bi.values()), 1)
        self.uni_total = max(sum(self.uni.values()), 1)

    def char_score(self, word: str) -> float:
        if not word:
            return -20.0
        padded = f"\u0002{word}\u0003"
        scores: list[float] = []
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            p_tri = self.tri.get(tri, 0) / self.tri_total
            p_bi = self.bi.get(padded[i : i + 2], 0) / self.bi_total
            backoff = 0.75 * p_bi * (self.uni.get(padded[i], 0) / self.uni_total + 1e-9)
            scores.append(math.log(max(p_tri, backoff, 1e-9)))
        return sum(scores) / max(len(scores), 1)

    def known(self, word: str) -> bool:
        if not word:
            return False
        if word in self.vocab:
            return True
        return _is_latin(word) and len(word) >= 3 and word.lower() in self.english

    def word_cost(self, word: str) -> float:
        if not word:
            return 1e9
        cached = self._word_cost_cache.get(word)
        if cached is not None:
            return cached
        count = self.freq.get(word, 0)
        if count and word not in self.blocked:
            cost = -math.log(count / self.total)
            # A once-seen long Latin token that no dictionary knows is more
            # likely two glued words than real vocabulary, so do not let its
            # single occurrence make it look trustworthy.
            if (
                _is_latin(word)
                and word.lower() not in self.english
                and count <= 2
                and len(word) >= 12
            ):
                cost = max(cost, OOV_BASE + OOV_PER_CHAR * len(word))
        elif word not in self.blocked and _is_latin(word) and len(word) >= 3 and word.lower() in self.english:
            cost = EN_COST
        else:
            cost = OOV_BASE + OOV_PER_CHAR * len(word) - 0.4 * self.char_score(word)
        if len(self._word_cost_cache) < 100_000:
            self._word_cost_cache[word] = cost
        return cost

    # ------------------------------------------------------------------ splits
    def _pair_evidence(self, parts: list[str]) -> bool:
        """True when the column uses these parts adjacently as separate tokens."""
        return all(self.pairs.get((a, b), 0) > 0 for a, b in zip(parts, parts[1:]))

    def best_split(self, token: str) -> tuple[list[str], float] | None:
        """Segment a glued token into 2-3 known words, or return None.

        Returns the parts and the cost of taking the split, so the DP and this
        decision agree on the same numbers.
        """
        if len(token) < 7 or self.known(token):
            return None
        if not (_is_latin(token) or _is_tamil(token)):
            return None
        # A token the column repeats is vocabulary, not a glue error. And a
        # camel/Pascal-case token ("DeepSeek", "GitHub") is a single name that
        # merely happens to decompose into dictionary words.
        if self.freq.get(token, 0) >= 2:
            return None
        if token not in self.blocked and _has_inner_capital(token):
            return None

        best: tuple[float, list[str]] | None = None
        # two parts
        for i in range(3, len(token) - 2):
            left, right = token[:i], token[i:]
            if self.known(left) and self.known(right):
                cost = self.word_cost(left) + self.word_cost(right) + SPLIT_PENALTY
                if best is None or cost < best[0]:
                    best = (cost, [left, right])
        # three parts
        if len(token) >= 11:
            for i in range(3, len(token) - 5):
                for j in range(i + 3, len(token) - 2):
                    parts = [token[:i], token[i:j], token[j:]]
                    if all(self.known(p) for p in parts):
                        cost = sum(self.word_cost(p) for p in parts) + SPLIT_PENALTY * 2
                        if best is None or cost < best[0]:
                            best = (cost, parts)

        if best is None:
            return None
        parts = best[1]

        # Non-Latin scripts are agglutinative: a Tamil stem carries its case
        # endings, so "both halves are known words" is *normal* ("பெண்" + "களின்"
        # inside "பெண்களின்"), not evidence of a glue error. On a small column that
        # is exactly what turned a correct "பெண்களின்" into "பெண் களின்". For those
        # scripts the column itself has to show the boundary elsewhere.
        if not _is_latin(token) and not self._pair_evidence(parts):
            return None

        # A split needs evidence. Either a part also appears as its own token
        # somewhere in this dataset, or the token is rare and long enough that
        # glue is the only plausible explanation. Without one of those,
        # "microservices" would become "micro services".
        has_evidence = any(self.freq.get(part, 0) > 0 for part in parts)
        long_rare_dictionary_word = (
            _is_latin(token)
            and len(token) >= 15
            and self.freq.get(token, 0) <= 1
            and all(part.lower() in self.english for part in parts)
        )
        if not (has_evidence or long_rare_dictionary_word):
            return None

        # Long tokens are what unseparated corruption produces, so a split earns
        # a small length bonus. A rare token that is not in the dictionary is
        # compared against the out-of-vocabulary cost instead of being trusted
        # as a real word just because it appeared once.
        cost = best[0] - 0.15 * len(token)
        baseline = self.word_cost(token)
        if (
            _is_latin(token)
            and token.lower() not in self.english
            and self.freq.get(token, 0) <= 2
        ):
            baseline = max(baseline, OOV_BASE + OOV_PER_CHAR * len(token))
        if cost >= baseline:
            return None
        return parts, cost

    # ------------------------------------------------------------------ repair
    def repair(self, text: str) -> tuple[str, list[Fix]]:
        cleaned = normalize(text)
        tokens = cleaned.split()
        if not tokens:
            return cleaned, []

        fixes: list[Fix] = []
        tokens, fixes = self._collapse_letter_runs(tokens, fixes)

        n = len(tokens)
        INF = float("inf")
        best = [INF] * (n + 1)
        back: list[tuple[int, str, list[str]] | None] = [None] * (n + 1)
        best[0] = 0.0

        for i in range(n):
            if best[i] == INF:
                continue
            core, suffix = _split_punct(tokens[i])
            # keep
            cost = best[i] + self.word_cost(core) - KEEP_BONUS
            if cost < best[i + 1]:
                best[i + 1] = cost
                back[i + 1] = (i, "keep", [tokens[i]])

            # split one glued token
            split = self.best_split(core)
            if split:
                parts, split_cost = split
                cost = best[i] + split_cost
                if cost < best[i + 1]:
                    best[i + 1] = cost
                    back[i + 1] = (i, "split", [" ".join(parts) + suffix])

            # merge neighbouring word fragments
            for k in range(2, min(7, n - i + 1)):
                chunk = tokens[i : i + k]
                cores = []
                ok = True
                for tok in chunk:
                    c, s = _split_punct(tok)
                    cores.append(c)
                    if s and tok is not chunk[-1]:
                        ok = False
                        break
                if not ok:
                    continue
                candidate = "".join(cores)
                if len(candidate) < 4 or not self.known(candidate):
                    continue
                if any(len(c) > 5 for c in cores[:-1]):
                    continue
                tail = _split_punct(chunk[-1])[1]
                # A join the dictionary already knows is trustworthy even when
                # the column's own rows were too corrupt to teach it, so it is
                # scored as a known word instead of as out-of-vocabulary. This is
                # what lets "down loaded" and "dow nloaded" reach "downloaded".
                merged_cost = self.word_cost(candidate)
                if _is_latin(candidate) and candidate.lower() in self.english:
                    merged_cost = min(merged_cost, EN_COST)
                cost = best[i] + merged_cost + MERGE_PENALTY * (k - 1)
                if cost < best[i + k]:
                    best[i + k] = cost
                    back[i + k] = (i, "merge", [candidate + tail])

        out: list[str] = []
        j = n
        while j > 0:
            step = back[j]
            if step is None:  # defensive
                out.append(tokens[j - 1])
                j -= 1
                continue
            start, action, value = step
            source_txt = " ".join(tokens[start:j])
            out.extend(value)
            if action == "merge":
                fixes.append(Fix("space", source_txt, value[0], "stray spaces inside a word"))
            elif action == "split":
                fixes.append(Fix("split", source_txt, value[0], "two words glued together"))
            j = start

        out.reverse()
        return " ".join(out), fixes

    def _collapse_letter_runs(self, tokens: list[str], fixes: list[Fix]) -> tuple[list[str], list[Fix]]:
        """Join runs of 3+ single-character tokens: 'd o w n l o a d e d'."""
        if sum(1 for t in tokens if len(_split_punct(t)[0]) == 1) < 3:
            return tokens, fixes
        merged: list[str] = []
        run: list[str] = []
        for tok in tokens + [""]:
            if len(_split_punct(tok)[0]) == 1:
                run.append(tok)
                continue
            if len(run) >= 3:
                word = "".join(run)
                fixes.append(Fix("space", " ".join(run), word, "letters split by spaces"))
                merged.append(word)
            elif run:
                merged.extend(run)
            run = []
            if tok:
                merged.append(tok)
        return merged, fixes


def analyze(text: str, model: BoundaryModel) -> dict:
    cleaned, fixes = model.repair(text)
    return {
        "original": text,
        "cleaned": cleaned,
        "fixes": [f.as_dict() for f in fixes],
        "changed": cleaned != normalize(text),
    }


def preserves(rule_fixes: list[dict], cleaned: str) -> bool:
    """An LLM answer must not undo a boundary fix the rules already made.

    The model gets the rule-cleaned text as its input, but a 1B model can still
    return a "correction" that re-breaks a word ("Spelling correction" back to
    "Spellingcorrection"). Every merge/split the rules made must survive.
    """
    for fix in rule_fixes:
        if fix.get("kind") in ("space", "split") and fix.get("after") and fix["after"] not in cleaned:
            return False
    return True


def _is_whole_token(text: str, start: int, end: int) -> bool:
    """True when text[start:end] is a complete whitespace-delimited word."""
    left = start == 0 or text[start - 1].isspace()
    right = end >= len(text) or text[end].isspace()
    return left and right


def _joined_word(text: str, start: int, end: int) -> str:
    """The word that a removed slice would fuse together: 'down loaded' -> 'downloaded'."""
    before = re.split(r"\s+", text[:start])[-1] if text[:start].strip() else ""
    after = re.split(r"\s+", text[end:])[0] if text[end:].strip() else ""
    return before + after


def freeze_boundaries(before: str, after: str, known=None) -> tuple[str, int]:
    """Keep a model answer's spelling edits and revert every spacing edit.

    A 1B model asked to proofread will happily invent spaces inside words
    ("downloaded" -> "down loaded", "stopwatch" -> "sto pwatch", "DeepSeek" ->
    "Deep Seek") and just as happily leave a real glue error alone. Word
    boundaries are the deterministic rule engine's job, so the model's answer is
    replayed over the rule-cleaned text one edit at a time:

      * an edit that replaces letters inside a word is kept (spelling),
      * an edit that adds or removes whitespace is reverted, unless it removes a
        space and the fused word is one ``known`` accepts (a join the dataset or
        the dictionary agrees with, e.g. "down loaded" -> "downloaded"),
      * deleting or inventing a whole word is reverted.

    Returns the repaired text and how many spacing edits were dropped.
    """
    if not after or after == before:
        return before, 0

    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    out: list[str] = []
    reverted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        old, new = before[i1:i2], after[j1:j2]
        if tag == "equal":
            out.append(old)
            continue

        # An escape sequence or a newline the model typed into the text is a
        # boundary edit in disguise: llama-server can return the two characters
        # "\n" inside a cell, which would otherwise look like a letter insertion.
        touches_space = (
            old.strip() != old
            or new.strip() != new
            or "\n" in new
            or "\t" in new
            or "\r" in new
            or ESCAPE_RE.search(new) is not None
            or ESCAPE_RE.search(old) is not None
        )
        if touches_space:
            # Only a join the lexicon agrees with is allowed through.
            allow_join = bool(
                old.strip() == ""
                and not new
                and known is not None
                and known(_joined_word(before, i1, i2).lower())
            )
            if allow_join:
                out.append(new)
            else:
                out.append(old)
                reverted += 1
            continue

        # A whole word appearing or disappearing is content damage, not spelling.
        if not new and _is_whole_token(before, i1, i2):
            out.append(old)
            reverted += 1
            continue
        if not old and _is_whole_token(after, j1, j2):
            reverted += 1
            continue

        out.append(new)

    return "".join(out), reverted
