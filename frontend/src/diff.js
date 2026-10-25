/** Git-diff-style text comparison.
 *
 * Two passes:
 *   1. a coarse LCS over whitespace-preserving word tokens (fast, keeps whole
 *      words readable and avoids "every letter moved" noise), then
 *   2. every replaced run is re-diffed per grapheme cluster, so only the
 *      characters that actually changed are highlighted.
 *
 * Grapheme clusters matter for Tamil: a vowel sign or combining mark must stay
 * attached to its base letter, otherwise the diff would colour half a glyph.
 */

const MAX_TOKENS = 1200;
const MAX_REFINE_CELLS = 400_000;

let segmenter = null;
try {
  segmenter =
    typeof Intl !== "undefined" && Intl.Segmenter
      ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
      : null;
} catch {
  segmenter = null;
}

export function graphemes(text) {
  const value = String(text ?? "");
  if (!value) return [];
  if (segmenter) return Array.from(segmenter.segment(value), (part) => part.segment);
  return Array.from(value);
}

export function tokenize(text) {
  return String(text ?? "").match(/\s*\S+|\s+/g) ?? [];
}

function lcsTable(a, b) {
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  return dp;
}

/** Coarse, word-token diff. Exact token equality: a whitespace-only change is
 *  a real change and is refined (and highlighted) in the second pass. */
function coarseDiff(before, after) {
  const a = tokenize(before);
  const b = tokenize(after);

  if (a.length === 0 && b.length === 0) return [];
  if (a.length > MAX_TOKENS || b.length > MAX_TOKENS) {
    return [{ type: "del", value: before }, { type: "ins", value: after }];
  }

  const dp = lcsTable(a, b);
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      ops.push({ type: "same", value: a[i] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", value: a[i] });
      i += 1;
    } else {
      ops.push({ type: "ins", value: b[j] });
      j += 1;
    }
  }
  while (i < a.length) ops.push({ type: "del", value: a[i++] });
  while (j < b.length) ops.push({ type: "ins", value: b[j++] });
  return ops;
}

/** Character-level LCS over grapheme clusters. */
function charDiff(before, after) {
  const a = graphemes(before);
  const b = graphemes(after);
  if (!a.length) return after ? [{ type: "ins", value: after }] : [];
  if (!b.length) return [{ type: "del", value: before }];
  if (a.length * b.length > MAX_REFINE_CELLS) {
    return [
      { type: "del", value: before },
      { type: "ins", value: after },
    ];
  }

  const dp = lcsTable(a, b);
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      ops.push({ type: "same", value: a[i] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", value: a[i] });
      i += 1;
    } else {
      ops.push({ type: "ins", value: b[j] });
      j += 1;
    }
  }
  while (i < a.length) ops.push({ type: "del", value: a[i++] });
  while (j < b.length) ops.push({ type: "ins", value: b[j++] });
  return ops;
}

function merge(ops) {
  const out = [];
  for (const op of ops) {
    if (!op.value) continue;
    const last = out[out.length - 1];
    if (last && last.type === op.type) last.value += op.value;
    else out.push({ type: op.type, value: op.value });
  }
  return out;
}

/** Turn each contiguous changed block into a character-level diff. */
function refine(ops) {
  const out = [];
  let i = 0;
  while (i < ops.length) {
    if (ops[i].type === "same") {
      out.push(ops[i]);
      i += 1;
      continue;
    }
    const removed = [];
    const added = [];
    while (i < ops.length && ops[i].type !== "same") {
      if (ops[i].type === "del") removed.push(ops[i].value);
      else added.push(ops[i].value);
      i += 1;
    }
    const before = removed.join("");
    const after = added.join("");
    if (before && after) {
      // A change that only moves spaces — a word joined ("down loaded" ->
      // "downloaded") or split ("homescreenwallpaper" -> "home screen
      // wallpaper") — has almost nothing to colour as a character diff: the
      // dropped space sits on the left and nothing is added on the right, so
      // the correction is invisible. Highlight the whole affected run on both
      // sides instead, which is what the reader needs to see.
      if (before.replace(/\s+/g, "") === after.replace(/\s+/g, "")) {
        out.push({ type: "del", value: before }, { type: "ins", value: after });
      } else {
        out.push(...charDiff(before, after));
      }
    }
    else if (before) out.push({ type: "del", value: before });
    else if (after) out.push({ type: "ins", value: after });
  }
  return merge(out);
}

/** Public diff: `[{type: "same"|"del"|"ins", value: string}, ...]` */
export function diffWords(before, after) {
  if (before === after) {
    return before ? [{ type: "same", value: before }] : [];
  }
  return refine(coarseDiff(before, after));
}

export function countChanges(ops) {
  let removed = 0;
  let added = 0;
  for (const op of ops) {
    if (op.type === "del") removed += graphemes(op.value).length;
    if (op.type === "ins") added += graphemes(op.value).length;
  }
  return { removed, added, total: removed + added };
}
