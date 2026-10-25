<!-- DEV draft id 4916787 — tags: hf26challenge, gemma, llm, ai. AI disclosure: some_ai.
     The source PNGs live in docs/screenshots/; on DEV they are hosted at the iili.io
     links used below, because DEV has no image-upload API. -->

*Submission for the [Hacktoberfest Open-Source AI Challenge: Week 3 — Slow and Steady](https://dev.to/events/challenges/hacktoberfest-week3-2026-10-19). Built during the challenge window, runs entirely on my laptop. Code: **[github.com/vishnumur777/text-dataset-cleaner](https://github.com/vishnumur777/text-dataset-cleaner)**.*

![Text Dataset Cleaner: original on the left with changes in red, corrected and editable text on the right with additions in green](https://iili.io/nk0rfbS.png)

## The chore I kept putting off

I have a folder of Tamil and English sentences scraped from forms, chat logs and OCR dumps. They are *mostly* fine — the worst kind of fine:

| In the data | What it should be |
| --- | --- |
| `down loa ded the dataset` | `downloaded the dataset` |
| `homescreenwallpaper` | `home screen wallpaper` |
| `d o w n l o a d e d` | `downloaded` |
| `வ ண க் க ம்` | `வணக்கம்` |
| `text .` (non-breaking space) | `text.` |

A regex can strip a zero-width space. It cannot decide whether `homescreenwallpaper` is one word or three — and that decision *is* the job. Tamil makes it harder: the script is alphasyllabic, so one visible word is several code points, and a stray space between a consonant and its vowel sign gives you text that looks almost right and is wrong.

Cleaning this by hand is exactly the task I keep not doing. So I built a tool that does it and shows its work.

## What it is, and how you actually use it

**Text Dataset Cleaner** is a small local web app. Five steps, no cloud:

1. **Bring a file** — drag in a CSV, TSV, JSON, JSONL, Parquet, Avro, Excel or a line-per-row `.txt`; pull one straight from **Hugging Face** or **Kaggle**; or open the bundled sample.
2. **Pick the text column** — the app guesses it and you confirm.
3. **Clean the column in one pass** with the *Clean all N rows* button, or step through one row at a time.
4. **Review the diff** — left is the original with deleted characters and spaces in red, right is the corrected text with additions in green and it is fully editable. You save the rows you agree with.
5. **Export or publish** — nine output formats including `.txt` and a zip bundle, or write the cleaned file back to Hugging Face or Kaggle.

![Choose a file from disk, Hugging Face or Kaggle](https://iili.io/nk0rIgj.png)

![Character-level red/green diff, editable on the right](https://iili.io/nk0rudx.png)

![Export in the original format, as plain text, or as a zip bundle](https://iili.io/nk0rErJ.png)

There is a light theme plus Tokyo Night, Synthwave and Catppuccin for the late shift:

![The same diff in a dark theme](https://iili.io/nk0rNLX.png)

## Two engines, and the bug that taught me which one should win

Two engines run in sequence. Getting the split of responsibility right was the entire project.

### Engine 1 — a rule engine that owns word boundaries

A **rule engine** learns token frequencies from the column you loaded and solves word boundaries with a dynamic program:

1. **Hygiene** — NFC normalisation, zero-width characters, control characters, NBSP, space before punctuation.
2. **Dataset lexicon** — a token is *blocked* from being re-split only with evidence: both halves must be used as their own words in the column **and** at least twice as often as the glued form. A Latin word the bundled dictionary knows is never treated as a glue error, so `downloaded` and `stopwatch` stay whole, while a corrupt `downloadthe` still fails to become a word.
3. **Dynamic program** over boundaries — keep, merge 2–6 fragments, or split one token into 2–3 known words. Costs come from unigram frequencies, an English word list for split candidates, and a character trigram model for unknowns.

### Engine 2 — Gemma as a spelling-only layer

The model never sees the raw cell. It receives the **already rule-cleaned** text and returns `{"cleaned": "..."}`. That ordering matters: the model is good at "documnet → document" and bad at everything about where a word ends.

### The bug that proved the split

My first version gave Gemma both jobs. The report that came back was precise and damning:

- it split real words — `downloaded` → `down loaded`, `stopwatch` → `sto pwatch`, `DeepSeek` → `Deep Seek`, and did the same to Tamil words;
- and it left a genuine glue error — `down loaded` — glued the wrong way round, i.e. untouched.

So a 1B model invented boundaries that were not there and missed the ones that were. The fix was to take boundaries away from it entirely. The model's answer is now replayed over the rule-cleaned text **one edit at a time**:

- a change of letters *inside* a word is kept (that is spelling);
- adding, removing or moving a space is **reverted**;
- the one exception is a removed space whose fused word the lexicon or the dictionary already knows — that is what lets `down loaded` become `downloaded`;
- deleting or inventing a whole word is reverted.

The rule-cleaned boundaries always win. The model keeps only its spelling fixes.

A second guard came out of the same testing, and it is the difference between a demo and something I would run on a real file. **A split needs evidence:** early on `microservices` became `micro services` because both halves are dictionary words, so now a split is refused outright for a token the column repeats, for a camel/Pascal-case name, and for anything the dictionary already knows. `homescreenwallpaper` still splits; `microservices`, `stopwatch` and `understanding` are left alone.

## Pointing it at a real dataset found a deeper bug

The bundled sample is tidy, so the first version passed it. Then I ran the tool on a real public dataset — a Tamil newspaper archive, thousands of pages — and took one day's parquet: six articles, about 2,900 characters each. It found three things the sample never could.

**The rule engine was splitting correct Tamil words.** Tamil is agglutinative: `பெண்களின்` is `பெண்` plus a case ending, so my test — *both halves are known words, so this must be a glue error* — is simply false for the language. It turned `பெண்களின்` into `பெண் களின்`. The fix keeps the same test for Latin, but for any non-Latin script a split must now also be evidenced by the column writing those two parts side by side *somewhere else*. Measured on the file, all three splits the engine was making were wrong, and the words it left whole were right.

**Gemma looked like it was doing nothing — but the model wasn't the problem, my code was.** I was giving it a *guessed* amount of room to answer, worked out from how many characters the text had, and capped at 1,024. One long article needed **1,284 tokens just to repeat itself**, so the reply ran out of space before it reached the corrected text — and the app quietly kept the old text as if the model had agreed. Now the app asks the model how much room the text actually needs, and how much its window holds; if a cell really is too big, it says so instead of staying silent. (A 1B model can still decide to summarise a four-thousand-character article rather than fix it — that answer is thrown out, with the reason shown.)

**And the parser was throwing away good answers.** Where the model did finish, it closed its JSON string with a curly quote copied from the article itself — `{"cleaned": "…கூறியுள்ளனர்.”}` — which is not valid JSON, so `json.loads` failed and 2,350 usable characters were discarded. The parser now accepts any quote flavour, and an answer the model leaves unclosed, because the boundary guard keeps the result safe either way.

The honest result on that file: the rules stopped damaging correct words, and the model layer stopped failing silently. It still misses a stray space inside a single Tamil word (`நினை க்கும்`) when the column never saw the joined form. I tried two merge rules that looked sound and both merged dozens of legitimate pairs, so I left that as a documented limit rather than a guess.

## The lazy payoff: pay the model once, not once per row

Running the model on the row you are looking at is a 7-second wait per row, and a 60-row file becomes a coffee break that keeps happening. Clicking *play* through a column is exactly the slow, steady, manual work this week is about — and I did not want to do it.

So there is a **Clean all 60 rows** button. One pass:

```text
rules pass on every row (milliseconds)
  → skip rows already clean
  → de-duplicate identical cells
  → Gemma on the rest, three calls in flight
  → cache every result on the session
```

After that, every row opens instantly. Numbers from the bundled 60-row sample on this laptop:

| | |
| --- | --- |
| rows the rule engine changed | 28 |
| rows the model never saw (already clean) | 29 |
| model calls | 31 |
| rejected by the guard (rules kept) | 1 |
| wall clock, three calls in flight | 163 s |
| one row on demand, afterwards | 0 s |

That is roughly the same model work as clicking every row — done once, before review, instead of during it. On a mostly-clean column it is far less, because the skipped rows never touch the model at all. `llama.cpp` runs four slots, so the batch worker keeps three requests in flight; measured throughput is about one cell per 5.5 s at three in flight versus 6.5–8 s for a single call. A 200-row column with half its rows needing polish finishes in about nine minutes, once.

## Why open innovation matters here

The data is the reason. These are Tamil sentences from forms and chats — the kind of text people assume is disposable and which is often the most personal thing in a dataset. With a local 1B model, nothing leaves the machine: no API key, no upload, no per-row bill. I can point the same code at Ollama or a bigger Gemma checkpoint by changing a URL, and I can read every line of the pipeline that touches my data.

A closed model would have been easier to call and impossible to inspect. For a tool whose entire output is "here is what I changed in your text", inspectability *is* the product.

## Community wisdom that shaped this

- [Trust the parser, not the prompt: what running a cheap LLM in production taught me](https://dev.to/avlihachev/trust-the-parser-not-the-prompt-what-running-a-cheap-llm-in-production-taught-me-3908) by [@avlihachev](https://dev.to/avlihachev) — the escalator of prompt rewrites that never fixed a cheap model, and the parse-boundary guard that did. Exactly why the boundary rule moved out of the prompt and into code here.
- [Stop Splitting Strings the Wrong Way: Discover Intl.Segmenter](https://dev.to/josephciullo/stop-splitting-strings-the-wrong-way-discover-intlsegmenter-4667) by [@josephciullo](https://dev.to/josephciullo) — splitting text on spaces is the classic bug; the grapheme-aware segmenter is what keeps a Tamil vowel sign attached to its base letter in the diff.
- [How I automated text dataset cleaning for ML training with AI (and why regex wasn't enough)](https://dev.to/lattanzi/how-i-automated-text-dataset-cleaning-for-ml-training-with-ai-and-why-regex-wasnt-enough-3h19) by [@lattanzi](https://dev.to/lattanzi) — the same case for judgment over regex, which is why the rule engine carries a frequency signal instead of a growing pile of patterns.

## Repo and running it locally

The whole thing is open source: **[github.com/vishnumur777/text-dataset-cleaner](https://github.com/vishnumur777/text-dataset-cleaner)** — MIT licensed, FastAPI backend, React UI and tests in one repository.

### 1. Clone it

```bash
git clone https://github.com/vishnumur777/text-dataset-cleaner.git
cd text-dataset-cleaner
```

### 2. Create `models/` and download Gemma into it

`run.sh` expects `models/gemma-3-1b-it-Q4_K_M.gguf`, so the weights go in a `models/` folder at the repo root:

```bash
mkdir -p models
curl -L -o models/gemma-3-1b-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

That is the 806 MB `Q4_K_M` quantisation — the exact file the numbers above were measured on. Another Gemma GGUF works too: point `TDC_MODEL_PATH` at it and `run.sh` will use that instead.

### 3. Give it a llama.cpp server

```bash
curl -LsSf https://llama.app/install.sh | sh   # installs `llama` into ~/.llama-app
```

Already have a `llama-server`? Set `LLAMA_BIN=/path/to/llama` and skip the install. `run.sh` launches the server on port 8080 and waits for its health check before starting the app.

### 4. Run it

```bash
./run.sh
```

That single command creates `.venv`, installs the backend dependencies, generates the bundled sample if it is missing, starts `llama.cpp` with the model, builds the UI on first run (needs `npm`), and serves the app.

Open `http://127.0.0.1:8000/file-select`, or `/clean-data?demo=1` for the bundled sample.

Skip steps 2 and 3 and nothing breaks: the app still runs, and with no model server it falls back to the deterministic rule engine on its own and says so. Gemma is the spelling polish, not the loader.

## Known limits

The pass is CPU-bound and honest about its ETA (both the download bar and the batch bar show live speed and an estimate). Avro and Excel exports need `fastavro` / `openpyxl`. Hugging Face publishing needs a write token in `HF_TOKEN`. Kaggle publishing goes to a companion `<slug>-cleaned` dataset, because Kaggle's API replaces a dataset's entire file list and writing into the source would delete its original files. `.txt` sources are manual uploads. And boundary repair learns from the column you loaded: on a thin file it can miss a stray space inside a word whose joined form never appears, which is a limit I chose to keep rather than trade for false merges. The model's in-word edits are not grammar-checked either — the guard enforces word boundaries, not meaning, so it let Gemma drop an inflectional ending it read as a typo on that Tamil file (`மந்தரில்` → `மந்தர்`); that edit shows up in the diff for exactly this reason, so you can undo it.

## Prize Categories

I'm entering **Best Use of Gemma** and the **Overall** prize.

**Best Use of Gemma.** Gemma 3 1B (`Q4_K_M`) runs locally through `llama.cpp` on CPU only — no API key, no GPU, no data leaving the machine. It is a real layer of the pipeline, not a garnish. The rule engine owns word boundaries, so the model is given only the job it is genuinely good at (spelling), and it is measured on a whole column rather than a demo row: three calls in flight, one pass, every result cached. Crucially, every answer passes a deterministic validation gate that is allowed to reject it — `freeze_boundaries` replays the model's output one edit at a time and keeps only in-word letter changes, reverting anything that adds, removes or moves a space. On the bundled sample it rejected 1 row outright; on the real six-article Tamil file it rejected every answer, and finding out why was the whole point — a character-count token budget that truncated long cells before the JSON value, and a strict parser that discarded a value closed with a curly quote. Both were bugs in my code, not the model, and both are fixed. A 1B model earns its place here, or it doesn't.

**Overall.** Judges reward a complete, working tool, and this one is finished rather than prototyped: nine import formats, nine export formats, in-app editing, publish back to Hugging Face or Kaggle, and a tested CLI. It deletes a manual chore I was genuinely avoiding, which is exactly the *Slow and Steady* point — the boring pass you keep not doing, done once, on your own machine.

---

*Session trail: the whole build above — and the real-dataset debugging — was one agent session, saved **in full** with DevRelay: every prompt, tool call and dead end, from the first command to the last fix, rather than a highlight reel.*

{% agent_session 1214 %}
