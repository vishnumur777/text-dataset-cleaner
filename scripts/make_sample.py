"""Generate small demo datasets with realistic corruption.

Corruption models the reported problems:
* spaces inserted inside words  ("down load" / "வ ண க்க ம்")
* words glued together          ("homescreenwallpaper")
* non-breaking spaces, zero-width characters, space before punctuation

Characters are preserved, so a correct repair is reconstructable.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(11)

TAMIL_WORDS = [
    "வணக்கம்", "நன்றி", "தமிழ்", "மொழி", "மாணவர்", "பள்ளி",
    "நூலகம்", "கணினி", "தரவு", "தொகுப்பு", "சுத்தம்", "பயிற்சி",
    "எழுத்து", "வார்த்தை", "சொல்", "பாடம்", "ஆசிரியர்", "முடிவு",
]

ENGLISH_PHRASES = [
    "home screen wallpaper",
    "download the dataset",
    "open source model",
    "text cleaning pipeline",
    "machine learning dataset",
    "local language model",
    "hugging face repository",
    "tamil language corpus",
    "data quality report",
    "spelling correction tool",
]

TAIL_TA = ["இன்று பயில்கிறோம்.", "நாளை பார்க்கலாம்.", "மிகவும் முக்கியம்."]
TAIL_EN = ["We test it today.", "See you tomorrow.", "This is important."]


def insert_spaces(word: str, groups: int = 3) -> str:
    """Insert spaces inside a word without deleting any character."""
    if len(word) < 5:
        return word
    cuts = sorted(random.sample(range(2, len(word) - 1), k=min(groups, max(1, len(word) // 4))))
    out, prev = [], 0
    for cut in cuts:
        out.append(word[prev:cut])
        prev = cut
    out.append(word[prev:])
    return " ".join(part for part in out if part)


def glue(phrase: str) -> str:
    return phrase.replace(" ", "")


def make_row(i: int) -> dict[str, str]:
    lang = "ta" if i % 2 == 0 else "en"
    if lang == "ta":
        first, second = random.sample(TAMIL_WORDS, 2)
        sentence = f"{first} {second}. {random.choice(TAIL_TA)}"
    else:
        sentence = f"{random.choice(ENGLISH_PHRASES).capitalize()}. {random.choice(TAIL_EN)}"

    roll = random.random()
    words = sentence.split(" ")

    if roll < 0.30:
        # stray spaces inside the first content word
        words[0] = insert_spaces(words[0])
    elif roll < 0.50:
        # glue the first two words together
        words[0], words[1] = glue(words[0] + " " + words[1]), ""
    elif roll < 0.62:
        # split the first word into single letters
        words[0] = " ".join(words[0])

    text = " ".join(w for w in words if w)

    if random.random() < 0.12:
        text = text.replace(" ", "\u00a0", 1)
    if random.random() < 0.10:
        text = text.replace(".", " .", 1)
    if random.random() < 0.08:
        text = "\u200b" + text

    return {"id": str(i + 1), "text": text, "lang": lang}


def main() -> None:
    out = Path("samples")
    out.mkdir(exist_ok=True)
    rows = [make_row(i) for i in range(60)]
    pd.DataFrame(rows).to_csv(out / "messy_tamil.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows[:30]).to_json(out / "messy_mixed.json", orient="records", force_ascii=False, indent=2)
    (out / "note.txt").write_text(
        "Synthetic demo data: Tamil and English rows with stray spaces inside words,\n"
        "glued words, single-letter splits, non-breaking spaces and zero-width characters.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to samples/")


if __name__ == "__main__":
    main()
