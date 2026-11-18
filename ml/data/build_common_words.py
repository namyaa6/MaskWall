"""Common-word statistics for the lowercase-name re-casing approach (docs/decisions.md Day 3.13). No word is chosen by hand.

    python ml/data/build_common_words.py
    -> ml/cache/common_words_stats.json   (raw counts/shares; the pre-registered thresholds are applied later, in the browser code)

Sources (all independent of every evaluation set):
  * Hinglish-TOP, TRAIN split only (WillHeld/hinglish_top; Apache-2.0). NOT its test split (= evaluation set E1) and not validation.
    Roman-script tokens of the code-switched queries (cs_query), counted; for each token the share of its occurrences INSIDE a person/contact slot of
    the parse (cs_parse) are recorded separately and are never "common".
  * WikiText-103 (raw), first train shard (Salesforce/wikitext on the Hugging Face Hub; Wikipedia text; licence on the dataset card:
    CC BY-SA 3.0 and GFDL). Word counts keep letter case. For each word: lowercase count and lowercase share = lower / (lower + Capitalised).
    The share is the name filter: "rose", "mark" are mostly lowercase in running text; "priya", "rahul", "john" are almost always capitalised.
    (The plan first named the Leipzig Corpora Collection; its licence could not be read from the publisher's anti-bot-protected pages, which
    must not be bypassed, so it was dropped and its download discarded. Decision and reason: docs/decisions.md Day 3.13.)
The NER model's own vocabulary (Apache-2.0) is the third source and is read in the browser, not here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "cache" / "common_words_stats.json"
# person/contact slots of the TOP parse: a token is a name, never a common word, when >= half of its occurrences are inside one of these
PERSON_SLOTS = {"SL:RECIPIENT", "SL:PERSON_REMINDED", "SL:CONTACT", "SL:SENDER", "SL:ATTENDEE", "SL:ORGANIZER_EVENT", "SL:MUSIC_ARTIST_NAME"}
MIN_LOWER_STORED = 50          # smallest T_en in the pre-registered grid


def top_train() -> dict:
    tr = pd.read_parquet(hf_hub_download("WillHeld/hinglish_top", "data/train-00000-of-00001-edd2764b201a5c1f.parquet", repo_type="dataset"))
    counts: Counter = Counter()
    in_person: Counter = Counter()
    for parse in tr.cs_parse:
        stack: list[str] = []
        for tok in str(parse).split():
            if tok.startswith("[") and ":" in tok:
                stack.append(tok[1:])
                continue
            if tok == "]":
                if stack:
                    stack.pop()
                continue
            for w in re.findall(r"[a-z]+", tok.lower()):
                counts[w] += 1
                if any(s in PERSON_SLOTS for s in stack):
                    in_person[w] += 1
    return {"rows": len(tr), "counts": counts, "in_person": in_person}


def wikitext() -> dict:
    """WikiText-103 (raw), first train shard only: Wikipedia text with letter case kept. Licence on the dataset card: CC BY-SA 3.0 and GFDL."""
    df = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet", repo_type="dataset"))
    lower: Counter = Counter()
    title: Counter = Counter()
    n_tokens = 0
    for line in df["text"]:
        for w in re.findall(r"[A-Za-z]+", line):
            if len(w) < 3:
                continue
            n_tokens += 1
            if w.islower():
                lower[w] += 1
            elif w[0].isupper() and w[1:].islower():
                title[w.lower()] += 1
    return {"lower": lower, "title": title, "n_tokens": n_tokens, "rows": len(df)}


def main() -> None:
    t = top_train()
    e = wikitext()
    english = [[w, c, round(c / (c + e["title"][w]), 4)] for w, c in e["lower"].items() if c >= MIN_LOWER_STORED]
    english.sort(key=lambda x: -x[1])
    # [word, count, share of its occurrences inside a person/contact slot]. A token is excluded later when that share is >= 0.5.
    # (The plan said "ever inside a person slot", which also removed function words such as ko, mujhe, apna that sit inside a slot span
    # now and then; amended before any evaluation, see docs/decisions.md Day 3.13.)
    hinglish = [[w, c, round(t["in_person"][w] / c, 3)] for w, c in t["counts"].items()]
    hinglish.sort(key=lambda x: -x[1])
    out = {
        "meta": {
            "english": {"source": "WikiText-103 raw, train shard 0 of 2 (Salesforce/wikitext)", "license": "CC BY-SA 3.0 and GFDL (Wikipedia text; per the dataset card)",
                        "rows": e["rows"], "tokens_counted": e["n_tokens"], "min_lower_stored": MIN_LOWER_STORED, "n": len(english)},
            "hinglish": {"source": "Hinglish-TOP train split (WillHeld/hinglish_top)", "license": "Apache-2.0", "train_rows": t["rows"], "person_slots": sorted(PERSON_SLOTS),
                         "n_token_types": len(t["counts"]), "n_majority_person_types": sum(1 for h in hinglish if h[2] >= 0.5), "exclude_if_person_share_at_least": 0.5},
        },
        "english": english,
        "hinglish": hinglish,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("english stored:", len(english), "| hinglish token types:", len(hinglish), "| majority-person types:", sum(1 for h in hinglish if h[2] >= 0.5), "->", OUT)


if __name__ == "__main__":
    main()
