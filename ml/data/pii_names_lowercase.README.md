# pii_names_lowercase.jsonl

**RETIRED AS A TEST SET (2026-11-18, docs/decisions.md Day 3.13): this set is "seen".** It was used for the round-1 verdict (Day 3.12) and its failures were read in detail, so it is now only a *development* set (parameter choice for round 2). Do not use it to prove anything built after seeing it; round 2's test set is `pii_names_lowercase_r2.jsonl`.

**AI-drafted (Claude), not reviewed by the project owner.** Before retirement: evaluation only, never used to write a word list, a rule or a threshold, and never trained on.

Written by Claude (Sonnet 5.5) for docs/decisions.md Day 3.12, **after** the lowercase-name approaches (frontend/src/pii/nameApproaches.ts) were committed and frozen.
The same assistant wrote the approaches and these sentences, which is a bias: it knows which cues it implemented. To limit it, 26 of the 60 positives (`no_cue`) are
written without any of the implemented cue words, and 12 (`ambiguous`) use names that are also ordinary words; results are reported per group.

- `set` = `pos` (sentence contains person names, all written in lowercase) or `neg` (no person name).
- `category`, positives: `cue` (typical cue such as "tell X", "X ne"), `no_cue` (no cue), `ambiguous` (name is also a common word: joy, rose, grace, deep...).
- `category`, negatives: `common_word` (words that can also be names), `place`, `brand`, `date_festival`, `hinglish_plain`, `generic`.
- `names`: the person-name strings as written (lowercase); every occurrence in the sentence is gold.
- One known labelling choice: "ganesh chaturthi" (a festival) and "eid mubarak" are negatives, although "ganesh" and "mubarak" are also given names.
