# pii_names_lowercase_r2.jsonl

**Round-2 TEST set for lowercase-name detection (docs/decisions.md Day 3.13). SEALED. AI-drafted by Gemini (`gemini-3.1-flash-lite`), not reviewed by the project owner.**

Generated and filtered by `ml/data/gen_lowercase_gemini.py` **before any word list was built and before any approach was run on it**. It must not be used for any word list,
parameter, rule or code; its only job is the single evaluation described in the plan. Prompts and raw Gemini responses are archived in `ml/data/lowercase_gemini/`.

- 60 positives (`set` = `pos`; lowercase person names) and 60 negatives (`set` = `neg`; no person name). Gemini tagged each positive itself:
  `cue` (a word like tell/call/ask/hi/"ne"/"ko"/"ji" next to the name), `no_cue`, `word_name` (the name is also an everyday word: rose, joy, deep, grace, mark...). Gemini's tags, not mine.
- Selection: the first 15 rows that survive the mechanical filters from each of the 4 prompts per side (all lowercase, every name an exact whole-word substring, no duplicates
  of the round-1 set, the existing names sample or a Hinglish-TOP *train* query (token overlap >= 0.8)). A first version took the first 60 in call order; that would have dropped most
  of the `no_cue` prompt, and was changed before any measurement.
- Two positives were dropped by hand after skimming all 120 rows, as label errors ("will" the modal verb, not a name): `lowercase_gemini/manual_drops.json`. Every drop is in `lowercase_gemini/drops.json`.
- Known limits: one generator, one prompt style; Gemini's "no_cue" sentences often still contain words such as "see" or "with"; the names are typical ones (rahul, priya, joy appear several times).
