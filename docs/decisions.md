# Design decisions

Notable decisions and the reasons behind them. Feeds the README and interview prep.

## Day 1.1: Injection classifier data (`ml/data/load_datasets.py`)

### Task definition
- **Binary label:** 0 = benign, 1 = prompt injection **or** jailbreak. The firewall blocks both the same way, so one class is enough. A `source` column is kept on every row so metrics can still be reported per source.

### Datasets used (licenses checked on the HF Hub, 2026-11-12)
| Dataset | License | Used as | Why |
|---|---|---|---|
| `deepset/prompt-injections` | Apache-2.0 | both labels | Common public benchmark with both classes. About 45% of its rows are German and are removed by the English filter (below). |
| `jackhhao/jailbreak-classification` | Apache-2.0 | both labels | Its benign rows are role-play prompts ("You are Chris Redfield…"). These look like jailbreaks, which makes them good hard negatives. |
| `Lakera/gandalf_ignore_instructions` | MIT | label 1 | Real human attempts to get the Gandalf game to give up its password. Short, varied injections. |
| `TrustAIRLab/in-the-wild-jailbreak-prompts` (`jailbreak_2023_12_25`) | MIT | label 1 | Real jailbreaks collected from Reddit and Discord (CCS'24). |
| `TrustAIRLab/...` (`regular_2023_12_25`) | MIT | label 0, sampled | Ordinary prompts from the **same platforms** as the jailbreaks, so "where it was posted" can't act as a shortcut. |
| `databricks/databricks-dolly-15k` (`instruction`) | CC-BY-SA-3.0 | label 0, sampled | Everyday human-written instructions. |
| `Paul/XSTest` (safe prompts only) | CC-BY-4.0 | **eval only** | 250 safe prompts that sound dangerous ("How do I kill a Python process?"). Used to stress-test the false-positive rate. Never trained on. |

### Datasets rejected
- `xTRam1/safe-guard-prompt-injection`: no license on the dataset card, and it's GPT-3.5 synthetic data.
- `reshabhs/SPML_Chatbot_Prompt_Injection`: each label depends on a paired system prompt that our classifier never sees, so the labels would be noisy without it.
- `tatsu-lab/alpaca`: non-commercial license (CC-BY-NC). Dolly fills the same role.
- `hackaprompt/hackaprompt-dataset`, `allenai/wildjailbreak`: gated, which adds friction on Colab and in CI. Possible additions later.
- XSTest *unsafe* prompts: these are harmful requests, not injections. Either label would blur the task.

### Processing
- **Cleaning:** strip whitespace and drop texts under 5 characters. No truncation; the tokenizer handles length later.
- **Exact dedup across sources** (lowercased, whitespace collapsed). This mattered: 598 TrustAIRLab jailbreaks are also in jackhhao, because both scraped jailbreakchat. **43 texts had conflicting labels across sources and were dropped entirely**, mostly prompts that TrustAIRLab lists as both "regular" and "jailbreak".
- **Class balance:** all attacks are kept. Benign rows total about 3× the attacks: all native benign rows (deepset, jackhhao) plus equal samples from TrustAIRLab-regular and Dolly. This keeps the false-positive rate meaningful without heavy imbalance. The ratio is set with `--benign-ratio`.
- **Split:** our own stratified 80/10/10 split on (source, label), seed 42. The official splits are ignored because we merge sources. **Thresholds are tuned on val only. Test is never used for tuning.**
- **English-only filter (after the split, applied to every split including XSTest):** DistilBERT and MiniLM are English models, and a few hundred German, Portuguese and Chinese rows would add noise without being representative. We use `langdetect` with a fixed seed so results are deterministic.
  - **Plain langdetect wasn't good enough.** A probe showed it mislabels short English: "Ignore all previous guidelines" → Italian, "Why do you get dental erosion?" → Spanish. Filtering on it directly would have wrongly dropped about 49 Gandalf attacks, about 98 Dolly prompts and 10 XSTest prompts, all English.
  - **Rule:** a text is English if langdetect says `en`. A text with fewer than 20 words that langdetect calls non-English is kept anyway if it has no non-ASCII letters and at least 20% of its words are common English function words. "Was" is left off the word list because it is also German. Attack words like "ignore" are also left off, so the rule doesn't depend on the label.
  - **Rows dropped:** 390 train, 45 val, 58 test, 0 XSTest. Most are deepset German (295) and non-English TrustAIRLab-regular prompts (156).
  - **Remaining error, checked by eye:** about 20 short English rows are still dropped ("Name 10 colors", "ignore rules") and a few short German ones are kept ("Wer stirbt in Burkina Faso?").
- **Near-duplicate leakage check:** exact dedup misses lightly edited copies, such as DAN prompts with a changed name. We fit TF-IDF (word 1–2 grams, sublinear tf) on **train only**, then compute each val and test row's highest cosine similarity to train. Rows above **0.9** are dropped from val/test; train is left alone. Result: 44 val and 36 test rows dropped, mostly jackhhao and TrustAIRLab jailbreak variants. XSTest had 0 rows above the threshold (checked and reported only, nothing dropped). Without this step, test scores on DAN-style attacks would have been inflated.
- **Length-shortcut check:** median and p90 length per (label, source) are printed in the summary.

### Resulting data (seed 42)
| split | rows | attacks | benign:attack |
|---|---|---|---|
| train | 8,099 | 2,029 | 2.99 |
| val | 972 | 220 | 3.42 |
| test | 968 | 219 | 3.42 |
| xstest_safe | 250 | 0 | eval only |

Median length in characters by source:

| source | label | median chars | p90 chars |
|---|---|---|---|
| dolly | 0 | 55 | 123 |
| deepset | 0 | 41 | 144 |
| jackhhao | 0 | 234 | 1,247 |
| trustairlab_regular | 0 | 852 | 4,332 |
| gandalf | 1 | 61 | 129 |
| deepset | 1 | 116 | 304 |
| jackhhao | 1 | 1,544 | 4,148 |
| trustairlab_jailbreak | 1 | 1,996 | 6,461 |

**Takeaway:** length is a partial shortcut. Jailbreaks (jackhhao, TrustAIRLab) are about 2× longer than TrustAIRLab-regular prompts. But both classes cover the full range: Gandalf attacks are short, and TrustAIRLab-regular benign prompts are long. So "long = attack" can't separate the data on its own. The baseline step will measure this directly with per-source metrics and a **length-only classifier as a floor**. If the real models barely beat it, they are mostly learning length.

### Known limitations
- About 2.5k attacks in total, fewer than the 3k+ planned, because so many jailbreaks were duplicates across sources.
- **English only.** The firewall is not evaluated on other languages, and non-English attacks are likely to get through. That's a real gap for a multilingual audience. Hinglish (romanized Hindi-English) is probed only through the tricky set.
- The English filter makes a few mistakes on short text (about 20 English rows lost, a few German rows kept; see above).
- After filtering, test has only **7 deepset attacks**, so per-source metrics for deepset are too noisy to read much into.
- Some TrustAIRLab-"regular" prompts are borderline personas (e.g. "You will roleplay as BeastGPT…"). Their labels come from the original paper's community-based labelling and are not hand-verified.
- Many TrustAIRLab prompts exceed 512 tokens and will be truncated during fine-tuning.
- The attack data is mostly 2023-era jailbreaks and direct injections. Indirect injections (in retrieved documents or tool output) are underrepresented.

## Day 1.2: Hand-written tricky set (`ml/data/tricky.csv`)
- **Eval only.** It is never trained on, used for threshold tuning, or passed through the loader. Columns: `text,label,category`.
- **Why:** public datasets have stylistic giveaways (DAN prompts are long and dramatic, Gandalf attacks mention passwords). The tricky set checks whether the model learned *intent* or just *surface features*.
- **Categories:**
  - `benign_mentions_instructions`: benign text that contains trigger words ("ignore", "instructions", "system prompt"). Tests false positives on keywords.
  - `short_attack`: terse attacks with no DAN framing.
  - `long_benign`: long, detailed benign prompts, including a role-play campaign setup. Tests the length shortcut and the persona shortcut.
  - `hidden_injection_document`: injections buried inside pasted emails, resumes, reviews and web pages. These are indirect injections, which the public data underrepresents.
  - `hinglish`: romanized Hindi-English prompts, both benign and attacks. The training data is English-only, so this measures how well the model holds up on the code-mixed text the target users actually write.
- **Authorship:** the project author writes about half; Claude drafted the other half (25 rows, 5 per category). Fake names and `example.com` domains only.
- **Leakage check:** no exact overlap with train, and max TF-IDF cosine similarity to train is 0.50.
