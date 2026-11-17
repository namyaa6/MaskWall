# Browser name (NER) model: measurement and selection

Protocol and rule: docs/decisions.md Day 3.2 (written before this ran). Sample: ml/data/pii_names_sample.jsonl (AI-drafted, not reviewed by the owner), 62 sentences, 56 names.
Headless Chromium, Transformers.js 3.8.1, WASM, q8 weights; crossOriginIsolated=false (single-threaded WASM), 8 logical cores. Latency = per sentence, warm run, after one warm-up.

| model | q8 MB | cold load s | warm load s | median ms | p95 ms | strict recall | any-overlap recall | false positives (on no-name sentences) | eligible |
|---|---|---|---|---|---|---|---|---|---|
| `onnx-community/distilbert-NER-ONNX` | 65.8 | 113.7 | 0.6 | 39.9 | 56.7 | 47/56 | 48/56 | 5 (0) | yes |
| `Xenova/bert-base-NER` | 109 | 39.4 | 0.8 | 72.9 | 100.8 | 48/56 | 49/56 | 4 (1) | yes |
| `Xenova/distilbert-base-multilingual-cased-ner-hrl` | 135.4 | 51.1 | 1.1 | 56.9 | 82.1 | 41/56 | 41/56 | 0 (0) | no |

**Selected by the rule: `onnx-community/distilbert-NER-ONNX`**

## Strict recall by category

| model | english | indian_english | hinglish | lowercase | quoted_document | word_names | no_names |
|---|---|---|---|---|---|---|---|
| `onnx-community/distilbert-NER-ONNX` | 12/12 | 16/16 | 10/13 (fp 3) | 1/7 | 6/6 (fp 2) | 2/2 | 0/0 |
| `Xenova/bert-base-NER` | 12/12 | 16/16 | 11/13 (fp 2) | 1/7 | 6/6 (fp 1) | 2/2 | 0/0 (fp 1) |
| `Xenova/distilbert-base-multilingual-cased-ner-hrl` | 12/12 | 16/16 | 6/13 | 0/7 | 6/6 | 1/2 | 0/0 |

## Missed names and false positives

**onnx-community/distilbert-NER-ONNX**

- missed (9): #28 Pooja Mehta, #29 Sanjay, #34 Anjali, #37 priya, #38 rahul sharma, #39 john, #41 mark, #42 arun, #42 meera
- false positives (5): #25 "Mera", #30 "Meri", #32 "aur", #43 "Regards", #44 "Ms"

**Xenova/bert-base-NER**

- missed (8): #26 Neha, #30 Divya, #38 rahul sharma, #39 john, #40 sunita, #41 mark, #42 arun, #42 meera
- false positives (4): #25 "Mera", #30 "Meri", #45 "me", #57 "Mahindra"

**Xenova/distilbert-base-multilingual-cased-ner-hrl**

- missed (15): #26 Neha, #27 Amit, #28 Pooja Mehta, #29 Sanjay, #30 Divya, #32 Simran, #32 Karan, #37 priya, #38 rahul sharma, #39 john, #40 sunita, #41 mark, #42 arun, #42 meera, #48 Will
- false positives (0): none
