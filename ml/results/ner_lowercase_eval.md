# Lowercase names: approaches compared (browser, WASM)

Protocol and rule: docs/decisions.md Day 3.12 (committed before any approach, evaluation sentence or measurement existed). Sets: ml/data/pii_names_lowercase.jsonl (AI-drafted, not reviewed by the owner); ml/data/pii_names_sample.jsonl (AI-drafted, not reviewed by the owner).
Headless Chromium, Transformers.js 3.8.1, single-threaded WASM, warm; latency = per sentence over all 182 sentences (includes re-casing and cue rules).

| approach | lowercase recall (66 names) | cue | no_cue | ambiguous | false-positive sentences on 60 negatives | existing: capitalised recall (49) | existing: false-positive spans | median / p95 ms |
|---|---|---|---|---|---|---|---|---|
| A. NER as typed (baseline) | **11/66** | 4/23 | 6/28 | 1/15 | 0 | 46/49 | 5 | 29.2 / 65.8 |
| B. NER on a selectively re-cased copy | **48/66** | 18/23 | 27/28 | 3/15 | 13 | 47/49 | 7 | 30.6 / 48 |
| C. NER + context cues | **29/66** | 21/23 | 6/28 | 2/15 | 5 | 49/49 | 9 | 33.2 / 50.7 |
| E. B + C | **53/66** | 23/23 | 27/28 | 3/15 | 15 | 49/49 | 11 | 31.5 / 47.6 |

## Pre-registered rule

| approach | gain over A (points) | no recall drop | FP negatives (<= A+3 and <= 6) | FP existing (<= A+2) | median <= 300 ms | gain >= 15 | eligible |
|---|---|---|---|---|---|---|---|
| B | 56.1 | yes | NO | yes | yes | yes | **no** |
| C | 27.3 | yes | NO | NO | yes | yes | **no** |
| E | 63.6 | yes | NO | NO | yes | yes | **no** |

**Selected by the rule: none (browser keeps approach A)**

## False positives on the negatives

**A. NER as typed (baseline)** (0 sentences, 0 spans): none

**B. NER on a selectively re-cased copy** (13 sentences, 15 spans): #65 "pooja", #81 "jaipur", #83 "barish", #94 "karu", #95 "swiggy", #95 "zomato", #98 "holi", #101 "eid mubarak", #102 "ganesh chaturthi", #105 "khatam", #108 "pahunchna", #109 "peene", #109 "mann", #111 "acchi", #112 "jeeta"

**C. NER + context cues** (5 sentences, 5 spans): #98 "holi", #103 "navratri", #105 "khatam", #109 "peene", #113 "emi"

**E. B + C** (15 sentences, 17 spans): #65 "pooja", #81 "jaipur", #83 "barish", #94 "karu", #95 "swiggy", #95 "zomato", #98 "holi", #101 "eid mubarak", #102 "ganesh chaturthi", #103 "navratri", #105 "khatam", #108 "pahunchna", #109 "peene", #109 "mann", #111 "acchi", #112 "jeeta", #113 "emi"

## Missed lowercase names

**A. NER as typed (baseline)** (55): #1 priya, #2 rahul, #3 neha, #4 amit, #6 kavya, #7 arjun mehta, #8 suresh, #9 meera, #10 vikram, #12 simran, #13 ananya, #14 kiran, #15 karan, #17 pooja, #18 deepak, #20 gaurav, #21 lakshmi, #22 harpreet, #22 nikhil, #23 rohan, #26 kavya, #28 divya, #29 nikhil, #30 gaurav, #31 tanvi, #32 priya, #33 rahul, #34 sunita, #35 meera, #36 manish, #36 shruti, #38 john miller, #39 david, #39 emily, #40 sarah, #41 daniel, #44 venkatesh, #45 aisha khan, #46 sneha, #47 harpreet kaur, #48 ryan, #49 joy, #51 grace, #52 deep, #53 mark, #54 will, #55 faith, #55 hope, #56 summer, #57 pooja, #58 priya, #58 joy, #59 ruby, #59 pearl, #60 amber

**B. NER on a selectively re-cased copy** (18): #6 kavya, #8 suresh, #14 kiran, #17 pooja, #21 lakshmi, #41 daniel, #49 joy, #51 grace, #52 deep, #53 mark, #54 will, #55 faith, #55 hope, #56 summer, #58 joy, #59 ruby, #59 pearl, #60 amber

**C. NER + context cues** (37): #22 harpreet, #22 nikhil, #23 rohan, #26 kavya, #28 divya, #29 nikhil, #30 gaurav, #31 tanvi, #32 priya, #33 rahul, #34 sunita, #35 meera, #36 manish, #36 shruti, #38 john miller, #39 david, #39 emily, #40 sarah, #41 daniel, #44 venkatesh, #45 aisha khan, #46 sneha, #47 harpreet kaur, #48 ryan, #49 joy, #51 grace, #52 deep, #53 mark, #54 will, #55 faith, #55 hope, #56 summer, #58 priya, #58 joy, #59 ruby, #59 pearl, #60 amber

**E. B + C** (13): #41 daniel, #49 joy, #51 grace, #52 deep, #53 mark, #54 will, #55 faith, #55 hope, #56 summer, #58 joy, #59 ruby, #59 pearl, #60 amber

## Existing sample: false positives and misses of capitalised names

**A. NER as typed (baseline)**: FP #25 "Mera", #30 "Meri", #32 "aur", #43 "Regards", #44 "Ms"; missed (capitalised) #28 Pooja Mehta, #29 Sanjay, #34 Anjali

**B. NER on a selectively re-cased copy**: FP #29 "bana sakte", #30 "shaadi agle mahine", #32 "rehte", #34 "kaha", #43 "Regards", #44 "Ms", #56 "beginner"; missed (capitalised) #28 Pooja Mehta, #29 Sanjay

**C. NER + context cues**: FP #25 "Mera", #29 "bana sakte", #34 "Mummy", #34 "kaha", #36 "gaane", #43 "Regards", #44 "Ms", #59 "Mumbai", #59 "Pune"; missed (capitalised) none

**E. B + C**: FP #29 "bana sakte", #30 "shaadi agle mahine", #32 "rehte", #34 "Mummy", #34 "kaha", #36 "gaane", #43 "Regards", #44 "Ms", #56 "beginner", #59 "Mumbai", #59 "Pune"; missed (capitalised) none
