# Lowercase names, round 2: ONE evaluation on the sealed Gemini set (browser, WASM)

Protocol and rule: docs/decisions.md Day 3.13 (committed before the set was generated). Frozen word-list setting: T_en 500, K_h 2, share 0.8 (5450 words plus the NER vocabulary).

| approach | lowercase names (60) | cue | no_cue | word_name | harmless sentences flagged (of 60) | existing capitalised recall (49) | existing FP spans | median / p95 ms |
|---|---|---|---|---|---|---|---|---|
| A. NER as typed (baseline) | **10/60** | 3/34 | 4/14 | 2/11 | 0 | 46/49 | 5 | 129.3 / 204 |
| B2. NER on a re-cased copy, independent word list | **48/60** | 31/34 | 14/14 | 2/11 | 16 | 47/49 | 11 | 123.4 / 179.7 |
| C2. NER + context cues (info only) | **37/60** | 30/34 | 4/14 | 2/11 | 7 | 49/49 | 8 | 124.9 / 187.9 |
| E2. B2 + C2 (info only) | **49/60** | 32/34 | 14/14 | 2/11 | 20 | 49/49 | 14 | 125.7 / 183 |

## B2 against the pre-registered rule

| check | limit | B2 | passes |
|---|---|---|---|
| harmless sentences flagged | <= 6 of 60 | 16 | NO |
| capitalised-name recall on the existing sample | >= A (46) | 47 | yes |
| false-positive spans on the existing sample | <= A + 2 (7) | 11 | NO |
| median latency | < 300 ms | 123.4 ms | yes |
| gain in lowercase recall over A | >= 15 points | 63.3 points | yes |

**B2 is NOT eligible: nothing ships, negative result.**

## Flagged harmless sentences and missed names

**A. NER as typed (baseline)**: FP on negatives (0 sentences): none; missed lowercase names (50): #1 ananya, #2 rahul, #4 rose, #5 arjun, #6 priyanka, #7 fatima, #8 vikram sharma, #9 karthik, #10 joy, #11 deep, #12 karan, #15 harpreet singh, #16 rahul, #17 sneha, #18 arjun, #19 priya, #20 amit, #21 deep, #22 meera, #23 rohan, #24 kavita, #26 sanjay, #27 joy, #28 anjali, #29 vikram, #30 neha, #31 sarah, #32 john smith, #33 rahul, #34 emily, #35 mark, #36 priya, #37 alice, #38 rohan, #39 michael johnson, #40 grace, #41 david, #42 anjali, #43 joy, #44 charlie, #45 sophie, #46 arjun, #48 rahul, #49 rose, #51 aman, #52 vikram, #54 priya, #55 karan, #56 joy, #58 kabir

**B2. NER on a re-cased copy, independent word list**: FP on negatives (16 sentences): #68 "delhi", #76 "bhai", #78 "zomato", #79 "diwali", #79 "sabne", #81 "mahina", #84 "dekhi", #85 "jayenge", #86 "saste", #86 "hain", #87 "wahan", #89 "chalo rakhi", #89 "hain", #94 "itne", #94 "kharch", #94 "diye", #97 "pls", #105 "bhul", #108 "pls", #117 "samajhne", #120 "shadi"; missed lowercase names (12): #4 rose, #10 joy, #11 deep, #21 deep, #27 joy, #30 neha, #35 mark, #40 grace, #41 david, #43 joy, #49 rose, #56 joy

**C2. NER + context cues (info only)**: FP on negatives (7 sentences): #78 "zomato", #79 "diwali", #80 "iphone", #83 "holi", #88 "uber", #109 "muze delhi", #120 "shadi"; missed lowercase names (23): #2 rahul, #4 rose, #6 priyanka, #9 karthik, #10 joy, #11 deep, #15 harpreet singh, #18 arjun, #20 amit, #21 deep, #23 rohan, #27 joy, #32 john smith, #35 mark, #37 alice, #39 michael johnson, #40 grace, #41 david, #43 joy, #44 charlie, #49 rose, #54 priya, #56 joy

**E2. B2 + C2 (info only)**: FP on negatives (20 sentences): #68 "delhi", #76 "bhai", #78 "zomato", #79 "diwali", #79 "sabne", #80 "iphone", #81 "mahina", #83 "holi", #84 "dekhi", #85 "jayenge", #86 "saste", #86 "hain", #87 "wahan", #88 "uber", #89 "chalo rakhi", #89 "hain", #94 "itne", #94 "kharch", #94 "diye", #97 "pls", #105 "bhul", #108 "pls", #109 "muze delhi", #117 "samajhne", #120 "shadi"; missed lowercase names (11): #4 rose, #10 joy, #11 deep, #21 deep, #27 joy, #35 mark, #40 grace, #41 david, #43 joy, #49 rose, #56 joy

