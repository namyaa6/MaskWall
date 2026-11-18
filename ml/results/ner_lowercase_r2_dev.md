# Lowercase names round 2: development run (round-1 set, now "seen"; the sealed round-2 set was NOT loaded)

Baseline A on the development set: 11/66 lowercase names, 0 flagged harmless sentences; existing sample capitalised recall 46/49, 5 false-positive spans.

| T_en | K_h | share | words | dev lowercase recall (66) | dev flagged harmless (of 60) | existing capitalised recall | existing FP spans | meets limits |
|---|---|---|---|---|---|---|---|---|
| 50 | 2 | 0.5 | 19846 | 49 | 10 | 47/49 | 11 | no |
| 50 | 2 | 0.8 | 17473 | 49 | 10 | 47/49 | 11 | no |
| 50 | 5 | 0.5 | 19487 | 49 | 12 | 47/49 | 13 | no |
| 50 | 5 | 0.8 | 17033 | 49 | 12 | 47/49 | 13 | no |
| 500 | 2 | 0.5 | 6042 | 49 | 10 | 47/49 | 11 | no |
| 500 | 2 | 0.8 | 5450 | 49 | 10 | 47/49 | 11 | no |
| 500 | 5 | 0.5 | 5616 | 49 | 12 | 47/49 | 13 | no |
| 500 | 5 | 0.8 | 4964 | 49 | 12 | 47/49 | 13 | no |

**Chosen and frozen: T_en = 500, K_h = 2, share = 0.8** (NO setting met the limits on the development set: the one with the fewest flagged sentences).
