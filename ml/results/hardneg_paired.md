# Hard-negative round: paired analysis (local, from the saved raw scores)

Per seed, same rows, each model at its own frozen block threshold. `only baseline` / `only arm` = discordant rows (flagged by one model only); p = exact McNemar (two-sided). Seeds are not independent, so p-values are per seed and not combined.

## B_neg_plus_attacks vs 0_baseline

| outcome | n | baseline k (3 seeds) | arm k (3 seeds) | discordant only-baseline / only-arm (3 seeds) | McNemar p (3 seeds) |
|---|---|---|---|---|---|
| E1 human Hinglish benign (FP) | 104 | 9 / 28 / 13 | 2 / 1 / 1 | 9-2 / 27-0 / 13-1 | 0.065 / 0.000 / 0.002 |
| E1 English control (FP) | 104 | 9 / 1 / 9 | 0 / 0 / 0 | 9-0 / 1-0 / 9-0 | 0.004 / 1.000 / 0.004 |
| E2 prompt-talk benign (FP) | 100 | 25 / 31 / 33 | 1 / 2 / 1 | 24-0 / 29-0 / 32-0 | 0.000 / 0.000 / 0.000 |
| E3 Hinglish attacks (caught) | 24 | 19 / 24 / 20 | 23 / 24 / 22 | 1-5 / 0-0 / 2-4 | 0.219 / 1.000 / 0.688 |
| test attacks (recall) | 208 | 178 / 178 / 164 | 185 / 185 / 176 | 2-9 / 0-7 / 4-16 | 0.065 / 0.016 / 0.012 |
| test benign (FP) | 717 | 13 / 12 / 11 | 14 / 14 / 17 | 4-5 / 3-5 / 3-9 | 1.000 / 0.727 / 0.146 |
| indirect_test attacks (recall) | 55 | 50 / 51 / 44 | 51 / 51 / 52 | 1-2 / 1-1 / 1-9 | 1.000 / 1.000 / 0.021 |
| indirect_test benign (FP) | 57 | 1 / 0 / 0 | 0 / 1 / 1 | 1-0 / 0-1 / 0-1 | 1.000 / 1.000 / 1.000 |
| tricky attacks (recall) | 21 | 14 / 17 / 13 | 17 / 18 / 16 | 1-4 / 1-2 / 1-4 | 0.375 / 1.000 / 0.375 |
| tricky benign (FP) | 28 | 8 / 8 / 5 | 2 / 2 / 1 | 6-0 / 6-0 / 4-0 | 0.031 / 0.031 / 0.125 |
| xstest_safe (FP) | 250 | 0 / 0 / 0 | 0 / 0 / 0 | 0-0 / 0-0 / 0-0 | 1.000 / 1.000 / 1.000 |
| long benign docs (FP, sliding) | 114 | 8 / 11 / 3 | 13 / 6 / 9 | 4-9 / 6-1 / 0-6 | 0.267 / 0.125 / 0.031 |
| long middle injections (recall, sliding) | 38 | 34 / 36 / 31 | 35 / 37 / 36 | 0-1 / 1-2 / 0-5 | 1.000 / 1.000 / 0.062 |

Test F1 (point estimate, paired bootstrap 95% CI of arm minus baseline), per seed:

- seed 42: baseline 0.892, arm 0.909, diff +0.017 [-0.005, +0.040]
- seed 43: baseline 0.894, arm 0.909, diff +0.015 [-0.003, +0.034]
- seed 44: baseline 0.856, arm 0.878, diff +0.021 [-0.007, +0.051]
- block thresholds (baseline -> arm, 3 seeds): 0.942, 0.961, 0.979 -> 0.948, 0.858, 0.964

## A_negatives vs 0_baseline

| outcome | n | baseline k (3 seeds) | arm k (3 seeds) | discordant only-baseline / only-arm (3 seeds) | McNemar p (3 seeds) |
|---|---|---|---|---|---|
| E1 human Hinglish benign (FP) | 104 | 9 / 28 / 13 | 0 / 0 / 0 | 9-0 / 28-0 / 13-0 | 0.004 / 0.000 / 0.000 |
| E1 English control (FP) | 104 | 9 / 1 / 9 | 1 / 1 / 1 | 8-0 / 1-1 / 8-0 | 0.008 / 1.000 / 0.008 |
| E2 prompt-talk benign (FP) | 100 | 25 / 31 / 33 | 0 / 0 / 1 | 25-0 / 31-0 / 32-0 | 0.000 / 0.000 / 0.000 |
| E3 Hinglish attacks (caught) | 24 | 19 / 24 / 20 | 0 / 1 / 1 | 19-0 / 23-0 / 19-0 | 0.000 / 0.000 / 0.000 |
| test attacks (recall) | 208 | 178 / 178 / 164 | 185 / 176 / 183 | 2-9 / 5-3 / 1-20 | 0.065 / 0.727 / 0.000 |
| test benign (FP) | 717 | 13 / 12 / 11 | 23 / 14 / 26 | 3-13 / 4-6 / 0-15 | 0.021 / 0.754 / 0.000 |
| indirect_test attacks (recall) | 55 | 50 / 51 / 44 | 50 / 43 / 51 | 2-2 / 8-0 / 1-8 | 1.000 / 0.008 / 0.039 |
| indirect_test benign (FP) | 57 | 1 / 0 / 0 | 2 / 0 / 1 | 0-1 / 0-0 / 0-1 | 1.000 / 1.000 / 1.000 |
| tricky attacks (recall) | 21 | 14 / 17 / 13 | 11 / 11 / 13 | 5-2 / 7-1 / 4-4 | 0.453 / 0.070 / 1.000 |
| tricky benign (FP) | 28 | 8 / 8 / 5 | 1 / 2 / 2 | 7-0 / 7-1 / 3-0 | 0.016 / 0.070 / 0.250 |
| xstest_safe (FP) | 250 | 0 / 0 / 0 | 0 / 0 / 0 | 0-0 / 0-0 / 0-0 | 1.000 / 1.000 / 1.000 |
| long benign docs (FP, sliding) | 114 | 8 / 11 / 3 | 11 / 2 / 13 | 3-6 / 9-0 / 0-10 | 0.508 / 0.004 / 0.002 |
| long middle injections (recall, sliding) | 38 | 34 / 36 / 31 | 35 / 32 / 36 | 0-1 / 5-1 / 0-5 | 1.000 / 0.219 / 0.062 |

Test F1 (point estimate, paired bootstrap 95% CI of arm minus baseline), per seed:

- seed 42: baseline 0.892, arm 0.889, diff -0.003 [-0.028, +0.022]
- seed 43: baseline 0.894, arm 0.884, diff -0.010 [-0.032, +0.011]
- seed 44: baseline 0.856, arm 0.878, diff +0.021 [-0.009, +0.053]
- block thresholds (baseline -> arm, 3 seeds): 0.942, 0.961, 0.979 -> 0.704, 0.703, 0.601

## 0_baseline@valB vs 0_baseline

| outcome | n | baseline k (3 seeds) | arm k (3 seeds) | discordant only-baseline / only-arm (3 seeds) | McNemar p (3 seeds) |
|---|---|---|---|---|---|
| E1 human Hinglish benign (FP) | 104 | 9 / 28 / 13 | 0 / 0 / 0 | 9-0 / 28-0 / 13-0 | 0.004 / 0.000 / 0.000 |
| E1 English control (FP) | 104 | 9 / 1 / 9 | 0 / 0 / 0 | 9-0 / 1-0 / 9-0 | 0.004 / 1.000 / 0.004 |
| E2 prompt-talk benign (FP) | 100 | 25 / 31 / 33 | 0 / 1 / 1 | 25-0 / 30-0 / 32-0 | 0.000 / 0.000 / 0.000 |
| E3 Hinglish attacks (caught) | 24 | 19 / 24 / 20 | 0 / 1 / 0 | 19-0 / 23-0 / 20-0 | 0.000 / 0.000 / 0.000 |
| test attacks (recall) | 208 | 178 / 178 / 164 | 48 / 70 / 68 | 130-0 / 108-0 / 96-0 | 0.000 / 0.000 / 0.000 |
| test benign (FP) | 717 | 13 / 12 / 11 | 0 / 0 / 0 | 13-0 / 12-0 / 11-0 | 0.000 / 0.000 / 0.001 |
| indirect_test attacks (recall) | 55 | 50 / 51 / 44 | 10 / 14 / 7 | 40-0 / 37-0 / 37-0 | 0.000 / 0.000 / 0.000 |
| indirect_test benign (FP) | 57 | 1 / 0 / 0 | 0 / 0 / 0 | 1-0 / 0-0 / 0-0 | 1.000 / 1.000 / 1.000 |
| tricky attacks (recall) | 21 | 14 / 17 / 13 | 0 / 2 / 2 | 14-0 / 15-0 / 11-0 | 0.000 / 0.000 / 0.001 |
| tricky benign (FP) | 28 | 8 / 8 / 5 | 0 / 0 / 1 | 8-0 / 8-0 / 4-0 | 0.008 / 0.008 / 0.125 |
| xstest_safe (FP) | 250 | 0 / 0 / 0 | 0 / 0 / 0 | 0-0 / 0-0 / 0-0 | 1.000 / 1.000 / 1.000 |
| long benign docs (FP, sliding) | 114 | 8 / 11 / 3 | 0 / 0 / 0 | 8-0 / 11-0 / 3-0 | 0.008 / 0.001 / 0.250 |
| long middle injections (recall, sliding) | 38 | 34 / 36 / 31 | 1 / 4 / 2 | 33-0 / 32-0 / 29-0 | 0.000 / 0.000 / 0.000 |

Test F1 (point estimate, paired bootstrap 95% CI of arm minus baseline), per seed:

- seed 42: baseline 0.892, arm 0.375, diff -0.517 [-0.597, -0.443]
- seed 43: baseline 0.894, arm 0.504, diff -0.391 [-0.462, -0.323]
- seed 44: baseline 0.856, arm 0.493, diff -0.364 [-0.437, -0.297]
- block thresholds (baseline -> arm, 3 seeds): 0.942, 0.961, 0.979 -> 0.999, 0.999, 0.998

## orig_labels vs 0_baseline

| outcome | n | baseline k (3 seeds) | arm k (3 seeds) | discordant only-baseline / only-arm (3 seeds) | McNemar p (3 seeds) |
|---|---|---|---|---|---|
| E1 human Hinglish benign (FP) | 104 | 9 / 28 / 13 | 7 / 19 / 17 | 3-1 / 9-0 / 4-8 | 0.625 / 0.004 / 0.388 |
| E1 English control (FP) | 104 | 9 / 1 / 9 | 7 / 1 / 8 | 3-1 / 0-0 / 4-3 | 0.625 / 1.000 / 1.000 |
| E2 prompt-talk benign (FP) | 100 | 25 / 31 / 33 | 26 / 27 / 29 | 2-3 / 6-2 / 6-2 | 1.000 / 0.289 / 0.289 |
| E3 Hinglish attacks (caught) | 24 | 19 / 24 / 20 | 16 / 24 / 21 | 3-0 / 0-0 / 1-2 | 0.250 / 1.000 / 1.000 |
| test attacks (recall) | 208 | 178 / 178 / 164 | 173 / 160 / 180 | 10-5 / 20-2 / 0-16 | 0.302 / 0.000 / 0.000 |
| test benign (FP) | 717 | 13 / 12 / 11 | 15 / 9 / 19 | 3-5 / 5-2 / 0-8 | 0.727 / 0.453 / 0.008 |
| indirect_test attacks (recall) | 55 | 50 / 51 / 44 | 50 / 52 / 53 | 2-2 / 1-2 / 0-9 | 1.000 / 1.000 / 0.004 |
| indirect_test benign (FP) | 57 | 1 / 0 / 0 | 1 / 0 / 1 | 0-0 / 0-0 / 0-1 | 1.000 / 1.000 / 1.000 |
| tricky attacks (recall) | 21 | 14 / 17 / 13 | 19 / 17 / 17 | 0-5 / 1-1 / 0-4 | 0.062 / 1.000 / 0.125 |
| tricky benign (FP) | 28 | 8 / 8 / 5 | 6 / 6 / 6 | 2-0 / 2-0 / 1-2 | 0.500 / 0.500 / 1.000 |
| xstest_safe (FP) | 250 | 0 / 0 / 0 | 0 / 0 / 0 | 0-0 / 0-0 / 0-0 | 1.000 / 1.000 / 1.000 |
| long benign docs (FP, sliding) | 114 | 8 / 11 / 3 | 9 / 3 / 5 | 2-3 / 9-1 / 0-2 | 1.000 / 0.021 / 0.500 |
| long middle injections (recall, sliding) | 38 | 34 / 36 / 31 | 35 / 30 / 35 | 0-1 / 6-0 / 0-4 | 1.000 / 0.031 / 0.125 |

Test F1 (point estimate, paired bootstrap 95% CI of arm minus baseline), per seed:

- seed 42: baseline 0.892, arm 0.874, diff -0.018 [-0.043, +0.007]
- seed 43: baseline 0.894, arm 0.849, diff -0.046 [-0.075, -0.017]
- seed 44: baseline 0.856, arm 0.885, diff +0.028 [+0.004, +0.055]
- block thresholds (baseline -> arm, 3 seeds): 0.942, 0.961, 0.979 -> 0.967, 0.984, 0.868

## Long benign documents (sliding), false positives out of 114

- 0_baseline: [8, 11, 3] of 114 (mean FPR 0.064)
- A_negatives: [11, 2, 13] of 114 (mean FPR 0.076)
- B_neg_plus_attacks: [13, 6, 9] of 114 (mean FPR 0.082)
