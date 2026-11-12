> **Superseded (2026-11-12):** measured before the template-cluster drop; current numbers are in `augmentation_comparison.md` and `metrics.json` (`baselines`).

### Cleaned labels (test n=953, 218 attacks)

| model | C | thr | P | R | F1 | FPR | ROC-AUC | PR-AUC | recall @ val-FPR≤1% (test FPR) | tricky | xstest FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| length_lr | 0.1 | 0.185 | 0.255 | 0.894 | **0.397** | 0.776 | 0.584 | 0.290 | 0.014 (0.007) | 16/49 | 93/250 |
| word_tfidf | 100 | 0.179 | 0.864 | 0.904 | **0.883** | 0.042 | 0.984 | 0.958 | 0.817 (0.012) | 39/49 | 0/250 |
| wordchar | 10 | 0.499 | 0.941 | 0.798 | **0.864** | 0.015 | 0.986 | 0.961 | 0.812 (0.018) | 32/49 | 0/250 |
| minilm_emb | 10 | 0.559 | 0.921 | 0.798 | **0.855** | 0.020 | 0.968 | 0.935 | 0.761 (0.011) | 32/49 | 2/250 |

### Original labels, same rows

| model | C | thr | P | R | F1 | FPR | ROC-AUC | PR-AUC | recall @ val-FPR≤1% (test FPR) | tricky | xstest FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| length_lr | 0.1 | 0.191 | 0.261 | 0.895 | **0.404** | 0.756 | 0.578 | 0.286 | 0.018 (0.007) | 16/49 | 83/250 |
| word_tfidf | 100 | 0.368 | 0.900 | 0.822 | **0.859** | 0.027 | 0.973 | 0.932 | 0.744 (0.015) | 35/49 | 0/250 |
| wordchar | 10 | 0.327 | 0.897 | 0.831 | **0.863** | 0.029 | 0.974 | 0.934 | 0.726 (0.012) | 34/49 | 0/250 |
| minilm_emb | 10 | 0.557 | 0.912 | 0.758 | **0.828** | 0.022 | 0.954 | 0.909 | 0.689 (0.014) | 36/49 | 5/250 |

### Tricky set per category, cleaned-label models (correct/total)

| category (label) | word_tfidf | wordchar | minilm_emb |
|---|---|---|---|
| benign_mentions_instructions (benign, 5) | 1/5 | 2/5 | 2/5 |
| benign_document (benign, 6) | 5/6 | 5/6 | 4/6 |
| benign_suspicious (benign, 6) | 6/6 | 6/6 | 6/6 |
| long_benign (benign, 5) | 5/5 | 5/5 | 4/5 |
| short_attack (attack, 5) | 5/5 | 3/5 | 2/5 |
| sneaky_attack (attack, 6) | 6/6 | 4/6 | 5/6 |
| hidden_injection_document (attack, 5) | 2/5 | 0/5 | 0/5 |
| hinglish (mixed, 11) | 9/11 | 7/11 | 9/11 |
