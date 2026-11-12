"""Head+tail tokenization shared by training (notebook), export parity checks and the benchmark.

Identical to `encode_head_tail` in ml/train_transformer.ipynb: first HEAD and last TAIL content tokens, wrapped as
[CLS] ... [SEP], so at most 512 tokens (the position limit). The server scanner and the browser layer must reproduce
this exactly (Transformers.js truncates the head only by default).
"""
HEAD, TAIL = 256, 254


def encode_head_tail(tok, texts):
    assert tok.cls_token_id is not None and tok.sep_token_id is not None
    ids_all = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in ids_all:
        if len(ids) > HEAD + TAIL:
            ids = ids[:HEAD] + ids[-TAIL:]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out
