"""Accuracy and latency vs token window (head+tail, proportional split) for the two int8 models; thresholds tuned on val per window.
Run from the repo root after export_quantize.py: python ml/window_eval.py  -> ml/results/window_eval.json. Test numbers are informational only."""
import sys, json, time
sys.path.insert(0,"ml")
import numpy as np, pandas as pd
import benchmark as bm, export_quantize as eq, train_baseline as tb
from transformers import AutoTokenizer
PROC=eq.ROOT/"ml/data/processed"
sets={n:pd.read_parquet(PROC/f"{n}.parquet") for n in ["val","test","indirect_val"]}
def enc_w(tok,texts,W):
    head=round((W-2)*256/510); tail=(W-2)-head
    ids_all=tok(list(texts),add_special_tokens=False,truncation=False)["input_ids"]
    out=[]; trunc=0
    for ids in ids_all:
        if len(ids)>W-2: ids=ids[:head]+ids[-tail:]; trunc+=1
        out.append([tok.cls_token_id]+ids+[tok.sep_token_id])
    return out,trunc/len(ids_all)
res={}
for job in ["cleaned_aug/minilm","cleaned_aug/deberta_v3_small"]:
    tok=AutoTokenizer.from_pretrained(str(eq.CKPT/job)); s=bm.session(eq.ONNX_DIR/job/"int8.onnx")
    for W in [128,256,384,512]:
        sc={}; ms=[]; tr={}
        for n,d in sets.items():
            enc,tr[n]=enc_w(tok,d.text.tolist(),W); v=[]
            for e in enc:
                ids=np.array([e],dtype=np.int64); t=time.perf_counter()
                lg=s.run(["logits"],{"input_ids":ids,"attention_mask":np.ones_like(ids)})[0][0]
                if n=="val": ms.append((time.perf_counter()-t)*1000)
                v.append(eq.softmax_p1(lg))
            sc[n]=np.array(v)
        yv=sets["val"].label.to_numpy(); thr,vf1=tb.tune_f1_threshold(yv,sc["val"])
        iv=tb.rates(sets["indirect_val"].label.to_numpy(),sc["indirect_val"]>=thr); te=tb.rates(sets["test"].label.to_numpy(),sc["test"]>=thr)
        r=dict(W=W,thr=round(thr,3),val_f1=round(vf1,3),iv_recall=round(iv["recall"],3),iv_fpr=round(iv["fpr"],3),test_f1=round(te["f1"],3),test_fpr=round(te["fpr"],3),
               val_truncated=round(tr["val"],3),lat_p50=round(float(np.percentile(ms,50)),1),lat_p95=round(float(np.percentile(ms,95)),1),lat_mean=round(float(np.mean(ms)),1))
        res.setdefault(job,[]).append(r); print(job,r,flush=True)
json.dump(res,open("ml/results/window_eval.json","w"),indent=2)
print("DONE")
