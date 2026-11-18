"""Prompt-injection / jailbreak scanner: the exported int8 ONNX classifier (MiniLM or DeBERTa-v3-small, fine-tuned on
train + indirect-injection augmentation; see docs/decisions.md Day 1.8 and 1.11).

Model folder (made by backend/scripts/prepare_model.py): model.int8.onnx, tokenizer.json, model_meta.json.
Uses `tokenizers` + `onnxruntime` only (no torch / transformers in the serving image). Inputs: input_ids, attention_mask
(batch size 1); output: logits; score = softmax(logits)[1] = P(attack).

Scan modes (see windowing.py): head_tail (one 512-token window) or sliding (overlapping windows, capped, max score).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .windowing import CONTENT, covered_tokens, head_tail_ids, select_windows, window_spans


@dataclass
class WindowResult:
    start: int
    end: int
    score: float
    ms: float


@dataclass
class ScanResult:
    score: float                      # P(attack), max over scanned windows
    coverage: str                     # "full" | "partial" (some tokens were never scanned)
    mode: str
    n_tokens: int
    n_windows_needed: int
    windows: list[WindowResult] = field(default_factory=list)
    tokenize_ms: float = 0.0
    inference_ms: float = 0.0
    scanned_tokens: int = 0

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_windows"] = self.n_windows
        return d


def _softmax_p1(logits: np.ndarray) -> float:
    z = logits - logits.max()
    e = np.exp(z)
    return float(e[1] / e.sum())


class InjectionScanner:
    def __init__(self, meta: dict, tokenizer, session, mode: str = "sliding", max_windows: int = 6):
        if mode not in ("head_tail", "sliding"):
            raise ValueError(f"unknown scan mode {mode!r}")
        self.meta, self.tokenizer, self.session = meta, tokenizer, session
        self.mode, self.max_windows = mode, max_windows
        self.cls_id, self.sep_id = int(meta["cls_id"]), int(meta["sep_id"])
        self.thr_f1 = float(meta["thr_f1"])
        self.thr_fpr1pct = float(meta["thr_fpr1pct"])

    @classmethod
    def load(cls, model_dir: Path | str, *, threads: int = 2, mode: str = "sliding", max_windows: int = 6) -> "InjectionScanner":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        meta_path = d / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found: run backend/scripts/prepare_model.py first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(str(d / meta["tokenizer_file"]))
        tok.no_truncation()   # truncation is ours (windowing.py), never the tokenizer's
        tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(d / meta["onnx_file"]), so, providers=["CPUExecutionProvider"])
        return cls(meta, tok, sess, mode=mode, max_windows=max_windows)

    # ---------------------------------------------------------------- inference
    def _score_ids(self, content_ids: list[int]) -> float:
        ids = np.array([[self.cls_id, *content_ids, self.sep_id]], dtype=np.int64)
        logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        return _softmax_p1(logits)

    def scan(self, text: str) -> ScanResult:
        t0 = time.perf_counter()
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids if text.strip() else []
        tok_ms = (time.perf_counter() - t0) * 1000
        n = len(ids)
        if n == 0:
            return ScanResult(0.0, "full", self.mode, 0, 0, [], tok_ms, 0.0, 0)

        if self.mode == "head_tail":
            spans, needed, partial = [(0, min(n, CONTENT))], 1, n > CONTENT
            batches = [head_tail_ids(ids)]
            scanned = len(batches[0])
        else:
            all_spans = window_spans(n)
            needed = len(all_spans)
            spans, partial = select_windows(all_spans, self.max_windows)
            batches = [ids[s:e] for s, e in spans]
            scanned = covered_tokens(spans)

        windows: list[WindowResult] = []
        for (s, e), content in zip(spans, batches):
            t = time.perf_counter()
            p = self._score_ids(content)
            windows.append(WindowResult(s, e, p, (time.perf_counter() - t) * 1000))
        return ScanResult(
            score=max(w.score for w in windows), coverage="partial" if partial else "full", mode=self.mode,
            n_tokens=n, n_windows_needed=needed, windows=windows, tokenize_ms=tok_ms,
            inference_ms=sum(w.ms for w in windows), scanned_tokens=scanned,
        )
