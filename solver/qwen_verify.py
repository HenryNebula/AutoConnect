"""Independent label-quality check using the local qwen3.5-4b vision model
(OpenAI-compatible API on :8000).

The harvest/build pipeline labels pairs from the oracle + NCC structure. To check
that labels (especially HARD negatives -- cross-type but visually similar) are not
systematically wrong, we sample pairs, render each as a side-by-side image, and
ask the VLM "are these the same Pokémon icon?". Agreement with our labels is an
independent estimate of label noise (the VLM never saw NCC or the oracle).

Usage:  python solver/qwen_verify.py [--n 60]
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image

import dsio

API = os.environ.get("QWEN_API", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("QWEN_MODEL", "qwen3.5-4b")
CANON = dsio.CANON


def _montage_b64(a: np.ndarray, b: np.ndarray) -> str:
    """Side-by-side RGB montage of two crops -> base64 JPEG."""
    gap = np.full((CANON, 4, 3), 255, dtype=np.uint8)
    combo = np.concatenate([a, gap, b], axis=1)
    img = Image.fromarray(combo)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _ask(a, b):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_montage_b64(a, b)}"}},
                {"type": "text", "text":
                 "These are two game tiles shown side by side. Do they depict the "
                 "EXACT SAME Pokémon character/icon (ignoring tiny position jitter)? "
                 "Answer with one word: SAME or DIFFERENT."},
            ]}],
        "max_tokens": 5, "temperature": 0.0,
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    txt = resp["choices"][0]["message"]["content"].strip().upper()
    return 1 if txt.startswith("SAME") else (0 if txt.startswith("DIFF") else -1)


def _load_pairs(split="test", limit=400, seed=0):
    rng = random.Random(seed)
    rows = []
    for f in sorted(os.listdir(dsio.DATASET_DIR)):
        if not (f.startswith(f"shard_{split}_") and f.endswith(".npz")):
            continue
        d = np.load(os.path.join(dsio.DATASET_DIR, f))
        for i in range(len(d["label"])):
            rows.append((d["ca"][i], d["cb"][i], int(d["label"][i]), int(d["kind"][i])))
    rng.shuffle(rows)
    return rows[:limit]


def verify(n=60, seed=0):
    rows = _load_pairs(split="test", limit=2000, seed=seed)
    if not rows:
        raise SystemExit("no dataset; run build_dataset.py first")
    # over-sample hard negatives (kind==1) so the check is meaningful
    hard = [r for r in rows if r[3] == 1]
    other = [r for r in rows if r[3] != 1]
    rng = random.Random(seed)
    rng.shuffle(hard); rng.shuffle(other)
    sample = (hard[: n // 2] + other[: n - n // 2])[:n]
    print(f"[qwen] checking {len(sample)} pairs ({sum(1 for r in sample if r[3]==1)} hard-neg) "
          f"vs {MODEL}")

    agree = 0
    per_kind = {0: [0, 0], 1: [0, 0], 2: [0, 0]}   # kind -> [agree, total]
    for a, b, label, kind in sample:
        try:
            pred = _ask(a, b)
        except Exception as ex:
            print(f"  query failed: {repr(ex)[:80]}; skip")
            continue
        if pred < 0:
            continue
        ok = (pred == label)
        agree += ok
        per_kind[kind][0] += ok
        per_kind[kind][1] += 1
    tot = sum(v[1] for v in per_kind.values())
    print(f"[qwen] overall agreement: {agree}/{tot} = {agree/max(tot,1):.3f}")
    for k, name in ((0, "easy-neg"), (1, "hard-neg"), (2, "positive")):
        a, t = per_kind[k]
        if t:
            print(f"  {name:9s}: {a}/{t} = {a/t:.3f}")
    print("[qwen] (hard-neg agreement < 1.0 indicates residual label noise -- expected & OK)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    verify(a.n, a.seed)
