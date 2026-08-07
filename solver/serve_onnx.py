"""Export the trained PairNet to ONNX and benchmark CPU inference latency vs NCC.

The runtime bot currently scores every candidate pair with colour NCC
(``gallery.color_ncc``); the trained NN must be servable on CPU within a 3x latency
budget of NCC. We export the net to ONNX (onnxruntime, CPUExecutionProvider) and
measure per-pair latency for: NCC, torch-CPU, and ONNX-CPU (single-pair and
batched). Also reports agreement so the ONNX model is verified against torch.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn

import dsio
from pairnet import PairNet, CANON, load_pairnet
from gallery import color_ncc


class _ServeWrapper(nn.Module):
    """PairNet -> same-type probability (sigmoid), for a clean ONNX output."""
    def __init__(self, pn: PairNet):
        super().__init__()
        self.pn = pn

    def forward(self, a, b):
        return torch.sigmoid(self.pn(a, b))


def _latest_model():
    cands = sorted(f for f in os.listdir(dsio.MODELS_DIR)
                   if f.startswith("pairnet_") and f.endswith(".pt"))
    return os.path.join(dsio.MODELS_DIR, cands[-1]) if cands else None


def export_onnx(model_path=None, onnx_path=None):
    model_path = model_path or _latest_model()
    assert model_path and os.path.exists(model_path), "no trained model"
    onnx_path = onnx_path or os.path.splitext(model_path)[0] + ".onnx"
    pn, cfg = load_pairnet(model_path)
    wrap = _ServeWrapper(pn).eval()
    a = torch.randn(1, 3, CANON, CANON)
    b = torch.randn(1, 3, CANON, CANON)
    torch.onnx.export(
        wrap, (a, b), onnx_path,
        input_names=["a", "b"], output_names=["prob"],
        dynamic_axes={"a": {0: "B"}, "b": {0: "B"}, "prob": {0: "B"}},
        opset_version=17,
    )
    print(f"[serve] exported -> {onnx_path}")
    return onnx_path


def _load_pairs(n=2000):
    ca, cb = [], []
    for f in sorted(os.listdir(dsio.DATASET_DIR)):
        if not f.startswith("shard_test_") or not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(dsio.DATASET_DIR, f))
        ca.append(d["ca"]); cb.append(d["cb"])
    if not ca:
        raise SystemExit("no test shard; run build_dataset.py")
    ca = np.concatenate(ca)[:n].astype(np.uint8)
    cb = np.concatenate(cb)[:n].astype(np.uint8)
    return ca, cb


def _time(fn, n_warm=20, n_iter=None):
    n_iter = n_iter or 200
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def benchmark(onnx_path, n=1000):
    import onnxruntime as ort
    ca, cb = _load_pairs(n)
    n = len(ca)

    # NCC per-pair
    def ncc_one(i):
        return color_ncc(ca[i], cb[i])
    ncc_lat = _time(lambda: ncc_one(np.random.randint(n)), n_iter=200)

    # torch CPU per-pair
    pn, cfg = load_pairnet(_latest_model())
    with torch.no_grad():
        def torch_one(i):
            a = torch.from_numpy(ca[i:i+1].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            b = torch.from_numpy(cb[i:i+1].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            return torch.sigmoid(pn(a, b)).item()
        torch_lat = _time(lambda: torch_one(np.random.randint(n)), n_iter=200)

    # ONNX CPU -- two sessions: single-thread (lowest dispatch overhead, best
    # for batch=1) and default multi-thread (best throughput for big batches).
    def make_sess(threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])

    a_all = (ca.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    b_all = (cb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    sess1 = make_sess(1)
    def onnx_one(i):
        return float(np.ravel(sess1.run(None, {"a": a_all[i:i+1], "b": b_all[i:i+1]})[0])[0])
    onnx_lat = _time(lambda: onnx_one(np.random.randint(n)), n_iter=200)

    # ONNX batched (amortised per-pair) -- the realistic bot path (many pairs)
    sessN = make_sess(0)   # 0 = use all cores
    B = 128
    def onnx_batch():
        i = np.random.randint(0, n - B)
        return sessN.run(None, {"a": a_all[i:i+B], "b": b_all[i:i+B]})[0]
    onnx_batch_lat = _time(onnx_batch, n_iter=50) / B

    # agreement: ONNX vs torch on a sample
    with torch.no_grad():
        a = torch.from_numpy(a_all[:256]); b = torch.from_numpy(b_all[:256])
        torch_prob = torch.sigmoid(pn(a, b)).numpy().ravel()
    onnx_prob = sess1.run(None, {"a": a_all[:256], "b": b_all[:256]})[0].ravel()
    mae = float(np.abs(torch_prob - onnx_prob).mean())

    print(f"\n[serve] CPU latency per pair (n={n}, {CANON}x{CANON}):")
    print(f"  NCC (colour, cv2)          : {ncc_lat*1e6:8.1f} us/pair")
    print(f"  PairNet torch CPU (batch=1): {torch_lat*1e6:8.1f} us/pair  ({torch_lat/ncc_lat:.1f}x NCC)")
    print(f"  PairNet ONNX CPU (batch=1) : {onnx_lat*1e6:8.1f} us/pair  ({onnx_lat/ncc_lat:.1f}x NCC)")
    print(f"  PairNet ONNX CPU (batch={B}) : {onnx_batch_lat*1e6:8.1f} us/pair  ({onnx_batch_lat/ncc_lat:.1f}x NCC)")
    print(f"  ONNX-vs-torch prob MAE: {mae:.6f}")
    bound = onnx_lat / ncc_lat
    print(f"\n  -> ONNX batch=1 is {bound:.2f}x NCC "
          f"({'WITHIN' if bound <= 3.0 else 'EXCEEDS'} the 3x budget)")
    return dict(ncc_us=ncc_lat*1e6, onnx_us=onnx_lat*1e6,
                onnx_batch_us=onnx_batch_lat*1e6, ratio_batch1=bound,
                ratio_batch=onnx_batch_lat/ncc_lat, mae=mae)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    onnx_path = export_onnx(a.model)
    if not a.export_only:
        benchmark(onnx_path, a.n)
