"""Generate a self-contained HTML gallery of harvested pairs for visual review.

Groups: POSITIVES (oracle same-type), HARD negatives (cross-type, NCC>=HARD_LO),
EASY negatives (cross-type, low NCC). Each card shows the two crops side-by-side
(upscaled), the level, the colour-NCC, and the category. Images are base64-
embedded so the single index.html is portable. Serve the output dir over HTTP.
"""
from __future__ import annotations

import base64
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image

import dsio
from gallery import color_ncc
from build_dataset import (_UF, _ncc_matrix, _clusters, LINK_THR, HARD_LO,
                           PAIR_NCC_MIN)

UP = 4  # upscale factor for display


def _b64(a, b):
    A = cv2.resize(a, (40 * UP, 40 * UP), interpolation=cv2.INTER_NEAREST)
    B = cv2.resize(b, (40 * UP, 40 * UP), interpolation=cv2.INTER_NEAREST)
    gap = np.full((40 * UP, 6 * UP, 3), 230, dtype=np.uint8)
    combo = np.concatenate([A, gap, B], axis=1)
    buf = io.BytesIO()
    Image.fromarray(combo).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def collect(n_pos=80, n_discard=40, n_hard=80, n_easy=40, seed=0):
    rng = random.Random(seed)
    pos, disc, hard, easy = [], [], [], []
    for _, sh in dsio.iter_harvest_shards():
        pairs = sh["pairs"]
        P = pairs.shape[0]
        level = int(sh["level"])
        crops = pairs.reshape(P * 2, 40, 40, 3).astype(np.uint8)
        n = len(crops)
        ncc = _ncc_matrix(crops)
        # garbage-filtered must-links: only trust reliable oracle pairs
        uf = _UF(n)
        for k in range(P):
            v = float(ncc[2 * k, 2 * k + 1])
            (pos if v >= PAIR_NCC_MIN else disc).append(
                (crops[2 * k], crops[2 * k + 1], v, level))
            if v >= PAIR_NCC_MIN:
                uf.union(2 * k, 2 * k + 1)
        clust = _clusters(P, ncc, uf)
        iu, ju = np.where(clust[None] != clust[:, None])
        for i, j in zip(iu.tolist(), ju.tolist()):
            if i < j:
                v = float(ncc[i, j])
                if v >= HARD_LO:
                    hard.append((crops[i], crops[j], v, level))
                elif v < 0.15:
                    easy.append((crops[i], crops[j], v, level))
    rng.shuffle(pos); rng.shuffle(disc); rng.shuffle(hard); rng.shuffle(easy)
    return pos[:n_pos], disc[:n_discard], hard[:n_hard], easy[:n_easy]


def card(a, b, ncc, level, cat):
    border = {"pos": "#2e7d32", "hard": "#c62828", "easy": "#1565c0",
              "disc": "#9e9e9e"}[cat]
    name = {"pos": "SAME (kept)", "hard": "HARD-neg (diff)",
            "easy": "EASY-neg (diff)", "disc": "DISCARDED (misdetect)"}[cat]
    return (f'<div class="card" style="border-color:{border};opacity:'
            f'{"0.65" if cat == "disc" else "1"}">'
            f'<img src="data:image/png;base64,{_b64(a, b)}"/>'
            f'<div class="meta"><b>{name}</b><br>L{level} · NCC={ncc:.3f}</div>'
            f'</div>')


def build(out_dir):
    dsio.ensure_dirs(out_dir)
    pos, disc, hard, easy = collect()
    print(f"[gallery] sampled pos={len(pos)} discarded={len(disc)} "
          f"hard={len(hard)} easy={len(easy)}")
    sections = [
        ("POSITIVES — oracle same-type pairs, NCC&ge;%.2f (kept for training)" % PAIR_NCC_MIN,
         "pos", pos),
        ("DISCARDED — raw oracle pairs with NCC&lt;%.2f (harvest mis-detections; "
         "excluded from positives AND from clustering)" % PAIR_NCC_MIN, "disc", disc),
        (f"HARD NEGATIVES — different cluster, NCC&ge;{HARD_LO:.2f} (confusable; "
         f"clusters are garbage-filtered so these should be genuinely different)", "hard", hard),
        ("EASY NEGATIVES — different cluster, low NCC (obviously different)", "easy", easy),
    ]
    body = []
    for title, cat, items in sections:
        body.append(f"<h2>{title} <small>({len(items)})</small></h2>")
        body.append('<div class="grid">')
        for a, b, ncc, level in items:
            body.append(card(a, b, ncc, level, cat))
        body.append("</div>")
    counts = sum(len(s[2]) for s in sections)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>AutoConnect harvest review</title><style>"
        "body{font-family:system-ui,sans-serif;background:#1e1e1e;color:#ddd;margin:20px}"
        "h2{border-bottom:1px solid #444;padding-bottom:6px;margin-top:30px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}"
        ".card{background:#2b2b2b;border:3px solid;padding:8px;border-radius:8px;text-align:center}"
        ".card img{width:100%;height:auto;image-rendering:pixelated;background:#000;border-radius:4px}"
        ".meta{font-size:12px;margin-top:6px;color:#bbb}"
        "small{color:#888;font-weight:normal}"
        "</style></head><body>"
        f"<h1>Harvested pair review (garbage-filtered)</h1>"
        f"<p>{counts} sampled pairs from {sum(1 for _ in dsio.iter_harvest_shards())} boards. "
        "Crops upscaled 4&times; (nearest-neighbour). NCC = translation-tolerant colour NCC.</p>"
        + "".join(body) + "</body></html>")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"[gallery] wrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "gallery"))
    a = ap.parse_args()
    build(a.out)
