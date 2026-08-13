"""Legitimate runtime bot: perceive the board by CV (a trained PairNet tile
classifier), solve with my own <=2-turn connectivity algorithm, and act via the
SWF's ``acRemovePair`` ExternalInterface handle (a CDP-click fallback exists but
is rarely needed). No game state is read for the pairing DECISION -- only the
game's tilesLeft count, as an acceptance signal.

The grid is detected once per level and tiles are snapshotted once: the level
"movement" mechanics are layout-only (tiles sit on a fixed lattice and do NOT
drift mid-level), so the per-move sim matrix + classifier reuse that snapshot,
masking removed cells via the present mask. Each removal is confirmed by
tilesLeft dropping; a non-drop means a mis-classification -- that pair is
remembered as "known different" and the bot re-picks. On a genuine deadlock (no
progress for a stretch) the SWF reshuffles (acPlayOne -> createNewMap) and the
board is re-snapshotted.

Level/game transitions are driven as control handles (see ``advance``): the
patched SWF's ExternalInterface callbacks (acSetEnabled/acReset/acStep) are used.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import cdp
import vision
import conn
import gameio
try:
    import conn_fast as _cf   # compiled rollout/exact lookahead (solver/cpp/build.sh)
except Exception:             # noqa: BLE001
    _cf = None
import gallery as galmod

CANON = galmod.CANON
DEFAULT_BG = np.array([53.0, 44.0, 26.0], dtype=np.float32)


def canon(crop):
    return cv2.resize(crop, (CANON, CANON), interpolation=cv2.INTER_AREA)


class Bot:
    def __init__(self, gallery_path=None, verbose=True, transition_mode="ei",
                 model_path=None, io=None):
        self.verbose = verbose
        self.grid = None
        self.bg = DEFAULT_BG.copy()
        self.gallery = galmod.GalleryClassifier(gallery_path, max_templates=8) if gallery_path else None
        self.known_diff = set()
        self.click_settle = 0.20
        self.verify_wait = 0.40
        self.gravity_settle = 0.6   # wait for L2+ per-level gravity (dongzuo) to
                                    # settle after a removal before re-snapping
        # Game I/O backend: CDPGameIO for the local-Chrome rig, WSGameIO for the
        # LAN browser backend. The brain calls self.io.* and never imports cdp
        # for game I/O (only the click-cell fallback still uses cdp directly).
        # See solver/gameio.py.
        self.io = io if io is not None else gameio.CDPGameIO(verify_wait=self.verify_wait)
        self.transition_mode = transition_mode   # "ei" (dev) or "click" (vanilla)
        self.empty_template = None     # learned empty-slot crop (CANON-sized)
        self.empty_thr = 0.55          # NCC >= this => cell is an empty slot
        self.std_thr = 60.0            # icon-std floor (learned at level start)
        self._recover_count = 0        # deadlock-reshuffle attempts this level
        self.use_remove_pair = None    # cached io.has_ei("acRemovePair"); None=unprobed
        # Learned same-type tile classifier (issue #3): the trained PairNet is a
        # drop-in scoring function for gallery.color_ncc. If no model is present
        # (or torch is missing) the bot falls back to colour-NCC, unchanged.
        self.nn = self._maybe_load_nn(model_path)
        self.use_nn = self.nn is not None and getattr(self.nn, "available", False)
        self.cand_thr = 0.5 if self.use_nn else 0.55   # NN prob vs NCC score
        # C++ lookahead policy (solver/cpp/conn_fast.cpp); no-op when _cf is None
        self.rollout_k = 40        # Monte-Carlo rollouts per candidate move
        self.rollout_m = 6         # top-N candidates evaluated by lookahead
        self.exact_tiles = 20      # <= this many tiles -> exact solvable (else rollout)
        self.safe_thr = 0.10       # rollout clear-rate floor: below this a move
                                   # is treated as deadlock-prone and skipped
        self.use_rollout = False   # C++ rollout lookahead (needs reshuffle +
                                   # stable session to benefit live); False keeps
                                   # the proven greedy + _ncc_safe path
        self.max_reshuffles = 6    # deadlock recoveries per level (acPlayOne ->
                                   # SWF createNewMap) before giving up
        self._cf_counter = 0       # per-call seed source for reproducible rollouts
        self._sim = None            # cached all-pairs tile-sim matrix (NN only)
        self._sim_tiles = None      # id(self.cur_tiles) the cache belongs to
        # Demo / visualization (--demo): paint a DOM bbox overlay on each chosen
        # pair in the live browser window, pause, THEN clear (acRemovePair is
        # instant, so the pause must precede it for the boxes to be visible).
        # Cosmetic only -- every overlay call is try/except-guarded so it can
        # never break a run. See _demo_show / _demo_pair / _demo_hide.
        self.demo = False          # paint the per-pair bbox overlay before clears
        self.demo_pause = 0.7      # seconds each pair's boxes stay on screen
        self.demo_flash = True     # flash a box over every grid cell at level start
        self.save_frames = False   # also write annotated PNGs to $tmp/demo/
        self._frame_idx = 0

    @staticmethod
    def _default_siamese_model():
        """Pick a trained siamese PairNet (NOT a SupCon variant -- those
        over-merge types). Prefer 'micro' (lowest latency, fewest hard
        false-negatives), then 'default'."""
        try:
            import dsio
            d = dsio.MODELS_DIR
        except Exception:  # noqa: BLE001
            return None
        for name in ("pairnet_micro_auc1.000.pt", "pairnet_default_auc1.000.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    def _maybe_load_nn(self, model_path):
        path = model_path or os.environ.get("AC_BOT_MODEL") or self._default_siamese_model()
        if not path or not os.path.exists(path):
            if self.verbose:
                print("[bot] no NN tile model found; using colour-NCC fallback")
            return None
        try:
            from gallery_nn import NNClassifier
            nn = NNClassifier(path)
        except Exception as e:  # noqa: BLE001  (torch missing, corrupt ckpt, ...)
            if self.verbose:
                print(f"[bot] NN model load failed ({e!r}); using colour-NCC fallback")
            return None
        if self.verbose and nn.available:
            print(f"[bot] NN tile classifier loaded: {os.path.basename(path)}")
        return nn

    def _ensure_sim(self):
        """Pre-compute the all-pairs tile-similarity matrix for the current board
        once per level, from whichever backbone is active: a batched NN forward
        (self.nn) or a pairwise colour-NCC matrix. Either way subsequent per-pair
        lookups in _pick_move / the rollout lookahead are cheap array indexing --
        this is what makes the vision backbone swappable."""
        tiles = self.cur_tiles
        if tiles is None or tiles is self._sim_tiles:
            return
        self._sim_tiles = tiles
        R, C = tiles.shape[:2]
        crops = np.stack([canon(tiles[r, c]) for r in range(R) for c in range(C)])
        if self.use_nn and self.nn is not None:
            # The NN was trained on dim (~mean 50) harvest crops, but the live
            # capture is bright (~mean 170) because the index.html veil isn't
            # compositing into Page.captureScreenshot. Re-normalize to the
            # training brightness so the NN is in-distribution (a no-op when the
            # board is already dim). NCC is brightness-invariant -> no dim needed.
            m = float(crops.mean())
            if m > 1:
                crops = (crops.astype(np.float32) * (50.0 / m)).clip(0, 255).astype(np.uint8)
            self._sim = self.nn.sim_matrix(crops).reshape(R, C, R, C)
        else:
            self._sim = self._ncc_matrix(crops, R, C)

    def _ncc_matrix(self, crops, R, C):
        """All-pairs colour-NCC similarity as an (R,C,R,C) float32 matrix, built
        once per level. Makes the NCC backbone as cheap to query as the NN
        (matrix-indexed _pair_sim + rollout lookahead) -- ~0.2s for a 96-tile
        board, amortised over the whole level."""
        n = len(crops)
        flat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                v = galmod.color_ncc(crops[i], crops[j])
                flat[i, j] = flat[j, i] = v
        return flat.reshape(R, C, R, C)

    def _pair_sim(self, a, b):
        """Same-type score in [0,1] for cells a=(r,c), b=(r,c), from the cached
        sim matrix (NN probability or colour-NCC, whichever backbone is active)."""
        if self._sim is not None:
            return float(self._sim[a[0], a[1], b[0], b[1]])
        return galmod.color_ncc(canon(self.cur_tiles[a[0], a[1]]),
                                canon(self.cur_tiles[b[0], b[1]]))

    # ---- low level --------------------------------------------------------
    def snap(self):
        return self.io.capture()

    def click_cell(self, r, c):
        x, y = float(self.grid["xs"][c]), float(self.grid["ys"][r])
        cdp.click(x, y, settle=self.click_settle)

    def click_xy(self, x, y):
        cdp.click(x, y, settle=self.click_settle)

    def establish_grid(self, retries=14):
        if self.grid is not None:
            return True
        for _ in range(retries):
            img = self.snap()
            g = vision.detect_grid(img)
            if g and (g["cols"], g["rows"]) == (12, 8):
                self.grid = g
                return True
            time.sleep(0.3)
        return False

    def reset_grid(self):
        self.grid = None

    # ---- perception -------------------------------------------------------
    def estimate_bg(self, img):
        """Per-frame background colour.

        Empty cells are low-variance and share the (drifting) board background
        colour. Among low-variance cells we cluster by colour; the background is
        the largest cluster, provided it has >= ``min_cluster`` members. The
        threshold is set above the largest possible single Pokémon-type cluster
        (<=8 per type), so a cluster of dark icons is never mistaken for the
        background -- only the growing set of emptied cells qualifies.
        """
        stds, mean_rgb = vision.cell_stats(img, self.grid)
        low = stds < 9.0
        n_low = int(low.sum())
        if n_low < 3:
            return
        cand = mean_rgb[low]                        # (n_low, 3)
        min_cluster = 9
        best_color = None
        best_size = 0
        used = np.zeros(n_low, dtype=bool)
        for i in range(n_low):
            if used[i]:
                continue
            d = np.abs(cand - cand[i]).sum(1) < 22.0
            sz = int(d.sum())
            if sz > best_size:
                best_size, best_color = sz, cand[d].mean(0)
            used |= d
        if best_color is not None and best_size >= min_cluster:
            self.bg = 0.5 * self.bg + 0.5 * best_color

    def _empty_ncc(self, img, r, c):
        """NCC of a cell's crop to the learned empty-slot template."""
        if self.empty_template is None:
            return 0.0
        g = self.grid
        ts = g["ts"]
        cy, cx = g["ys"][r], g["xs"][c]
        h = ts * 0.5
        crop = img[int(cy - h):int(cy + h), int(cx - h):int(cx + h)]
        return galmod.color_ncc(canon(crop), self.empty_template)

    def _cell_stds(self, img):
        g = self.grid
        rows, cols = g["rows"], g["cols"]
        ts = g["ts"]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        stds = np.zeros((rows, cols), dtype=np.float32)
        h = ts * 0.5
        for r in range(rows):
            for c in range(cols):
                cy, cx = g["ys"][r], g["xs"][c]
                win = gray[int(cy - h):int(cy + h), int(cx - h):int(cx + h)]
                stds[r, c] = win.std() if win.size else 0.0
        return stds

    def learn_std_threshold(self, img):
        """At level start (full board, all icons) learn the icon-std floor.
        Icons are opaque (not brightness-transformed), so their std is stable
        across the whole level; empty cells (showing the background) sit well
        below it. Threshold = min icon std - margin."""
        stds = self._cell_stds(img)
        self.std_thr = float(stds.min()) - 8.0

    def present_via_std(self, img):
        return self._cell_stds(img) > self.std_thr

    def present_by_count(self, img, n):
        """Present mask = the n highest-std cells (icons are colourful -> high
        std; empty cells uniform -> low std). Ranks instead of using an absolute
        threshold, so it is immune to the dim/veiled board state -- used to
        re-perceive a PARTIAL board after a reshuffle, where present_via_std's
        full-board-calibrated threshold misreads. ``n`` is the game's tilesLeft."""
        stds = self._cell_stds(img)
        total = stds.size
        if n >= total:
            return np.ones(stds.shape, dtype=bool)
        if n <= 0:
            return np.zeros(stds.shape, dtype=bool)
        # threshold = the nth-highest std; >= keeps the top-n (plus any ties)
        thr = np.partition(stds.flatten(), total - n)[total - n]
        return stds >= thr

    def classify_present(self, img, present):
        """Label every present cell (gallery NN; unseen grouped by colour-NCC)."""
        g = self.grid
        tiles = vision.extract_tiles(img, g)
        self.cur_tiles = tiles
        rows, cols = g["rows"], g["cols"]
        labels = np.full((rows, cols), vision.EMPTY, dtype=int)
        if self.gallery is not None:
            for r in range(rows):
                for c in range(cols):
                    if present[r, c]:
                        t, v = self.gallery.classify_crop(canon(tiles[r, c]))
                        if t >= 0:
                            labels[r, c] = t
        unk = [(r, c) for r in range(rows) for c in range(cols)
               if present[r, c] and labels[r, c] == vision.EMPTY]
        if unk:
            crops = [canon(tiles[r, c]) for r, c in unk]
            sub = self._cluster(crops)
            base = (labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
            for k, (r, c) in enumerate(unk):
                labels[r, c] = base + sub[k]
        return labels

    def classify_board(self, img):
        present = self.present_via_std(img)
        labels = self.classify_present(img, present)
        return present, labels

    @staticmethod
    def _cluster(crops, thr=0.80):
        n = len(crops)
        if n == 0:
            return []
        S = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                v = galmod.color_ncc(crops[i], crops[j])
                S[i, j] = S[j, i] = v
        from scipy.sparse.csgraph import connected_components
        _, lab = connected_components(S >= thr, directed=False)
        remap = {}
        out = []
        for x in lab:
            if x not in remap:
                remap[x] = len(remap)
            out.append(remap[x])
        return out

    def _cell_change(self, img_a, img_b, r, c):
        g = self.grid
        ts = g["ts"]
        cy, cx = g["ys"][r], g["xs"][c]
        h = ts * 0.62 / 2
        a = img_a[int(cy - h):int(cy + h), int(cx - h):int(cx + h)].astype(np.float32)
        b = img_b[int(cy - h):int(cy + h), int(cx - h):int(cx + h)].astype(np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.abs(a - b).mean())

    def _adapt_bg(self, img_before, img_after, present_before):
        """Update self.bg from cells that transitioned icon->background this
        move: present before, changed a lot, now low-variance. Robust to dark
        tiles (which were already low-variance and didn't change)."""
        stds_after, mean_after = vision.cell_stats(img_after, self.grid)
        acc = np.zeros(3, dtype=np.float64)
        cnt = 0
        rows, cols = present_before.shape
        for r in range(rows):
            for c in range(cols):
                if not present_before[r, c]:
                    continue
                if self._cell_change(img_before, img_after, r, c) > 22 and stds_after[r, c] < 9:
                    acc += mean_after[r, c]
                    cnt += 1
        if cnt >= 1:
            self.bg = 0.5 * self.bg + 0.5 * (acc / cnt)

    def _tiles_left(self):
        """Current tile count from the game's ``acStatus`` -- the acceptance
        signal (a real removal drops it by 2). Used to verify clicks instead of
        per-cell pixel-diff, which is unreliable on the dim (~50) veiled board:
        the selection-light toggle and low-contrast removals both produce
        diff-40-ish false positives/negatives that desync the present mask."""
        return int(self.io.status().get("tilesLeft", 99))

    def _ei_remove_ok(self):
        """True if the patched SWF exposes the acRemovePair EI handle (probed once)."""
        if self.use_remove_pair is None:
            self.use_remove_pair = self.io.has_ei("acRemovePair")
        return self.use_remove_pair

    # ---- demo / visualization (--demo) -----------------------------------
    # Paint a DOM bbox overlay on the chosen pair in the live browser window so
    # the solver's picks are visible (and screen-recordable) before each clear.
    # Coordinates reuse the same grid xs/ys/ts the click path uses, so boxes sit
    # exactly on tiles; the overlay is position:fixed at z-index max so it paints
    # above Ruffle's shadow-DOM dimming veil (bright on the dim recorded board).
    # Every call is try/except-guarded -- the overlay is cosmetic and must never
    # break a run.
    def _demo_show(self, boxes):
        """Render the given boxes [{x,y,w,h,color,label?}] via the I/O backend
        (a DOM overlay on document.body for CDP; an overlay canvas in-browser)."""
        if not self.demo:
            return
        self.io.draw_overlay(boxes)

    def _demo_hide(self):
        if not self.demo:
            return
        self.io.hide_overlay()

    def _demo_pair(self, r1, c1, r2, c2):
        """Outline the chosen pair: tile 1 cyan, tile 2 yellow (with order labels)."""
        if not self.demo:
            return
        g = self.grid
        ts = float(g["ts"])

        def box(r, c, color, label):
            return {"x": float(g["xs"][c]) - ts / 2, "y": float(g["ys"][r]) - ts / 2,
                    "w": ts, "h": ts, "color": color, "label": label}
        self._demo_show([box(r1, c1, "#00FFFF", "1"), box(r2, c2, "#FFFF00", "2")])
        if self.save_frames:
            self._save_demo_frame(r1, c1, r2, c2)

    def _save_demo_frame(self, r1, c1, r2, c2):
        """Snapshot the (bright, unveiled) bot view, draw the pair boxes, save a
        PNG. For demo frame sequences / recording material; off unless
        --save-frames. These PNGs are the BRIGHT screenshot (the dimming veil
        doesn't composite into Page.captureScreenshot), distinct from the dim
        recorded window."""
        from PIL import Image, ImageDraw
        pil = Image.fromarray(self.snap())
        d = ImageDraw.Draw(pil)
        g = self.grid
        ts = float(g["ts"])

        def rect(r, c, color):
            x = float(g["xs"][c]) - ts / 2
            y = float(g["ys"][r]) - ts / 2
            d.rectangle([x, y, x + ts, y + ts], outline=color, width=4)
        rect(r1, c1, (0, 255, 255))
        rect(r2, c2, (255, 255, 0))
        out_dir = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "demo")
        os.makedirs(out_dir, exist_ok=True)
        self._frame_idx += 1
        pil.save(os.path.join(out_dir, f"{self._frame_idx:04d}.png"))

    def _click_pair_removes(self, r1, c1, r2, c2, retries=3):
        """Remove a pair and confirm by the game's tile count dropping.

        In --demo mode, paint the bbox overlay on the chosen pair and pause
        BEFORE the (instant) acRemovePair dispatch so the boxes are visible
        while the tiles are still on screen; remove the overlay afterwards
        (try/finally so it never leaks across returns / level transitions).
        """
        if self.demo:
            self._demo_pair(r1, c1, r2, c2)
            time.sleep(self.demo_pause)
        try:
            return self._click_pair_removes_inner(r1, c1, r2, c2, retries)
        finally:
            if self.demo:
                self._demo_hide()

    def _click_pair_removes_inner(self, r1, c1, r2, c2, retries=3):
        """Remove a pair and confirm by the game's tile count dropping.

        Uses the SWF's ``acRemovePair`` ExternalInterface handle when available
        (instant, deterministic -- bypasses the lossy headless-canvas click
        path); falls back to CDP clicks otherwise. Returns True if removed."""
        if self._ei_remove_ok():
            # acRemovePair is deterministic: a fail means the pair isn't truly
            # same-type, and the caller blacklists it / eventually reshuffles.
            # io.remove_pair executes it + polls acStatus until the removal lands
            # (verify_wait); a real removal drops tilesLeft below the pre-call tl0.
            tl0 = self._tiles_left()
            if tl0 < 2:
                return True
            return self.io.remove_pair(r1, c1, r2, c2) < tl0
        for _ in range(retries):                       # fallback: lossy CDP clicks
            tl0 = self._tiles_left()
            if tl0 < 2:
                return True
            self.click_cell(r1, c1)
            time.sleep(self.click_settle)
            self.click_cell(r2, c2)
            time.sleep(self.verify_wait)
            if self._tiles_left() < tl0:
                return True
        return False

    def _sync_present(self):
        """Re-snapshot and re-derive the present mask + tiles after a board
        change (drift, an external move, or a reshuffle). Uses present_by_count
        (top-tilesLeft by std) so it stays accurate on PARTIAL boards, where
        present_via_std's full-board threshold fails. Returns None if grid lost."""
        if not self.establish_grid():
            return None
        img = self.snap()
        self.cur_tiles = vision.extract_tiles(img, self.grid)
        return self.present_by_count(img, self._tiles_left())

    def _try_reshuffle(self, present):
        """Stuck-recovery (no progress for a while): re-derive present from
        pixels; if a genuine move appears, return the fresh present (it was
        drift / a transient false-positive run). Otherwise the board is
        deadlocked -> trigger the SWF's reshuffle (acPlayOne -> createNewMap
        when no pair exists) within the per-level cap and return the post-
        reshuffle present. Return None to give up (cap reached / grid lost)."""
        present = self._sync_present()
        if present is None:
            return None
        if not present.any() or self._tiles_left() < 2:
            return present                       # cleared
        if self._pick_move(present) is not None:
            return present                       # was drift / transient
        if self._recover_count >= self.max_reshuffles:
            if self.verbose:
                print(f"[bot] deadlock; reshuffle cap ({self.max_reshuffles}) "
                      f"reached", flush=True)
            return None
        self._recover_count += 1
        tl0 = self._tiles_left()
        self.io.reshuffle()                       # acReshuffle -> createNewMap (no builtin solver)
        time.sleep(1.5)                            # let the reshuffle render settle
        present = self._sync_present()
        if present is None:
            return None
        if self.verbose:
            print(f"[bot] deadlock at {tl0} tiles -> reshuffle "
                  f"#{self._recover_count} ({int(present.sum())} tiles)", flush=True)
        return present

    def clear_board(self, max_moves=140):
        """Clear the current board.

        Re-perceives the board EVERY move (re-snapshot + re-extract tiles +
        rebuild the sim matrix + re-derive present). L2+ levels apply per-level
        gravity after every removal (``dongzuo``: L2 = createBottomMap, tiles
        fall; L3+ = left/up/right/diagonal variants), so a level-start snapshot
        would desync and cause false-positive pair picks. (L1 has no gravity, but
        re-perceiving is harmless there.) Each removal is confirmed by tilesLeft
        dropping; on a deadlock the SWF reshuffles (acReshuffle -> createNewMap).
        """
        if not self.establish_grid():
            return False, 0
        img = self.snap()
        self.learn_std_threshold(img)
        # Initial perception -- used by the warmup for the first, pre-gravity
        # move; the main loop re-perceives every move after that.
        self.cur_tiles = vision.extract_tiles(img, self.grid)
        self._recover_count = 0
        present = np.ones((self.grid["rows"], self.grid["cols"]), dtype=bool)

        if self.demo and self.demo_flash:
            # Flash a faint box over every detected grid cell so a viewer sees
            # the full lattice the solver perceives, before move-by-move picks.
            g = self.grid
            ts = float(g["ts"])
            cells = [{"x": float(g["xs"][c]) - ts / 2, "y": float(g["ys"][r]) - ts / 2,
                      "w": ts, "h": ts, "color": "rgba(0,255,255,0.35)"}
                     for r in range(g["rows"]) for c in range(g["cols"])]
            self._demo_show(cells)
            time.sleep(1.5)
            self._demo_hide()

        # Warm the click path: click top picks (with retry) until tilesLeft
        # first drops, tracking the removal so the present mask stays exact.
        for _ in range(15):
            if not present.any() or self._tiles_left() < 2:
                break
            mv = self._pick_move(present)
            if mv is None:
                break
            (r1, c1), (r2, c2) = mv
            if self._click_pair_removes(r1, c1, r2, c2):
                present[r1, c1] = False
                present[r2, c2] = False
                break

        moves = 0
        fails = 0
        cleared = False
        while moves < max_moves:
            tl = self._tiles_left()
            if tl < 2:
                cleared = True
                break
            # Re-perceive EVERY move. L2+ levels apply per-level gravity after
            # each removal (dongzuo(): L2=createBottomMap tiles fall, L3+ left/
            # up/right/diagonal variants), so a level-start snapshot desyncs and
            # causes false-positive pair picks. Re-snapshot + re-extract + rebuild
            # the sim matrix + re-derive present from the live board.
            img = self.snap()
            self.cur_tiles = vision.extract_tiles(img, self.grid)
            self._sim = None                       # force _ensure_sim rebuild
            present = self.present_by_count(img, tl)
            if not present.any():
                cleared = True
                break
            mv = self._pick_move(present)
            if mv is not None and self._click_pair_removes(
                    mv[0][0], mv[0][1], mv[1][0], mv[1][1]):
                moves += 1
                fails = 0
                if moves % 12 == 0 and self.verbose:
                    print(f"[bot] {moves} moves, {self._tiles_left()} left", flush=True)
                time.sleep(self.gravity_settle)     # let the gravity anim settle
                continue
            # No progress: a deterministic acRemovePair fail means the two tiles
            # at those cells aren't truly same-type -- blacklist and re-pick.
            if mv is not None:
                self.known_diff.add(frozenset((mv[0], mv[1])))
            fails += 1
            if fails and fails % 20 == 0:
                self.known_diff.clear()
                if self.verbose:
                    print(f"[bot] ...stalled: {moves} moves, fails={fails}, "
                          f"tilesLeft={self._tiles_left()}", flush=True)
            if fails < 15:
                continue
            # Stuck for ~15 attempts -> re-derive present; reshuffle
            # (acReshuffle -> createNewMap) if genuinely deadlocked, then resume.
            present = self._try_reshuffle(present)
            if present is None:
                break
            self.known_diff.clear()
            fails = 0
            if self._tiles_left() < 2:
                cleared = True
                break
        if not cleared:
            cleared = self._tiles_left() < 2
        if self.verbose:
            print(f"[bot] level done: cleared={cleared} moves={moves} "
                  f"tilesLeft={self._tiles_left()}")
        return cleared, moves

    def _solvable(self, present, depth=0, memo=None):
        """Recursive solvability check: can `present` be fully cleared using
        same-type connectable pairs (NN or NCC)? My own deadlock-prevention
        algorithm. Memoised on the present-mask for speed. Used only for small
        boards."""
        self._ensure_sim()
        n = int(present.sum())
        if n == 0:
            return True
        if n % 2 == 1:
            return False
        if memo is None:
            memo = {}
        key = present.tobytes()
        if key in memo:
            return memo[key]
        if depth > 40:
            memo[key] = True   # bail: assume solvable (avoid blowup)
            return True
        # enumerate high-score connectable pairs (same-type candidates)
        cands = []
        for (a, b) in conn.all_connectable_pairs_anylabel(present):
            try:
                v = self._pair_sim(a, b)
            except Exception:  # noqa: BLE001
                continue
            if v >= self.cand_thr:
                cands.append((v, a, b))
        cands.sort(reverse=True)
        for v, a, b in cands[:12]:
            p2 = present.copy()
            p2[a[0], a[1]] = False
            p2[b[0], b[1]] = False
            if self._solvable(p2, depth + 1, memo):
                memo[key] = True
                return True
        memo[key] = False
        return False

    def _ncc_safe(self, present, pair):
        """Is removing `pair` safe (leaves a solvable board)? Uses the recursive
        check for small boards, 1-ply for larger."""
        self._ensure_sim()
        (r1, c1), (r2, c2) = pair
        p2 = present.copy()
        p2[r1, c1] = False
        p2[r2, c2] = False
        if int(p2.sum()) == 0:
            return True
        if int(p2.sum()) <= 10:
            return self._solvable(p2)
        # 1-ply fallback for larger boards (cheap)
        k = 0
        for (a, b) in conn.all_connectable_pairs_anylabel(p2):
            k += 1
            if k > 40:
                return True
            try:
                if self._pair_sim(a, b) > self.cand_thr:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _cf_safety(self, present, pair):
        """Lookahead safety score in [0,1] for removing `pair` (1 = leaves a
        clearable board, 0 = doomed): exact solvable in the endgame, else the
        Monte-Carlo clear-rate over rollout_k random completions. Returns -1 if
        the C++ extension or NN sim matrix is unavailable (caller falls back to
        _ncc_safe)."""
        if _cf is None or self._sim is None:
            return -1.0
        (r1, c1), (r2, c2) = pair
        p2 = present.copy()
        p2[r1, c1] = False
        p2[r2, c2] = False
        if int(p2.sum()) == 0:
            return 1.0
        if int(p2.sum()) <= self.exact_tiles:
            return 1.0 if _cf.solvable(p2, self._sim, self.cand_thr, 0, 12) else 0.0
        self._cf_counter += 1
        return float(_cf.rollout_clear_rate(p2, self._sim, self.cand_thr,
                                             self.rollout_k, self._cf_counter))

    def _pick_move(self, present):
        """Pick the best connectable same-icon pair (the tile classifier),
        skipping game-rejected pairs. When the C++ lookahead is available, the
        top candidates are ranked by Monte-Carlo clear-rate (deep deadlock
        avoidance) / exact solvable in the endgame; otherwise the 1-ply
        _ncc_safe heuristic. Classifier = trained PairNet when available
        (issue #3), else colour-NCC."""
        self._ensure_sim()
        use_cf = self.use_rollout and _cf is not None and self._sim is not None
        cands = []
        any_mv = None
        enum = (((int(a), int(b)), (int(c), int(d)))
                for a, b, c, d in _cf.connectable_pairs(present)) if use_cf \
            else conn.all_connectable_pairs_anylabel(present)
        for (a, b) in enum:
            if frozenset((a, b)) in self.known_diff:
                continue
            any_mv = (a, b) if any_mv is None else any_mv
            try:
                v = self._pair_sim(a, b)
            except Exception:  # noqa: BLE001
                continue
            if v >= self.cand_thr:
                cands.append((v, (a, b)))
        cands.sort(reverse=True)
        if cands:
            top = cands[:(self.rollout_m if use_cf else 6)]
            if use_cf:
                # rollout as a deadlock FILTER, not a ranking: clear-rate is
                # reliable for "this move dooms the board" (~0) but noisy for
                # ranking good moves. So drop deadlock-prone candidates, then pick
                # the highest-sim survivor (greedy on true positives -- strong on
                # clearable boards, where this reduces to pure greedy). If every
                # candidate looks doomed, take the least-bad clear-rate.
                scored = [(self._cf_safety(present, pair), v, pair) for v, pair in top]
                safe = [t for t in scored if t[0] >= self.safe_thr]
                if safe:
                    return max(safe, key=lambda t: t[1])[2]      # highest sim
                return max(scored, key=lambda t: t[0])[2]        # least-bad rate
            # Python fallback: first _ncc_safe candidate, else top-scoring
            for _v, pair in top:
                if self._ncc_safe(present, pair):
                    return pair
            return cands[0][1]
        if any_mv is not None:
            return any_mv
        n_present = int(present.sum())
        if n_present <= 8:
            cells = [(r, c) for r in range(present.shape[0])
                     for c in range(present.shape[1]) if present[r, c]]
            best2, best2_ncc = None, 0.0
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    a, b = cells[i], cells[j]
                    if frozenset((a, b)) in self.known_diff:
                        continue
                    try:
                        v = self._pair_sim(a, b)
                    except Exception:  # noqa: BLE001
                        continue
                    if v > best2_ncc:
                        best2_ncc, best2 = v, (a, b)
            return best2
        return None

    def _shuffle(self):
        """Deadlock fallback (rare with lookahead). Not wired in dev build."""
        return

    # ---- transitions (control handles) ------------------------------------
    def advance(self):
        """Dismiss the level-complete / game-complete overlay and proceed.

        Called only right after clear_board() emptied the board. Uses acAdvance,
        which dismisses the result overlay (acHandleResult -> next level /
        restart) ONLY -- never acStep/acPlayOne, which can reach the builtin
        pair-finder (vum.cunufi). Stops the moment a fresh board appears.
        (Equivalent to clicking 'continue'; no board state is read.)"""
        if self.transition_mode == "ei":
            for _ in range(20):
                self.io.advance()       # acAdvance: dismiss result overlay only
                time.sleep(0.45)
                img = self.snap()
                g = vision.detect_grid(img)
                # a fresh playable board has re-appeared (next level / new run)
                if g and (g["cols"], g["rows"]) == (12, 8):
                    pres = vision.present_mask(img, g, bg_color=self.bg)
                    if int(pres.sum()) > 30:
                        return "next"
                # game-complete overlay has no board -> keep stepping (restarts)
            return "done"
        else:
            return self._click_continue()

    def _click_continue(self):
        """Vanilla: click the result-screen continue button (CV). TODO."""
        pass

    # ---- full game --------------------------------------------------------
    def _restart_session(self):
        """Restart the headless game session (chrome+Xvfb via session.sh) between
        successful runs to reset the Ruffle renderer, which degrades after
        ~15-20 min of continuous play and breaks grid detection (issue #2). The
        HTTP server persists across this (session.sh only manages chrome+Xvfb).
        Each run then starts from a fresh renderer, well under the window."""
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        sh = os.path.join(here, "session.sh")
        env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":99")}
        for cmd in ([sh, "stop"], [sh, "start"]):
            try:
                subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=90)
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[run] session {cmd[-1]} error: {e!r}")
            time.sleep(2.0)
        # Robustly wait for chrome to come back + the EI bridge to register.
        # cdp._send can raise while chrome is mid-startup; tolerate and retry.
        deadline = time.time() + 90.0
        ready = False
        while time.time() < deadline:
            try:
                if Bot._wait_player(timeout=5.0):
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        if not ready:
            raise RuntimeError("EI bridge did not come back after session restart")
        time.sleep(4.0)  # let the SWF settle into a playable state
        if self.verbose:
            print("[run] session restarted; renderer reset", flush=True)

    def play_game(self, runs=3, max_level=13, per_level_timeout=260.0, restart_between=True):
        clears = 0
        for run in range(runs):
            if run > 0 and restart_between:
                # only reached if the previous run succeeded (a failure breaks)
                if self.verbose:
                    print("\n[run] restarting session to reset the renderer before the next run", flush=True)
                self._restart_session()
            if self.verbose:
                print(f"\n=== RUN {run+1}/{runs} ===")
            self.reset_grid()
            self._begin_run()
            level = 1
            run_ok = True
            while level <= max_level:
                t0 = time.time()
                if self.verbose:
                    print(f"[run] level {level}: starting (tilesLeft={self._tiles_left()})...",
                          flush=True)
                cleared, moves = self.clear_board()
                dt = time.time() - t0
                if self.verbose:
                    print(f"[run] level {level}: cleared={cleared} moves={moves} ({dt:.0f}s)",
                          flush=True)
                if not cleared:
                    run_ok = False
                    break
                level += 1
                if level <= max_level:
                    self.advance()
                    self.reset_grid()
                    time.sleep(1.5)
            if run_ok and level > max_level:
                clears += 1
                if self.verbose:
                    print(f"[run] RUN {run+1} CLEARED all {max_level} levels ✓ (clears={clears})")
            else:
                if self.verbose:
                    print(f"[run] RUN {run+1} failed at level {level} -- stopping (fail-fast)")
                break  # FAIL FAST: don't restart/continue on a failed run
        if self.verbose:
            print(f"\n=== {clears}/{runs} runs fully cleared ===")
        return clears

    def _begin_run(self):
        if self.transition_mode == "ei":
            self.io.set_enabled(False)
            self.io.reset()
            time.sleep(3.0)
            self.io.set_enabled(False)

    @staticmethod
    def _wait_player(timeout=40.0):
        """Poll until the SWF's EI bridge (acStatus) is callable. Convenience for
        external launchers (solver/demo.sh); the bot itself uses self.io.wait_ready."""
        return gameio.CDPGameIO().wait_ready(timeout)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=os.path.join(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "gallery_lvl1.npz"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-level", type=int, default=13)
    ap.add_argument("--transition", default="ei", choices=["ei", "click"])
    ap.add_argument("--model", default=None,
                    help="path to a trained PairNet .pt (default: siamese micro)")
    ap.add_argument("-q", action="store_true")
    ap.add_argument("--rollout", action="store_true",
                    help="enable the C++ rollout lookahead (conn_fast)")
    ap.add_argument("--ncc", action="store_true",
                    help="use colour-NCC as the tile backbone (skip the NN model)")
    ap.add_argument("--demo", action="store_true",
                    help="paint a bbox overlay on each chosen pair in the browser "
                         "window before clearing (for live viewing / recording)")
    ap.add_argument("--pause", type=float, default=0.7,
                    help="seconds each pair's boxes stay on screen (--demo)")
    ap.add_argument("--save-frames", action="store_true",
                    help="also save annotated PNGs of each pick to $tmp/demo/")
    ap.add_argument("--no-flash", action="store_true",
                    help="skip the level-start full-grid flash (--demo)")
    args = ap.parse_args()
    bot = Bot(args.gallery, verbose=not args.q, transition_mode=args.transition,
              model_path=args.model)
    if args.rollout:
        bot.use_rollout = True
    if args.ncc:
        # NCC backbone: skip the NN, use a higher same-type threshold for precision
        # (NCC>=0.8 is guaranteed same-type; the NN's 0.5 is too low for NCC).
        bot.use_nn = False
        bot.nn = None
        bot.cand_thr = 0.7
        print("[bot] backbone = colour-NCC (cand_thr=0.7)")
    if args.demo:
        bot.demo = True
        bot.demo_pause = args.pause
        bot.demo_flash = not args.no_flash
        bot.save_frames = args.save_frames
        print(f"[bot] demo mode on (pause={args.pause}s, flash={bot.demo_flash}, "
              f"save_frames={bot.save_frames})")
    bot.play_game(args.runs, args.max_level)


if __name__ == "__main__":
    main()
