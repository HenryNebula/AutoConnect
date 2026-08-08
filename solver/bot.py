"""Legitimate runtime bot: perceive the board by CV, solve with my own
connectivity algorithm, and act via CDP mouse clicks. No game state is read
for gameplay decisions.

Per move the board is RE-PERCEIVED from pixels (so tile-drift levels, where the
game shifts tiles after each move, are handled): find a connectable same-icon
pair with my <=2-turn connector + one-ply deadlock lookahead, CDP-click it, then
re-perceive and confirm the tile count dropped by two (the game's own acceptance
signal -- equivalent to a human seeing the pair clear). A non-decreasing count
means a mis-classification: that pair is remembered as "known different" and the
bot re-picks. The board state is never read from the game.

Level/game transitions are driven as control handles (see ``advance``): while
developing, the working SWF's ExternalInterface continue/restart callbacks are
used; for the final vanilla run these are replaced by CDP clicks on the result
buttons (same control actions, no state read).
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import cdp
import vision
import conn
import gallery as galmod

CANON = galmod.CANON
DEFAULT_BG = np.array([53.0, 44.0, 26.0], dtype=np.float32)


def _ev(expr):
    r = cdp._send("Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return r.get("result", {}).get("value")


def _player_call(method, *args):
    a = ",".join(json.dumps(x) for x in args)
    expr = ("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
            "if(!e||typeof e.%s!=='function')return undefined;"
            "try{return e.%s(%s);}catch(x){return 'ERR:'+x;}})()" % (method, method, a))
    return _ev(expr)


def canon(crop):
    return cv2.resize(crop, (CANON, CANON), interpolation=cv2.INTER_AREA)


class Bot:
    def __init__(self, gallery_path=None, verbose=True, transition_mode="ei",
                 model_path=None):
        self.verbose = verbose
        self.grid = None
        self.bg = DEFAULT_BG.copy()
        self.gallery = galmod.GalleryClassifier(gallery_path, max_templates=8) if gallery_path else None
        self.known_diff = set()
        self.click_settle = 0.20
        self.verify_wait = 0.40
        self.transition_mode = transition_mode   # "ei" (dev) or "click" (vanilla)
        self.empty_template = None     # learned empty-slot crop (CANON-sized)
        self.empty_thr = 0.55          # NCC >= this => cell is an empty slot
        self.std_thr = 60.0            # icon-std floor (learned at level start)
        self._recover_count = 0        # deadlock-reshuffle attempts this level
        # Learned same-type tile classifier (issue #3): the trained PairNet is a
        # drop-in scoring function for gallery.color_ncc. If no model is present
        # (or torch is missing) the bot falls back to colour-NCC, unchanged.
        self.nn = self._maybe_load_nn(model_path)
        self.use_nn = self.nn is not None and getattr(self.nn, "available", False)
        self.cand_thr = 0.5 if self.use_nn else 0.55   # NN prob vs NCC score
        self._sim = None            # cached all-pairs tile-sim matrix (NN only)
        self._sim_tiles = None      # id(self.cur_tiles) the cache belongs to

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
        once per move (one batched NN forward). Subsequent per-pair lookups in
        _pick_move / lookahead are then cheap array indexing. The NCC fallback
        skips this and keeps scoring per-pair, as before (an O(n^2) NCC matrix
        would be slower than scoring only the connectable pairs)."""
        tiles = self.cur_tiles
        if (not self.use_nn) or tiles is None or tiles is self._sim_tiles:
            return
        self._sim_tiles = tiles
        R, C = tiles.shape[:2]
        crops = np.stack([canon(tiles[r, c]) for r in range(R) for c in range(C)])
        self._sim = self.nn.sim_matrix(crops).reshape(R, C, R, C)

    def _pair_sim(self, a, b):
        """Same-type score in [0,1] for cells a=(r,c), b=(r,c). NN probability
        when a model is loaded, else colour-NCC (issue #3's old classifier)."""
        if self._sim is not None:
            return float(self._sim[a[0], a[1], b[0], b[1]])
        return galmod.color_ncc(canon(self.cur_tiles[a[0], a[1]]),
                                canon(self.cur_tiles[b[0], b[1]]))

    # ---- low level --------------------------------------------------------
    def snap(self):
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "_bot.png")
        cdp.capture(p)
        return vision.load_img(p)

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

    def _count_drift_changes(self, img_a, img_b, present, exclude, thr=60):
        """Count present cells (excluding ``exclude``) whose pixels changed by
        more than ``thr`` between two frames -- the signature of tile drift.
        Vectorised over the whole board."""
        g = self.grid
        ts = g["ts"]
        h = int(ts * 0.45)
        dif = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).sum(2)
        excl = set(exclude)
        cnt = 0
        for r in range(g["rows"]):
            for c in range(g["cols"]):
                if not present[r, c] or (r, c) in excl:
                    continue
                cy, cx = g["ys"][r], g["xs"][c]
                win = dif[int(cy - h):int(cy + h), int(cx - h):int(cx + h)]
                if win.size and win.mean() > thr:
                    cnt += 1
        return cnt

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
        s = _player_call("acStatus")
        if isinstance(s, str):
            try:
                return int(json.loads(s).get("tilesLeft", 99))
            except Exception:  # noqa: BLE001
                return 99
        return 99

    def _click_pair_removes(self, r1, c1, r2, c2, retries=3):
        """Click a pair and confirm removal by the game's tile count dropping.
        Retries the full two-click sequence -- the first few post-load CDP
        clicks often don't register (cold path), so a no-removal is not proof
        the pair is wrong. Returns True if the game removed the pair."""
        for _ in range(retries):
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

    def clear_board(self, max_moves=140):
        """Clear the current board, drift-safe and fast.

        A move is chosen by the tile classifier (trained PairNet, else colour-
        NCC) over connectable pairs; each removal is confirmed by the GAME's
        tile count dropping (``acStatus`` tilesLeft), with click retries for the
        cold post-load click path. Present is tracked from confirmed removals
        (stable); ``present_via_std`` is only re-derived on a perceived deadlock,
        since mid-clear animations make per-frame std unreliable.
        """
        if not self.establish_grid():
            return False, 0
        img = self.snap()
        self.learn_std_threshold(img)
        self._recover_count = 0
        present = np.ones((self.grid["rows"], self.grid["cols"]), dtype=bool)

        # Warm the click path: click top picks (with retry) until tilesLeft
        # first drops, tracking the removal so the present mask stays exact.
        for _ in range(15):
            if not present.any() or self._tiles_left() < 2:
                break
            self.cur_tiles = vision.extract_tiles(self.snap(), self.grid)
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
            if not present.any() or self._tiles_left() < 2:
                cleared = True
                break
            self.cur_tiles = vision.extract_tiles(self.snap(), self.grid)
            mv = self._pick_move(present)
            if mv is None:
                fails += 1
                if fails > 70:
                    # No connectable pair among tracked-present cells. On a
                    # static level that is a real deadlock; on a drift level
                    # the tracked mask may be stale, so re-derive once from
                    # pixels before giving up.
                    img = self.snap()
                    present = self.present_via_std(img)
                    if not present.any() or self._tiles_left() < 2:
                        cleared = True
                        break
                    if self.verbose:
                        print(f"[bot] stuck (deadlock); {int(present.sum())} left")
                    break
                continue
            (r1, c1), (r2, c2) = mv
            if self._click_pair_removes(r1, c1, r2, c2):
                moves += 1
                fails = 0
                present[r1, c1] = False
                present[r2, c2] = False
                if moves % 12 == 0 and self.verbose:
                    print(f"[bot] {moves} moves, {int(present.sum())} left", flush=True)
            else:
                self.known_diff.add(frozenset(((r1, c1), (r2, c2))))
                fails += 1
                if fails % 20 == 0:
                    # Stall recovery: the CDP click path is intermittently flaky,
                    # so good pairs can get wrongly blacklisted in known_diff and
                    # a tile selection can stick. Reset both, then re-warm clicks.
                    self.known_diff.clear()
                    empties = [(r, c) for r in range(present.shape[0])
                               for c in range(present.shape[1]) if not present[r, c]]
                    if empties:
                        er, ec = empties[0]
                        self.click_cell(er, ec)
                        time.sleep(0.2)
                    if self.verbose:
                        print(f"[bot] stall recovery at {moves} moves (fails={fails}): "
                              f"cleared known_diff, reset selection", flush=True)
                elif self.verbose and fails % 10 == 0:
                    print(f"[bot] ...stalled: {moves} moves done, {fails} fails, "
                          f"tilesLeft={self._tiles_left()} present={int(present.sum())}",
                          flush=True)
                if fails > 80:
                    if self.verbose:
                        print(f"[bot] too many fails; aborting at {int(present.sum())}",
                              flush=True)
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

    def _pick_move(self, present):
        """Pick the highest-scoring connectable same-icon pair (the tile
        classifier), skipping game-rejected pairs. The classifier is the trained
        PairNet when available (issue #3), else colour-NCC."""
        self._ensure_sim()
        cands = []
        any_mv = None
        for (a, b) in conn.all_connectable_pairs_anylabel(present):
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
            # Prefer a move that doesn't create a deadlock: among the top
            # candidates, return the first whose removal leaves a board that
            # still has a connectable same-type pair (1-ply, cheap on large
            # boards via _ncc_safe; exact _solvable on small ones). Pure-greedy
            # move order otherwise deadlocks ~1-in-10 layouts. Fall back to the
            # top-scoring pair if none of the top candidates is provably safe.
            for _v, pair in cands[:6]:
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

        Called only right after clear_board() emptied the board, so the only
        non-no-op path inside the SWF's action dispatch is the result-overlay
        handler (acHandleResult -> next level / restart). On an empty board the
        solver branch (acPlayOne) is a no-op, so this never solves for us. We
        stop the moment a fresh board appears. (Equivalent to clicking
        'continue'; no board state is read.)"""
        if self.transition_mode == "ei":
            for _ in range(20):
                _player_call("acStep")
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
    def play_game(self, runs=3, max_level=13, per_level_timeout=260.0):
        clears = 0
        for run in range(runs):
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
                    print(f"[run] RUN {run+1} failed at level {level}")
        if self.verbose:
            print(f"\n=== {clears}/{runs} runs fully cleared ===")
        return clears

    def _begin_run(self):
        if self.transition_mode == "ei":
            _player_call("acSetEnabled", False)
            _player_call("acReset")
            time.sleep(3.0)
            _player_call("acSetEnabled", False)

    @staticmethod
    def _wait_player(timeout=40.0):
        t = time.time()
        while time.time() - t < timeout:
            v = _ev("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
                    "return !!(e && typeof e.acStatus==='function');})()")
            if v is True:
                return True
            time.sleep(0.4)
        return False


def _has_ei(method):
    return _ev("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
               "return !!(e && typeof e.%s==='function');})()" % method) is True


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
    args = ap.parse_args()
    bot = Bot(args.gallery, verbose=not args.q, transition_mode=args.transition,
              model_path=args.model)
    bot.play_game(args.runs, args.max_level)


if __name__ == "__main__":
    main()
