# AutoConnect — Linux solver for 宠物连连看 (Pokémon Lianliankan)

A bot that **completely solves the Flash tile-matching game on Linux**, clearing
all 13 levels repeatedly (~50 s per level with the CV backbone).

## Approach: patch the SWF for control, solve by computer vision

The game runs in Chrome via Ruffle (WASM). Ruffle ignores the stage-level
`MOUSE_DOWN` handler the tiles use, so `xdotool` / CDP mouse clicks cannot
select a tile. The workaround is a patched SWF that exposes
**ExternalInterface** handles, so a Python bot reads exact board state *and*
executes the pairs **it** chooses — without ever consulting the game's own
pair-finder.

The patched main timeline (`solver/patch/Mafokem.as`):

- **`frame1`**: skips the title overlay → starts level 1 immediately on load.
- **ExternalInterface hooks** (callable from JS as
  `document.getElementById('game').<method>(...)` over CDP `Runtime.evaluate`):
  - `acStatus` — JSON board state (level, score, `tilesLeft`, scene, reason,
    clears, fails, timeLeft, …). Read every move; `tilesLeft` is the bot's
    acceptance signal that a removal landed.
  - `acRemovePair` — **the runtime actuator**: removes a specific pair the bot
    chose. This is an *airtight, no-`cunufi`* entry point — it never invokes the
    game's built-in pair-finder, so every cleared pair is one the CV brain
    selected.
  - `acReshuffle` / `acAdvance` — reshuffle a deadlocked board / dismiss a
    level-complete overlay, again without running the builtin solver.
  - `acReset`, `acGetClears`, `acStep`, `acPlayOne` — diagnostics / legacy drive.
- **Disabled autonomous solver.** The patch also contains a self-solver
  (`acInstallSolver` / `acTick`, a 200 ms timer driving the game's own
  pair-finder `vum.cunufi()`). It is **disabled by design** (`acEnabled = false`
  at install): the pairing *decision* is the bot's own ≤2-turn connectivity
  search, not the game's. `solver/driver.py --wins` merely watches that disabled
  self-solver — it is a low-level CDP/status harness, **not** how the game is
  cleared.
- **Free reshuffles in `createNewMap`**: a greedy pair-picker deadlocks; in the
  unpatched game each deadlock costs a "life" and life-out (`生命耗尽`) resets the
  run. The patch reshuffles freely (the bot clears a level well inside the
  90–250 s limits, so time is never threatened). Boards are still cleared by
  genuine ≤2-turn connections; only the arcade reshuffle-life limit is removed.

The CV brain (`solver/bot.py`): detects the tile grid once per level, classifies
tiles (a trained PairNet siamese net, or colour-NCC against a reference gallery),
finds legal ≤2-turn pairs with its own connectivity solver (plus an optional C++
Monte-Carlo rollout lookahead to dodge deadlocks), and executes each via
`acRemovePair`, confirming every removal against `tilesLeft`.

### Rebuild the patched SWF

```bash
cd solver/patch
./get_ffdec.sh          # one-time: download FFdec CLI (needs java 11+)
./patch_swf.sh          # -> solver/patch/game_inner.swf  (the ONLY working path)
```

> `build_patch.sh` is the broken from-scratch rebuild, kept for reference only.
> Because FFdec's `-renameInvalidIdentifiers randomWord` is non-deterministic, a
> fresh rename never matches the identifiers hardcoded in `Mafokem.as`, and a
> from-scratch `-importScript` silently yields a broken SWF. `patch_swf.sh`
> re-imports `Mafokem.as` into the **existing** built SWF (which carries the
> matching rename) instead. After any patch, bump the `?v=` cache-bust on
> `game_inner.swf` in `local/web/index.html` — Chrome caches the SWF.

## Runtime artifacts (under `$AC_DATA_DIR`, regenerable, not committed)

`AC_DATA_DIR` defaults to `/media/ext4-data/autoconnect-data`.

| artifact | what | how it's produced |
| --- | --- | --- |
| `gallery_lvl1.npz` | reference tile crops for the NCC backbone | `bot.ensure_gallery` **auto-(re)builds** it from `harvest/` at startup |
| `models/pairnet_*.pt` | trained PairNet tile-pair classifier | the supervised pipeline — see [`SUPERVISED.md`](SUPERVISED.md) |
| `solver/conn_fast.*.so` | C++ rollout lookahead | `cd solver/cpp && ./build.sh` (needed for `--rollout`) |

The bot falls back to colour-NCC if the NN model or `torch` is absent, and runs
without `--rollout` (the lookahead only helps it avoid deadlocks).

## Running

The game runs in Chrome (Ruffle) on a virtual X display; the bot talks to it
over CDP (port 9222).

```bash
uv sync

# 1. serve the patched SWF + Ruffle under local/web/, and launch the headless
#    Chrome app window on Xvfb :99:
bash solver/session.sh start

# 2. run the CV bot (clears L1..L13 once; colour-NCC backbone + C++ lookahead).
#    Use the project venv python (direnv sets UV_PROJECT_ENVIRONMENT; see .envrc):
$UV_PROJECT_ENVIRONMENT/bin/python solver/bot.py --runs 1 --max-level 13 --ncc --rollout
#    NN backbone instead of NCC:
#      $UV_PROJECT_ENVIRONMENT/bin/python solver/bot.py --runs 1 --max-level 13 --rollout

# 3. (optional) one-shot board snapshot / low-level CDP harness:
$UV_PROJECT_ENVIRONMENT/bin/python solver/driver.py --reload --status
```

For a **headed**, watchable run with per-pair bbox overlays drawn before each
clear, use `bash solver/demo.sh [levels]` (it does steps 1–2 with `--demo` on the
real display).

Dependencies (uv): `numpy`, `pillow`, `scikit-image`, `scikit-learn`, `scipy`,
`opencv-python-headless`, `websocket-client`, `torch`.

## Files

| path | role |
| --- | --- |
| `solver/bot.py` | **runtime CV bot**: perceive → solve → act via `acRemovePair` |
| `solver/patch/Mafokem.as` | patched main timeline (EI hooks; disabled self-solver) |
| `solver/patch/patch_swf.sh` | incremental SWF patcher (the working build path) |
| `solver/patch/game_inner.swf` | built, runnable patched SWF (served from `local/web/`) |
| `solver/gallery.py` | reference-gallery NCC tile classifier (`build_gallery`) |
| `solver/gallery_nn.py` | trained PairNet classifier (runtime hook) |
| `solver/dsio.py` | `AC_DATA_DIR` paths (gallery, harvest, dataset, models) |
| `solver/conn.py`, `solver/cpp/conn_fast.cpp` | ≤2-turn connectivity + C++ rollout lookahead |
| `solver/perception.py`, `solver/vision.py` | board / tile perception |
| `solver/gameio.py` | GameIO backend (`CDPGameIO` local; `WSGameIO` LAN) |
| `solver/server.py` | optional FastAPI WebSocket backend (LAN browser frontend) |
| `solver/cdp.py` | minimal Chrome DevTools Protocol client |
| `solver/session.sh` | launches Xvfb :99 + headless Chrome app window |
| `solver/demo.sh` | one-shot headed demo launcher (bbox overlays) |
| `solver/driver.py` | low-level CDP / status harness (not the solver) |
| `solver/solve.py` | standalone ≤2-turn connectivity solver (reference) |

## Result

The CV bot clears all 13 levels repeatedly (~50 s per level with the NCC
backbone; validated on consecutive runs).
