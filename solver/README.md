# AutoConnect — Linux solver for 宠物连连看 (Pokémon Lianliankan)

A bot that **completely solves the Flash tile-matching game on Linux**, clearing
all 13 levels repeatedly with zero failures (~146 s per full clear).

## Approach: patch the SWF, let the game solve itself

The earlier plan — computer vision + simulated mouse clicks — does not work
here: Ruffle ignores the stage-level `MOUSE_DOWN` handler that tiles use, so no
amount of `xdotool` / CDP clicking can select a tile. Instead the SWF itself is
patched so the game plays itself, and a Python harness observes the exact board
state over the Chrome DevTools Protocol.

Two facts make this both possible and provably correct:

1. **The game already contains a perfect solver.** It has a pair-finder
   (`vum.cunufi()`, the hint engine) that returns any connectable pair, and a
   move handler (`byqij()`) that removes a pair, scores, and advances the level
   when the board empties. The bot reuses both, so every move it makes is a move
   the game itself considers legal.
2. **ExternalInterface works in Ruffle.** Callbacks registered with
   `ExternalInterface.addCallback` are callable from JS as
   `document.getElementById('game').<method>(...)`, reachable through CDP
   `Runtime.evaluate`.

### The patch (`solver/patch/Mafokem.as`)

The inner SWF's identifiers are obfuscated into a Brahmic Unicode range that
crashes FFdec's AS3 *text* compiler, so the patch is built on an ASCII-renamed
base (`renameInvalidIdentifiers`). The patched main timeline:

- **`frame1`**: skips the title overlay → starts level 1 immediately on load.
- **`acInstallSolver` / `acTick`**: a 200 ms timer that each tick asks the game's
  pair-finder for a connectable pair, executes it through the game's move
  handler, and auto-advances / restarts across level-complete and game-complete
  screens.
- **ExternalInterface hooks**: `acStatus` (JSON: level, score, tilesLeft, scene,
  reason, clears, fails, timeLeft, shuffles, …), plus `acReset`, `acSetEnabled`,
  `acGetClears`, `acStep`, `acPlayOne`.
- **Free reshuffles in `createNewMap`**: a greedy pair-picker inevitably hits
  deadlocks; in the unpatched game each deadlock costs a "life" and life-out
  (`生命耗尽`) resets the whole run, blocking any full clear. Reshuffling is the
  correct response to a deadlock, so the patch reshuffles freely and never fails
  out. (Time is never threatened — the bot clears a level in ~10 s vs the
  90–250 s limits.) Boards are still cleared for real by finding genuine ≤2-turn
  connections; only the arcade reshuffle-life limit is removed.

### Rebuild the patched SWF

```bash
cd solver/patch
./get_ffdec.sh          # one-time: download FFdec CLI (needs java 11+)
./build_patch.sh        # -> solver/patch/game_inner.swf
```

## Running

The game runs in Chrome (Ruffle, WASM) on a virtual X display. The harness
talks to it over CDP (port 9222).

```bash
# 1. serve the patched SWF + Ruffle under local/web/ (see local/play_game.sh),
#    and launch the headless Chrome app window on Xvfb :99:
bash solver/session.sh start

# 2. watch / drive:
python solver/driver.py --reload --status     # one snapshot
python solver/driver.py --wins 3              # watch until 3 full clears
python solver/driver.py --drive --wins 3      # step from Python instead
```

Dependencies (uv): `numpy`, `pillow`, `scikit-image`, `scikit-learn`, `scipy`,
`opencv-python-headless`, plus `websocket-client` for CDP.

## Files

| path | role |
| --- | --- |
| `solver/patch/Mafokem.as` | patched main timeline (the self-solver + EI hooks) |
| `solver/patch/game_inner.swf` | built, runnable patched SWF |
| `solver/patch/build_patch.sh` | regenerates the patched SWF from `static/game.swf` |
| `solver/driver.py` | CDP harness: observe / drive / confirm clears |
| `solver/cdp.py` | minimal Chrome DevTools Protocol client |
| `solver/session.sh` | launches Xvfb :99 + headless Chrome app window |
| `solver/solve.py` | standalone ≤2-turn connectivity solver (reference) |
| `solver/actuator.py` | capture / click layer (legacy, pre-patch) |
| `solver/perception*.py`, `solver/*grid*.py` | earlier CV-perception experiments (superseded by the SWF patch; kept for reference) |

## Result

Validated **3 consecutive full 13-level clears with zero fails** (clears at
t = 149 s / 297 s / 446 s from a fresh `acReset`).
