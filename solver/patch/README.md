# Patching the game SWF (`game_inner.swf`)

The bot does **not** play the original Flash game. The game's inner SWF is
patched so it (a) plays itself with its own built-in solver and (b) exposes its
state over ExternalInterface so the Python harness can observe/control it. This
file explains what the patch is, where the files live, and how to change it.

## File roles (there should be exactly three SWFs)

| file | tracked? | role |
| --- | --- | --- |
| `static/game.swf` | yes | **pristine original** game (the Flex outer wrapper + embedded inner SWF). Never edit. |
| `solver/patch/game_inner.swf` | yes | **canonical built SWF** — the inner GameClass SWF with the AutoConnect patches applied. This is what gets served. |
| `local/web/game_inner.swf` | no (`local/` is gitignored) | **served copy** of the above — what Chrome/Ruffle loads over HTTP. |

The patch source is `solver/patch/Mafokem.as` (the decompiled + edited main
timeline, `kawai2_fla.Mafokem`). FFdec lives at `solver/tools/ffdec/`
(run `get_ffdec.sh` once). Needs Java 11+.

> **Do not create extra `game_inner*.swf` / `*.bak` files.** Past confusion came
> from stale builds and backups lying around. If you need to experiment, work in
> `solver/build/` (gitignored scratch) and clean up after.

## What the patch adds (`Mafokem.as`)

- **`frame1()`**: skips the title overlay → starts level 1 immediately on load,
  and calls `acInstallSolver()`.
- **Autonomous solver** — `acInstallSolver` / `acTick` / `acDoAction` /
  `acPlayOne`: a 200 ms `Timer` that reuses the game's own pair-finder
  (`vum.cunufi()`) and move handler (`byqij()`) to clear the board, then
  auto-advances/restarts across level-complete and game-complete screens.
- **Free reshuffles** in `createNewMap`: a greedy solver hits deadlocks; the
  patch reshuffles freely and never fails the run on a deadlock (time is never
  threatened — a level clears in ~10 s vs the 90–250 s limits).
- **ExternalInterface hooks** (registered in `acInstallSolver`, callable from JS
  as `document.getElementById('game').<method>(...)`, reached over CDP):

  | hook | purpose |
  | --- | --- |
  | `acStatus()` | JSON: `level, maxLevel, score, tilesLeft, scene, reason, clears, fails, ei, timeLeft, shuffles, hints, lastFail` |
  | `acSetEnabled(bool)` | pause/resume the internal solver Timer |
  | `acReset()` | zero counters, restart at level 1 |
  | `acGetClears()` | current full-clear count |
  | `acStep()` | do one solver action (not gated by `acEnabled`) and return `acStatus` |
  | `acPlayOne()` | remove one connectable pair |

- **`acInstallSolver()` is idempotent** (`if(acTimer != null) return;`). It lives
  in `frame1()`, which re-runs on every frame-0 entry; without the guard each
  re-entry re-enables the solver (overriding `acSetEnabled(false)`) **and** leaks
  a running `Timer`. See `memory/autoconnect-acinstallsolver-bug.md`.

## How to change the patch

1. Edit `solver/patch/Mafokem.as`.
2. Run the patcher (recompiles only `kawai2_fla.Mafokem` and re-imports it):
   ```bash
   ./solver/patch/patch_swf.sh        # patches solver/patch/game_inner.swf + deploys to local/web/
   python solver/driver.py --reload    # reload the page so the browser loads it
   ```
3. Verify: `python solver/driver.py --status` should print `ei=1` and a rising
   score (self-solving) / rising level.

`patch_swf.sh` aborts if the import is a no-op (see gotcha below).

## Build gotcha — never rebuild from scratch

The full from-scratch build (`build_patch.sh`: extract inner SWF →
`-renameInvalidIdentifiers randomWord` → `-importScript`) is **broken** and is
kept only for reference. The rename step is non-deterministic, so each run
produces different obfuscated names and the committed `Mafokem.as` (which uses
one particular rename's names) no longer matches a fresh rename — `-importScript`
silently produces a broken/empty SWF.

**Therefore the SWF is patched incrementally**: `patch_swf.sh` re-imports
`Mafokem.as` into the *existing* `solver/patch/game_inner.swf`, which already
carries the matching rename. Consequences:

- `solver/patch/game_inner.swf` is **precious and not regenerable from scratch**.
  It is committed to git on purpose (it's the only carrier of the rename). Don't
  delete it; if it's lost, restore it from git.
- A no-op import (output identical to input) means the names didn't match — you
  imported into the wrong base. The committed `Mafokem.as` matches
  `solver/patch/game_inner.swf`; it does **not** match any other SWF (e.g. an
  old `solver/build/*.swf` or a differently-renamed build).
- To **revert** a patch: `git show <commit>:solver/patch/Mafokem.as` to recover
  the old source, then run `patch_swf.sh` again.
