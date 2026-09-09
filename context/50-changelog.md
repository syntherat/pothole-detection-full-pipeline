# 50 — Changelog

**Append-only.** Every agent and contributor writes here on every change (Rule 5). Newest entries at the top.

This is not a git log — git already records *what* changed. This records **why**, what it means for the system, and what a future agent needs to know that the diff does not show.

---

## Entry format — copy this

```markdown
## YYYY-MM-DD — Short title
**Author:** <agent name / model / person>
**Scope:** <subsystem(s) touched>

**What changed**
- Concrete list of code/config/data changes.

**Why**
- The reason. If the user asked for it, say so. If it fixes a known issue, cite its number from 40-known-issues-and-gaps.md.

**Contracts affected**
- Any data shape from 20-data-contracts.md that changed, and every consumer updated. Write "None" if none.

**Context files updated**
- List them. Write "None" only if the change genuinely affected no documented behaviour, and say why.

**Verified**
- What you actually ran or checked, and the result. Distinguish verified from assumed.

**Notes for the next agent**
- Anything surprising, deferred, or newly discovered. Write "None" if nothing.
```

Rules for entries:

- One entry per logical change, not per file.
- **Never edit or delete an existing entry.** Corrections go in a new entry that references the old one.
- If you changed code and updated no context files, justify it explicitly — Rule 5 treats a code change with stale docs as incomplete.
- Be honest about what you did not verify. A future agent will trust this.

---

## 2026-09-08 — Level B P3: vision stage run on real pothole frames — 0/45 confirmed, domain gap measured
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/analyse_run.py`, context. No change to the vision or integration code.

**What changed**
- `carla_sim/analyse_run.py` — replaced the `SensorSession` + `iterrows` loop with `PotholeDetector` driven
  directly over `itertuples`. Same FSM, same order, same inputs; **1.9 s instead of >40 min** (issue #42).
- **New** `carla_sim/score_vision_run.py` — the script that produced the numbers below. It reports three
  values per frame, not one, because "no detections" has three different causes needing different fixes:
  raw detections before any road filtering (the domain gap alone), whether the road mask came from
  segmentation or the lower-60% fallback, and the score the orchestrator would actually see.

**What was measured — the vision stage, for the first time, on frames containing real potholes**
Run over the 45 picked frames of `levelB_vision` (15 events x 2.0/1.0/0.5 s lead). Results written to
`carla_sim/out/levelB_vision/vision_results.csv`.

| | result |
|---|---|
| **confirmed @ conf=0.35** (what the orchestrator sees) | **0 / 45** |
| any detection @ conf=0.05 | 25 / 45 |
| highest confidence anywhere in the set | **0.302** |
| frames where `road_seg.pt` predicted `visible_road` | **0 / 45** |

- **`best.pt` confirms nothing.** The domain gap the plan predicted is real and it is the blocker.
- **It is a near-miss, not blindness.** Peak 0.302 against a 0.35 gate, and the signal is ordered by
  distance: mean raw confidence **0.045 @ 2.0 s → 0.091 @ 1.0 s → 0.103 @ 0.5 s**, with every top-5 frame at
  0.5 s or 1.0 s lead. Random texture firing would not produce that gradient.
- **NOT verified: that those low-confidence boxes are spatially on the tiles.** The gradient is suggestive,
  not proof. Projecting the known 3D tile positions into image space is what would settle it — and that is
  the same code the auto-labelling fallback needs, so it is not wasted work.
- **Do not lower the gate to ~0.25 to manufacture detections.** With no false-positive measurement on CARLA
  frames, three detections at 0.25-0.30 says nothing about precision. Rule 6.

**The vision run is the same experiment as the bowls run — so frames and sensor rows describe the same holes**
`analyse_run.py` over `levelB_vision`: **7 FSM detections**, all true, at t = 7.080 / 32.295 / 32.700 /
38.685 / 44.252 / 44.545 / 44.657 s, az max 1616.74. Against `levelB_bowls`: 7 detections at 7.080 / 32.300
/ 32.700 / 38.685 / 44.255 / 44.545 / 44.657 s, az max 1616.86. Sub-5 ms differences from camera-induced
nondeterminism. **This is what closes issue #18** — for the first time frames and sensor rows come from one
recording of one real hole.

**Independent re-verification of the P2 pair**
Re-ran both runs after the speed fix, from the recorded CSVs: **bowls 7 detections (7 true, 0 false),
control 1 (1 true, 0 false)**, az max 1616.86 vs 380.80, DROP-band 398 vs 284, IMPACT-band 50 vs 19. Matches
the previous entry's table exactly. The 7-vs-1 result is now confirmed by a second agent from the raw data.

**Contracts affected**
None.

**Context files updated**
- `40-known-issues-and-gaps.md` — new issue #42; issue #30 extended with the CARLA measurement **and
  corrected** (see below).
- `04-current-state.md` — P3 status, next step.
- This file.

**A correction to `40-known-issues-and-gaps.md` issue #30**
It claimed `roadside_object` escapes the exclude set "because that set lists `roadside object` with a space
while the class name uses an underscore", and concluded the cascade would hunt for potholes inside roadside
objects if the mask ever populated. **That is wrong.** `_normalize_class_name()` does `.replace("_", " ")`
before matching, so `roadside_object` → `roadside object` and it *is* excluded. Verified by resolving all 7
class names through `_resolve_class_ids`: only `visible_road` is ever admitted, which is correct. The
substring matching is not a latent hazard and fix option (a) is moot. Corrected in place per Rule 1.

**Verified**
- Ran, not inferred: 45 frames through `VisionSession`, three runs of `analyse_run.py`, and the class
  resolution check. `vision_results.csv` is the raw per-frame output.
- **Not verified:** box-vs-tile spatial correspondence; any false-positive rate for vision on CARLA frames.

**Notes for the next agent**
- **The auto-labelling fallback is now the main path, not a contingency.** 1,200 frames, exact tile
  coordinates, labels derivable rather than hand-drawn. **Archive `best.pt` before any training run.**
- The frame-picking logic that produced `vision_picks.csv` exists only as that file — **no script in the
  repo writes it.** If the picks matter, the selection logic needs to become a committed script.
- `sensor_adapter.py` still has the per-row `predict_proba` cost (#42). `orchestrator.py` goes through it,
  so batch it before running the cascade over any CARLA-length recording.

---

## 2026-09-08 — Level B P2: sensor stage fires on REAL geometry; flat control proves the cavities cause it
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/scenario/tiles.py` (new), `carla_sim/analyse_run.py` (new),
`carla_sim/scenario/drive_and_record.py`, context

**What changed**
- **New** `carla_sim/scenario/tiles.py` — spawns Level B tiles at the Level A pothole locations, yaw-aligned
  to the lane, severity mapped onto the three real depths; `control=True` swaps in the flat tile.
- **New** `carla_sim/analyse_run.py` — runs the sensor stage over a recording, reporting detections **with
  the az envelope**, because "no detections" and "the wheel never fell far enough" need different fixes.
- `carla_sim/scenario/drive_and_record.py` — `--level A|B` and `--control`. Level B skips the impulse
  entirely; `PotholeTracker` still runs because it produces the ground-truth labels; tiles are destroyed in
  the `finally` block. **Level A's path is untouched.**

**Why**
- P2 is the point of Level B: find out whether the sensor stage fires on a physics-derived drop rather than
  Level A's scripted impulse.

**Verified — two runs, identical seed/route/tile positions/tick count, differing ONLY in whether the tiles
have cavities**

| | BOWLS | FLAT CONTROL |
|---|---|---|
| tiles / wheel-over events / labelled samples | 8 / 15 / 315 | 8 / 15 / 315 |
| az max | 1616.9 | 380.8 |
| below DROP (6.81) / above IMPACT (19.81) | 398 / 50 | 284 / 19 |
| \|az\| > 1000 after settling | **8** | **0** |
| **FSM detections** | **7** (7 true, 0 false) | **1** (1 true, 0 false) |

- **7 vs 1 — the cavities cause the detections, not the tile lip.** Everything else was held constant.
- Zero false positives in both runs.
- Every genuine large impact comes from a cavity: 8 post-settling \|az\| > 1000 samples vs **zero**.
- **The lip is not free**: the control still produced 1 detection and 284 DROP-band samples. Quote every
  future Level B figure against its control.

**A correction I made mid-analysis**
- I flagged an `az = -149443` spike as a possible tile artifact "worth chasing". It is not. It sits at
  **row 1, t = 0.0025 s, speed 0.19 m/s**, identical in both runs — the spawn settling transient this repo
  already documents ("row 1 carries az = -157477 ... skip the first few rows"). Checking it turned a vague
  worry into a cleaner result: excluding row 1 is what leaves the 8-vs-0 split.

**Two bugs of mine, found and fixed during the smoke test**
1. **`wait_for_tick()` deadlock.** The recorder drives the world SYNCHRONOUSLY, where only `world.tick()`
   advances time; `wait_for_tick()` blocked until the 120 s client timeout. `tiles.py` now branches on
   `world.get_settings().synchronous_mode`. The probe script had the same call and worked only because that
   world was asynchronous.
2. **Leaked actors on failure.** When that timeout fired *inside* `spawn_tiles`, the function never
   returned, so the caller's cleanup had nothing to destroy — **4 stray tiles were left in the world**, ready
   to contaminate the next run with unlabelled events. Found by inspecting world state, not by trusting the
   `finally` block. `spawn_tiles` now destroys its own actors before re-raising.

**NOT established (Rule 6)**
- **No vision result. The camera has still never seen a pothole** — both runs used `--no-camera`. That is the
  entire reason Level B exists, and it is the next step: record WITH frames and run the vision stage.
- Issue #18 (no end-to-end accuracy figure) remains open.
- **7 detections over 15 wheel-over events is NOT a recall figure.** Wheel-overs pair up (two wheels, one
  pothole), so the real denominator is nearer 8. Do not quote a rate from this run.

## 2026-09-08 — Level B tiles VERIFIED as real holes in CARLA's physics; FBX scale bug found and fixed
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/assets/`, context

**What changed**
- **New** `carla_sim/assets/probe_tile_collision.py` — spawns each imported tile in a live simulator and
  ray-probes it, the same method used on the 99 stock props.
- `generate_pothole_meshes.py` — **`FBX_GLOBAL_SCALE` 100.0 -> 1.0** (issue #41).
- `verify_pothole_meshes.py` — rewritten in metres, and its docstring now states plainly that it is
  **necessary but not sufficient**.
- `40-known-issues-and-gaps.md` — **new issue #41**.
- `15-carla-testbed-plan.md` — P1 records the in-engine verification; the Environment table's
  "set FBX scale 100" instruction corrected.

**Why**
- P1's tiles were unproven: a bowl in a mesh is worthless if CARLA seals it with convex collision, which is
  what happened to all 99 stock props.

**Verified — in a running simulator, by measurement**
| blueprint | rim | floor | depth measured | depth authored | hit% |
|---|---|---|---|---|---|
| `static.prop.potholetile_deep` | 0.120 | 0.010 | **0.110 m** | 0.11 | 0.85 |
| `static.prop.potholetile_medium` | 0.080 | 0.010 | **0.070 m** | 0.07 | 0.85 |
| `static.prop.potholetile_shallow` | 0.050 | 0.010 | **0.040 m** | 0.04 | 0.85 |
| `static.prop.potholetile_flatcontrol` | 0.080 | 0.080 | **0.000 m** | 0.00 | 1.00 |

- **The bowls are genuinely empty in CARLA's physics** — the exact property all 99 stock props lacked
  (they returned depth 0.00 at hit-fraction 1.00, rays hitting a convex lid that does not visually exist).
- The flat control at **0.000** is the negative control: the probe is not manufacturing depth from tile
  thickness or ramp geometry.
- Independent evidence from the import: **40** `Triangulating mesh UCX_... for collision model` lines
  = 13 + 13 + 13 + 1, exactly the authored decomposition.
- All four props register as `static.prop.potholetile_*` blueprints.

**The bug this session's most important lesson (issue #41)**
- Tiles first imported at **160 m x 160 m x 14 m** — 100x oversized. `Import.py` hardcodes
  `bConvertSceneUnit: 1`, so UE converts metres->centimetres itself; exporting at `global_scale=100` (as the
  plan instructed) made it happen twice.
- **`verify_pothole_meshes.py` PASSED while this was true**, reporting "160.0 cm across, the
  metres-vs-centimetres trap avoided". An FBX is internally consistent at either scale, so inspecting the
  file could never catch it. Only spawning the asset and reading `actor.bounding_box` could.
- **Verify artifacts in their real context, not in isolation.** Checking the artifact instead of the exit
  code was the right instinct and still was not enough here.

**NOT established — do not overstate (Rule 6)**
- That a *wheel* drops in. Ray geometry is necessary, not sufficient: CARLA's raycast wheels have zero width
  and the suspension response is a separate question.
- That the FSM fires on the resulting signature. Needs a recorded drive (P2).
- The lip is real and unavoidable at Level B — the Deep tile is an 11 cm hole behind a 12 cm ramp. **Drive
  `PotholeTile_FlatControl` alongside every experiment** and subtract it, or detections are confounded by
  the tile's own edge.

**Notes for the next agent**
- Working sequence: refresh `Import/` from `carla_sim/assets/fbx/`, `python Util\BuildTools\Import.py`,
  `python carla_sim/assets/fix_package_paths.py`, launch, then `probe_tile_collision.py`.
- **A modal RenderDoc dialog blocks every launch.** `CarlaUE4.uproject` enables `RenderDocPlugin` and
  RenderDoc is not installed, so UE prompts "Locate main RenderDoc executable..." and **waits**. Click
  Cancel. This will silently hang any unattended launch or import. Disabling the plugin is a one-line edit
  to a CARLA-tracked file; left to the user's discretion, not done.
- To get an RPC server without a human pressing Play in the editor:
  `UE4Editor.exe CarlaUE4.uproject -game -windowed -carla-rpc-port=2000`. First run against uncooked content
  is very slow (skeletal meshes, texture compression, ~12k shaders); the second is far quicker once the DDC
  is warm (33 -> 15,376 files across this session).

## 2026-09-08 — Pothole tiles imported into CARLA; three silent-failure traps in the import path
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/assets/` (one new file), context

**What changed**
- **New** `carla_sim/assets/fix_package_paths.py` — rewrites the asset paths in
  `PavePotholes.Package.json` to match what was actually imported. Must run after every import.
- `carla_sim/assets/generate_pothole_meshes.py` — manifest now emits `size` (`PROP_SIZE = "Medium"`).
- `40-known-issues-and-gaps.md` — **new issue #40**, the three import traps.
- `15-carla-testbed-plan.md` — P1 records the import, the working command sequence, and what is still unverified.

**Why**
- P1 needs the tiles inside CARLA as spawnable props, not just as FBX on disk.

**Contracts affected**
- None of the documented internal contracts. External: CARLA's `Import.py` props JSON now carries
  `name` / `source` / `size` / `tag`; `size` is required.

**Verified (by artifact — every failure below returned success)**
- `Import.py` exit 0; both import commandlets `Success - 0 error(s)`.
- Four `.uasset` files under `Content/PavePotholes/Static/static/`, plus
  `Config/PavePotholes.Package.json` listing all four props.
- **One `.uasset` per tile, not fourteen** — so UE consumed the 13 `UCX_` bodies as collision rather than
  importing them as separate meshes. First positive evidence the collision decomposition survived import.
- Paths in the package file corrected from `X.X` to the real `X_X.X_X` form.

**Three traps, each of which reported success at the point of failure**
1. **`make import` imports nothing and exits 0.** `Windows.mk` runs `"Util/BuildTools/Import.py"` as a bare
   command with a forward-slash path via the `.py` file association; cmd does not resolve it. Zero output,
   zero work. Run `python Util\BuildTools\Import.py` instead.
2. **`KeyError: 'size'` after a successful mesh import.** The FBX imported fine, then
   `generate_package_file()` crashed, so no `.Package.json` was written and the props were never registered
   as blueprints. The content tree looked populated and correct.
3. **Registered paths pointed at non-existent assets.** `Import.py` assumes a single-mesh FBX and derives the
   object path from the file name. An FBX carrying `UCX_` bodies has multiple root nodes, so UE names the
   asset `<fbx>_<mesh>` and the registration is wrong. The prop would have failed to spawn with no error at
   any stage.

**NOT verified — do not claim otherwise (Rule 6)**
- **That the imported collision is genuinely concave in-engine.** UE consuming the UCX bodies is necessary
  but not sufficient. The decisive test is the `world.cast_ray` grid probe used on the 99 stock props on
  2026-09-07, run against a spawned tile in a live simulator. Until it passes, these are candidate holes.

**Notes for the next agent**
- Import sequence that works, in order: refresh `Import/` from `carla_sim/assets/fbx/`, run
  `python Util\BuildTools\Import.py`, then `python carla_sim/assets/fix_package_paths.py`.
- Import.py rewrites `PavePotholes.Package.json` on every run, so the fixup is not a one-off.
- The engine's **boot DDC is full** (`Maximum cache size reached. CurrentSize 524276 kb / MaxSize: 524288`),
  so some shader work is discarded and recompiled each launch. The main DDC (~9,100 files, ~4 GB) is fine.
  Raise `BootDerivedDataCache` only if repeated shader compiles become a real cost.
- Shader compilation is entirely local — no internet needed once `Setup.bat` and packman have run.

## 2026-09-08 — Level B P1 started: pothole tiles generated in Blender and measured
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/assets/` (new), context

**What changed**
- **New** `carla_sim/assets/generate_pothole_meshes.py` — parametric Level B tile generator, runs headless in
  Blender. Emits 4 FBX plus `PavePotholes.json` in CARLA's `Import.py` props format.
- **New** `carla_sim/assets/verify_pothole_meshes.py` — re-imports each FBX and measures scale, bowl depth,
  UCX body count, and whether the bowl interior is actually empty.
- **New** `carla_sim/assets/fbx/` — the generated output.
- `15-carla-testbed-plan.md` — new "P1 — pothole assets" section; phase table marks P1 STARTED.
- `02-repository-map.md` — new rows for `assets/`; the stale "Level A scaffold, never run" header replaced.

**Why**
- P1 is the remaining Level B work now that the source build is verified. Level B exists to give the camera
  and the suspension a real hole, which is the only path to an end-to-end accuracy number (issue #18).

**Contracts affected**
- None of the documented data contracts. Adds one external contract: CARLA's `Import.py` props JSON
  (`name` / `source` / `tag`), landing at `/Game/PavePotholes/Static/static/<name>`.

**Verified (by measurement, re-importing the FBX)**
- `PotholeTile_Shallow` 3.9 cm deep / 22 cm radius, `PotholeTile_Medium` 6.9 / 28,
  `PotholeTile_Deep` 10.9 / 34, `PotholeTile_FlatControl` 0.0 (control). Declared depths are 4 / 7 / 11 cm;
  the 0.1 cm shortfall is grid discretisation at the bowl centre, within the 1 cm tolerance.
- All four measure **160.0 cm** across — FBX scale-100 conversion correct.
- **No collision body encloses the space above the bowl floor**, so the bowl is genuinely empty.

**Design decisions, each traceable to a prior measurement**
1. **Collision is authored as explicit `UCX_` convex bodies** (12 rim wedges + a floor slab), not left to UE's
   hulling. The 2026-09-07 prop probe found 0 of 99 stock props had a usable cavity because CARLA props get
   sealed convex collision. A single hull would make these tiles bumps rather than holes.
2. **A flat control tile is shipped deliberately.** A prop sits ON the road and cannot cut into it, so the
   deepest hole a tile can present equals its own thickness — a 10.9 cm pothole implies a 10.9 cm lip. The
   control isolates the lip's own IMU contribution so it can be subtracted instead of assumed away. This is
   the "tile edge lips" risk in the plan, and the reason Level C stays higher fidelity.
3. Bowl radii >= 0.10 m, and the generator refuses anything narrower: CARLA's raycast wheels have zero width
   and would drop into holes a real tyre bridges. The plan says document the bias, not tune it out.

**Notes for the next agent**
- **The verifier initially failed all four tiles, and the bug was in the verifier.** It sampled the tile's
  flat underside along with the top surface, overstating every depth by exactly `SKIRT` (2 cm). The tell was
  that all four were wrong by the same constant, and that the flat control reported a 10 cm "bowl" when its
  thickness is 8 cm. Fixed, with the failure mode recorded in a comment. Generator `[OK]` output proves
  nothing on its own — run the verifier.
- Blender is invoked by full path (`D:\Program Files\Blender Foundation\Blender 5.2\`); it is not on PATH.
- Tunables live at the top of the generator: `TILE_SIZE`, `SKIRT`, `RAMP_WIDTH`, `GRID`, `UCX_SECTORS`, and
  the `VARIANTS` table. Changing a depth requires the thickness to exceed it, which the generator enforces.

## 2026-09-08 — Blender was installed all along; corrects every "Blender is absent" claim
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only.

**What changed**
- `04-current-state.md`, `15-carla-testbed-plan.md`, `40-known-issues-and-gaps.md` — every claim that Blender
  needs installing replaced with its real location.

**Why**
- The user questioned the claim. It was wrong.

**Verified**
- `D:\Program Files\Blender Foundation\Blender 5.2\blender.exe` → **Blender 5.2.0 LTS**
- `D:\Program Files\Blender Foundation\Blender 4.4\blender.exe` → **Blender 4.4.1**
- Neither is on PATH; invoke by full path. Found via the uninstall registry keys, which named both instantly.

**CORRECTION**
- **"Blender is absent" — WRONG**, recorded in four context files since 2026-09-06. The audit checked only
  `C:\Program Files\Blender Foundation` and PATH. Blender is on **D:**.
- **This is the second instance of the same error pattern**, after missing the VS 2022 **BuildTools** install
  at `%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools` (which led to the v142/v143 mistake).
  Both times: probed two or three default paths, found nothing, recorded absence as fact.

**Notes for the next agent**
- **Never conclude a Windows program is absent from a filesystem probe of default paths.** Query the
  uninstall registry first:
  ```powershell
  foreach($k in @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
                  'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
                  'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*')){
    Get-ItemProperty $k -EA SilentlyContinue | ? { $_.DisplayName -match 'NAME' } |
      Select DisplayName, DisplayVersion, InstallLocation }
  ```
  Add `Get-AppxPackage` for Store apps. This machine installs software to **D:\Program Files**, so
  `%ProgramFiles%`-only checks are unreliable here.
- Level B P1 is therefore **not blocked on an install** — the pothole meshes can be authored now. Use
  5.2.0 LTS unless the FBX exporter misbehaves, in which case 4.4.1 is the fallback. Remember FBX scale 100
  (Blender metres vs Unreal centimetres).

## 2026-09-08 — CARLA source build CONFIRMED WORKING: the editor opens and renders Town10HD_Opt
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no repo code touched.

**What changed**
- `04-current-state.md` and `15-carla-testbed-plan.md` — Level B's build phase marked **verified**, not just
  "builds cleanly". Next step is now the content work (P1), not `make launch`.
- `40-known-issues-and-gaps.md` — issue #38 gains the first-run timing data and the authoritative
  progress source.

**Why**
- `make launch` had built cleanly twice but nobody had seen a working viewport. It has now been observed.

**Verified**
- **The CarlaUE4 editor opens and renders `Town10HD_Opt`**, **4,013 actors** in the World Outliner,
  viewport drawing the town. `MainWindowTitle = "CarlaUE4 - Unreal Editor"`, `Responding = True`.
- Post-load work still draining at the time of writing: **Compiling Shaders (13,567)** and
  **Building Mesh Distance Fields (468)**. The editor is usable while these run.
- This validates the whole chain built from source: UE 4.26 engine + Program targets → LibCarla →
  CarlaUE4 plugin (752/752 actions) → PythonAPI wheel → editor rendering a CARLA map.

**First-run timings on this machine (all one-off, cached afterwards)**
- Engine editor target: ~89 min. CarlaUE4 plugin: ~27 min.
- **Asset discovery: `LogAssetRegistry: Asset discovery search completed in 1920.5251 seconds` (~32 min).**
  This is what the splash spends most of its time on while parked at 95 %, scanning the 43,359 content files.
- Then mesh distance fields, then the remaining shaders.
- DerivedDataCache grew 33 → 9,029+ files (~4 GB). An interrupted run resumes from it; a force-kill loses
  only in-flight work.

**Notes for the next agent**
- **The authoritative progress source is `Unreal/CarlaUE4/Saved/Logs/CarlaUE4.log`** — timestamped, names the
  current phase, and reports how long each took. Use it instead of inferring from CPU counters, which is what
  earlier sessions wasted effort on. The splash percentage is near-useless: it jumps 39 % → 95 % and then sits
  through asset discovery, distance fields and shaders.
- **On first open the editor asks "Project file is out of date. Would you like to update it?" — answer
  "Not Now".** Letting UE rewrite `CarlaUE4.uproject` changes engine association and plugin entries in a file
  CARLA ships and tracks; it is a needless way to break `make launch` later. "New plugins are available" can
  be dismissed.
- Shader/distance-field counters appear in the editor's bottom-right once the window is up, and *those* are
  real progress bars — unlike the splash.

## 2026-09-08 — `make launch` reaches shader compilation; the engine build was incomplete
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no repo code touched.

**What changed**
- `40-known-issues-and-gaps.md` — **new issue #38** (`Build.bat UE4Editor` does not build the engine's
  Program targets) and **new issue #39** (stale `CMakeCache.txt` blocks a generator change tree-wide).
- `15-carla-testbed-plan.md` and `04-current-state.md` — the "engine built" claim **qualified**: it covered
  the editor target only.

**Why**
- `make launch` failed twice: once on a stale CMake cache, then on a missing `ShaderCompileWorker`. Both were
  consequences of earlier decisions of mine that the context files recorded as clean successes.

**Contracts affected**
- None.

**Verified**
- `Build.bat CarlaUE4Editor` via `make launch`: **752/752 actions, 1628 s (~27 min), zero errors.**
- Engine Program targets built 2026-09-08, zero errors: `ShaderCompileWorker.exe` (320,000 B),
  `UnrealLightmass.exe` (1,143,296 B), `UnrealPak.exe` (177,664 B).
- After that, `make launch` reached the editor and **13 `ShaderCompileWorker` processes** ran at ~126 s of
  combined CPU per 12 s of wall clock, machine at 100 %, editor `Responding: True`. Shader compilation was
  still in progress when this entry was written — **the editor has not yet been confirmed usable.**

**CORRECTIONS to earlier entries (left unedited, per the append-only rule)**
1. **"UE 4.26 built successfully… 4238/4238 actions, zero errors" — INCOMPLETE.** True of the *editor
   target*; the engine was not fully built. `Build.bat UE4Editor` does not build the Program targets, and
   without `ShaderCompileWorker.exe` the editor cannot compile any shader — it started, printed
   `Unable to launch .../ShaderCompileWorker.exe`, and exited. **The verification was too weak:** checking
   that `UE4Editor.exe` and 417 DLLs existed does not establish that the engine is complete. Check all four
   binaries (see issue #38).
2. **"EDITOR LAUNCHED" was reported from process existence alone — that is not success.** `UE4Editor.exe`
   sat at 1.2 GB working set having already failed. Judge by `ShaderCompileWorker` count and accumulating
   CPU instead.

**Notes for the next agent**
- **The editor splash parks at `Initializing. 39%` for the whole shader phase.** A frozen percentage there is
  normal, not a hang. Real signals: `ShaderCompileWorker` process count and their CPU delta. DDC file counts
  are *not* a good indicator — UE batches those writes.
- **Shader compilation is cached.** Killing the editor mid-compile loses only in-flight shaders; the
  `DerivedDataCache` keeps what finished, so a later `make launch` resumes rather than restarting.
- The stale-cache sweep in issue #39 should be run **whenever the generator changes**, not one directory at
  a time as failures surface. Fixing only the dirs that had already failed is what caused the second
  `make launch` failure.
- Still not done: **`make package`** (needs `UnrealPak`, now built) and the Level B content work — Blender
  tiles → FBX → `make import`. **Blender is still not installed.**

## 2026-09-07 — `make PythonAPI` succeeds; corrects the v142 and MSYS claims recorded earlier
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no repo code touched. CARLA lives outside the repo at `D:\dev\carla-source`.

**What changed**
- `40-known-issues-and-gaps.md` — **new issue #37** (CARLA's Windows build scripts report success on
  failure). Inside resolved #36: the **v142 claim corrected to v143**, and the `vswhere` note annotated with
  its real cause.
- `15-carla-testbed-plan.md` — **new section "CARLA source build on this machine — the working recipe"**
  with all eight fixes; the MSYS misattribution corrected; the v142 row and the "three unknowns settled"
  paragraph corrected.
- `04-current-state.md` — Level B advanced to "`make PythonAPI` succeeds, next is `make launch`".

**Why**
- Level B needs a working CARLA source build. It now builds. Two facts recorded earlier as settled were
  wrong and would have misled the next agent.

**Contracts affected**
- None.

**Verified (by artifact, not exit code)**
- `PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl`, **5,444,427 bytes**, containing
  `carla/libcarla.cp312-win_amd64.pyd` at **17,665,536 bytes**. **Zero** LNK/compile errors on the final run.
- `PythonAPI/carla/dependencies/include/OSM2ODR.h` present; `dependencies/lib/xerces-c_3.lib` (Release)
  present.
- boost built with the v143 toolset: `libboost_python312-vc143-mt-x64-1_84.lib` and four others.

**CORRECTIONS to earlier entries (which stay unedited, per the append-only rule)**
1. **"v142 inside VS 2022 satisfies UE 4.26's UnrealBuildTool" — WRONG.** UBT's own log:
   `Using Visual Studio 2019 14.44.35228 toolchain (C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207)`.
   It used **v143**, from a **BuildTools** instance the 2026-09-06 audit never scanned — that audit checked
   Professional under `%ProgramFiles%` and looked for a `2019` folder, but never
   `%ProgramFiles(x86)%\Microsoft Visual Studio\2022`. CARLA agrees (`--boost-toolset msvc-14.3`).
   **The v142 component installed on 2026-09-06 was probably never needed.**
2. **"Git Bash `cmd //c` mangles arguments" — WRONG.** `'Setup.bat' is not recognized` was caused by
   `NoDefaultCurrentDirectoryInExePath=1`, inherited by every spawned shell, which stops cmd resolving
   executables in the current directory. Moving to PowerShell only appeared to fix it because absolute paths
   were adopted at the same time. The same variable also broke vcvarsall's `vswhere` lookup — which is what
   made CMake's Visual Studio generator unable to find a compiler, and therefore what triggered the whole
   NMake detour below.

**Notes for the next agent**
- **The NMake detour was a self-inflicted cascade.** Forcing `GENERATOR="NMake Makefiles"` (to work around a
  problem that was really correction #2) caused three further failures: osm2odr's `-A x64` rejected, Xerces
  silently built **Debug** because NMake is single-config and ignores `--config Release` (→ 50 unresolved
  `xercesc_3_2` externals), and a single-threaded LibCarla build. **Use CARLA's default
  `Visual Studio 17 2022` generator.** Lesson: when a root cause is found, revert the workarounds it justified.
- **boost.numpy cannot build against NumPy 2.** Build boost alone with `PYTHONNOUSERSITE=1`; CARLA never uses
  boost_numpy. Do **not** set that flag for the whole build — setuptools, wheel and pip are user-site here too.
- **boost 1.84 cannot auto-detect the BuildTools edition of VS 2022**; it needs an explicit `user-config.jam`.
- Clearing a poisoned b2 config cache matters: after the toolset was fixed, cached checks still read
  `default address-model: none` / `cxx11_static_assert: no`. Deleting `boost-1.84.0-source/build` made them
  read `64-bit` / `yes`.
- Writing Windows paths into these context files from a Python heredoc keeps corrupting them via octal
  escapes (`\14` → form feed, `\202` → 0x82, `\7z` → BEL). **Stage the text in a file and splice it**, and
  scan for control bytes afterwards.

## 2026-09-07 — CARLA 0.9.16 source + 20 GB content pack in place; `Update.bat` is a trap on a flaky link
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no repo code touched. CARLA source lives outside the repo at `D:\dev\carla-source`.

**What changed**
- `15-carla-testbed-plan.md` and `04-current-state.md` — Level B progress: CARLA source cloned, content
  installed, `UE4_ROOT` set. Next action is `make PythonAPI` / `make launch`.

**Why**
- Continuing the Level B source build after the engine build succeeded.

**Contracts affected**
- None.

**Verified**
- `carla-simulator/carla` cloned shallow at tag **`0.9.16`** (HEAD `294096e`, `git describe` = `0.9.16`) to
  **`D:\dev\carla-source`**; clean tree, 2.5 GB, Git LFS smudged 14 files.
- **`UE4_ROOT` = `D:\dev\UnrealEngine_4.26`** (User scope), and the target verified to actually contain
  `UE4Editor.exe` — not merely to exist.
- Content pack **`20250912_2171890.tar.gz`**, server `Content-Length` **21,567,002,106 bytes (20.09 GB)**,
  downloaded byte-exact (expected == actual).
- Extraction, both 7-Zip passes reported **"Everything is Ok"**: `.tar.gz` → 25,120,921,600-byte tar →
  **1,379 folders / 43,359 files**. Final tree **23.36 GB**. Both archives deleted. D: **175.6 GB** free.
- Top-level content dirs: `Blueprints`, `Config`, `HDMaps`, `HoudiniEngine`, `Maps`, `Static`, `road_xodr`,
  `hooks`, `LICENSE`.

**Notes for the next agent — `Update.bat` will destroy your download**
- **`Update.bat` downloads with `(New-Object System.Net.WebClient).DownloadFile(...)`. It has no resume, it
  does have a timeout, and its `:bad_exit` path runs `rd /s /q "%CONTENT_FOLDER%"` — so ANY failure deletes
  everything already fetched.** On a 20 GB download this is a coin flip on the whole transfer window staying
  clean. Two attempts were lost this way: one hung at 481 MB after a network switch, one died at ~7 GB with
  `"The operation has timed out."` and wiped the folder.
- **Working approach:** fetch the tarball with `curl -L -C - --retry 15 --retry-all-errors --speed-time 120
  --speed-limit 10240` in a loop that compares the local size against the server's `Content-Length`, then run
  `Update.bat`'s own extraction by hand — 7-Zip twice, `.tar.gz` → `.tar` → content, deleting each source.
  `.version` is declared in the script but never written, so nothing else needs reproducing.
- **If you must kill a running `Update.bat`, kill the parent `cmd` FIRST, then the PowerShell child.** Killing
  the child alone lets the batch fall through to `:bad_exit` and delete the partial download.
- **Speed was never the issue.** curl and WebClient both sustained 4–6 MB/s. An early curl reading of
  105 KB/s was measured during a degraded-network window, not CDN throttling of ranged requests — the same
  URL later ran the full 20 GB in one clean pass at ~5 MB/s.

## 2026-09-07 — UE 4.26 (CARLA fork) built successfully on Windows; all three toolchain unknowns settled
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no repo code touched. The engine lives outside the repo at `D:\dev\UnrealEngine_4.26`.

**What changed**
- `15-carla-testbed-plan.md` — Level B section records the engine build as done, with the measured numbers
  and the two operational traps found on the way.
- `40-known-issues-and-gaps.md` — resolved issue #36's three "never verified" items are now **verified**.
- `04-current-state.md` — Level B item advanced: engine built, next action is `UE4_ROOT` + the CARLA clone.

**Why**
- Level B needs a CARLA source build, which needs UE 4.26 from Epic's CARLA fork built from source. This is
  the long pole and it is now behind us.

**Contracts affected**
- None.

**Verified**
- Clone: `CarlaUnreal/UnrealEngine`, branch **`carla`**, HEAD **`e9d9e60c8`** ("Add VS2026 compatibility for
  removed MSVC extensions"), shallow (`--depth 1`), 138,005 files, **1.81 GB**.
- `Setup.bat`: **11,731 MiB** of dependencies over 63,398 files, completed 100 %, then "Registering git
  hooks" / "Installing prerequisites", exit 0. Tree after setup: **49.4 GB**.
- `GenerateProjectFiles.bat`: exit 0. Produced `UE4.sln` and `Engine/Binaries/DotNET/UnrealBuildTool.exe`.
  The generated project references toolsets **v141, v142 and v147**; solution is VS2019 format
  (`VisualStudioVersion = 16.0`).
- **Build: `Build.bat UE4Editor Win64 Development -WaitMutex` → exit 0.** **4238/4238 actions**,
  **5357.29 s total (~89 min)**, parallel executor 5224.51 s. **Zero** lines matching
  `error C####|fatal error|LINK : fatal|ERROR:` across 6,918 output lines.
- Artifacts: `UE4Editor.exe`, `UE4Editor-Cmd.exe`, `UE4Editor.target`, **417 DLLs** in
  `Engine/Binaries/Win64`.
- Disk after build: tree **104.51 GB**, D: **201.6 GB** free, C: 25.6 GB.

**Three previously-unverified items — now settled by a successful compile**
1. **v142 inside VS 2022 satisfies UE 4.26's UnrealBuildTool.** It does. 4,238 compile/link actions passed.
2. **Windows 10 SDK 10.0.26100 is acceptable** to this 4.26-era build. The concern that it wanted ~10.0.18362
   did not materialise.
3. **Absent `clang` is irrelevant on Windows.** The build never needed it.

**Notes for the next agent**
- **Windows Defender was throttling the build tooling by roughly four orders of magnitude.** During
  `Setup.bat`'s dependency check the process showed **38 KB/s** of I/O, 199 metadata ops/s and 3.4 % CPU —
  no TCP connections, no disk reads, just a live process going nowhere. After the user added an exclusion
  for `D:\dev\UnrealEngine_4.26`, the same process jumped to **415 MB/s** and **88 % CPU**. If any Unreal or
  CARLA step ever crawls with low CPU and high "IO Other Operations/sec", check exclusions first. Add one
  for the CARLA source tree before building it.
- **`Setup.bat`'s download can hang with the process still alive.** It froze at 90 % (10609.6/11731.4 MiB)
  with 0.00 MiB/s for 604 consecutive progress lines. Killing `GitDependencies` and re-running resumed and
  only re-fetched the remaining 1,126 MiB — it does **not** restart from zero. Diagnose by watching whether
  the byte count advances, not whether the process exists.
- **Do not launch `.bat` files through Git Bash `cmd //c`** — MSYS mangles the arguments and it fails with
  `'Setup.bat' is not recognized`. Use PowerShell: `Set-Location <dir>; & cmd /c '.\Setup.bat'`.
- A global `%APPDATA%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml` on this machine contains a UE5-era
  `MemoryPerActionBytes` element that 4.26's schema rejects. It is a warning only; UBT ignores it and builds.

## 2026-09-07 — No stock CARLA prop can fake a pothole; the source build is unavoidable
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no code touched. Throwaway probe scripts left in the session scratchpad, not the repo.

**What changed**
- `15-carla-testbed-plan.md` — the "worth 30 minutes" prop shortcut replaced with the measured finding;
  **P0 item 3 (prop spawning) marked VERIFIED**.
- `40-known-issues-and-gaps.md` — resolved issue #36's "Not yet attempted" note replaced with the result.
- `04-current-state.md` — Level B item: shortcut closed off, source build confirmed as the only path.

**Why**
- The user chose to run this check before committing an afternoon to the CARLA source build. If a stock prop
  had a usable cavity, Level B's vision test could have happened with no build at all. It does not.

**Contracts affected**
- None.

**Verified**
- Packaged 0.9.16, Town10HD_Opt, live simulator. **99 `static.prop.*` blueprints; 99/99 spawn successfully**
  at a map spawn point — this verifies P0 item 3.
- 66 props have footprint ≥ 0.30 m and height ≥ 0.15 m; 50 are pothole-scale (≤ 3.0 m, height 0.15–2.5 m).
- 9×9 `world.cast_ray` grid down through each: **not one has a usable cavity.** Every visually-open
  container (`bin`, `container`, `clothcontainer`, `box01`–`03`, all `plantpot*`, `trashcan04/05`) returns
  **depth 0.00 at hit-fraction 1.00** — rays hit one uniform top surface and never enter the opening.
  `open_interior_cells` = **0 for every solid prop**.
- Nonzero depths are hull artifacts: `warningaccident`/`warningconstruction` 0.98 m is an A-frame with rays
  passing between the legs (hit 0.78); `doghouse` 0.28 m sloping roof; `trashbag` 0.27, `glasscontainer`
  0.19, `trashcan01/03` 0.15, `trafficcone02` 0.14, `barrel` 0.11 are curvature.
- Only see-through props have interior gaps — `shoppingtrolley` (7 % hit), `plasticbag` (1 %),
  `plastictable` (between legs). None would catch a wheel.
- World left clean: 0 stray actors after the probe.

**Assumed, NOT verified**
- That the collision volumes are literally convex hulls. This is **inferred from the ray pattern**, not read
  from the assets' collision flags. A 100 % hit rate across the mouth of an open bin admits no other
  explanation, but it was not confirmed at the asset level.

**Notes for the next agent**
- **Do not re-run this probe hoping for a different prop.** The answer is structural, not a matter of
  searching harder. Beyond collision convexity there is a geometric argument that no prop can beat: the road
  surface stays put, so a prop sits *on* it. The best possible case is rim-then-cavity — climb a lip, drop
  in, climb out — which contaminates the IMU signature with an entry bump and still does not read as a hole
  in tarmac to a camera. This is the same lip artifact the Level B risk table flags for Blender tiles, and
  it is the reason Level C (holes cut into the road mesh) stays the highest-fidelity option.
- `world.cast_ray`, `project_point`, `ground_projection` and `get_level_bbs` all exist on the 0.9.16 client
  and work. `cast_ray` returns multiple `LabelledPoint`s per ray; take the highest hit above ground to get
  the top surface. It is a good tool for interrogating map geometry without driving anything.
- CARLA was left running after the probe.

## 2026-09-07 — Level B toolchain complete: make 3.81 installed, issue #36 resolved
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no code touched

**What changed**
- `40-known-issues-and-gaps.md` — **issue #36 moved to Resolved.** Rewritten as a closed record with the
  final measured state plus the two traps worth keeping (the `vswhere -requires` misreporting, and the
  broken chocolatey `make` package with its working alternative).
- `15-carla-testbed-plan.md` — Level B table: `make` row flipped to ✅; risk row updated to say the
  toolchain is complete.
- `04-current-state.md` — Level B item rewritten: no blockers, with the concrete build sequence, the disk
  headroom warning, and the three unverified items to check if the build complains.

**Why**
- The user installed GNU Make 3.81, closing the last toolchain prerequisite for Level B. Supersedes the two
  entries below, which are left unedited per the append-only rule.

**Contracts affected**
- None.

**Verified**
- `C:\GnuWin32\bin\make.exe` reports **GNU Make 3.81** (2006 build, i386-pc-mingw32). Confirmed both by
  invoking the binary directly and by the user in a fresh shell after the PATH change.
- `C:\GnuWin32\bin` is at **index 0** of the Machine PATH. The chocolatey `make` shim is gone
  (`choco uninstall make` succeeded, removing 4.4.1).
- Machine PATH backed up to `D:\dev\tools\path-backup.txt` (2022 bytes) before the edit.
- Installer identified as **Inno Setup** by scanning the binary for the signature in both ASCII and UTF-16;
  that is what made the silent flags predictable. Copy kept at `D:\dev\tools\make-3.81-setup.exe`.
- Disk re-measured 2026-09-07: **C: 25.7 GB, D: 306.5 GB** free.
- Unchanged: Blender still absent.

**Assumed, NOT verified**
- Carried forward unchanged from the previous two entries: whether absent `clang` matters on Windows;
  whether Windows 10 SDK 10.0.26100 is acceptable to a UE 4.26-era build; whether v142-inside-VS-2022
  satisfies UnrealBuildTool. All three are now testable — the build is the test.

**Notes for the next agent**
- **The chocolatey `make` 3.81 package is broken; do not retry it.** It saves an Inno Setup `.exe` as
  `...Install.zip` and hands it to 7-Zip, which cannot unpack Inno in any case. The download is valid —
  run it directly with `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="C:\GnuWin32"`.
- **PATH ordering had to be Machine, not User.** Windows composes System entries before User ones, so a
  User entry cannot outrank a system-level chocolatey shim. This is why the first instinct (User PATH)
  would have silently failed.
- Install `make` to a **space-free** directory; `make` and `C:\Program Files (x86)` interact badly.
- Blender is the only Level B item still missing, and it gates mesh authoring (P1), not the build.

## 2026-09-06 — Level B toolchain: C++ workload and v142 installed; corrects the entry below
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no code touched

**What changed**
- `40-known-issues-and-gaps.md` — **issue #36** retitled and rewritten. Its blockers 1 and 2 (no C++
  compiler; no v142 toolset) are **closed**; `make` 3.81 is now the only blocker. Added a warning that
  `vswhere -requires` misreports on this machine, and a new unverified item about the Windows SDK version.
- `15-carla-testbed-plan.md` — Level B audit table and the source-build risk row updated to match.
- `04-current-state.md` — the Level B next-step item updated.

**Why**
- The user installed the Desktop-development-with-C++ workload and the MSVC v142 toolset after the audit
  earlier in this session, and asked whether it was already done. It was. This entry corrects the previous
  entry ("Level B prerequisite audit: go, but no C++ compiler on the machine") rather than editing it,
  per the append-only rule.

**Contracts affected**
- None.

**Context files updated**
- `40-known-issues-and-gaps.md`, `15-carla-testbed-plan.md`, `04-current-state.md`, this file.

**Verified**
- `VC\Tools\MSVC\` now holds **14.29.30133** and **14.44.35207**. `cl.exe` from the 14.29 tree reports
  **"Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30159 for x64"** — MSVC 14.29 = **v142 =
  VS 2019 16.11**, which is what UE 4.26 requires. `Microsoft.VisualStudio.Workload.NativeDesktop` resolves
  to 17.14.37314.3.
- The earlier "absent" finding was correct when made: two independent probes (`ls` on `VC\Tools\MSVC` and a
  recursive `find` for `cl.exe`) both returned empty at audit time. The install happened in between.
- Re-checked and unchanged: `make` is still GNU Make **4.4.1** at `C:\ProgramData\chocolatey\bin\make`, no
  GnuWin32 make on disk; Blender still absent; 7-Zip 26.03 still present.

**Assumed, NOT verified**
- Still open from the previous entry: whether absent `clang` matters on Windows, and whether the
  v142-inside-VS-2022 arrangement actually satisfies UE 4.26's UnrealBuildTool. The toolset is now present,
  so the second one is finally testable — but it has not been tested.
- **New:** only Windows 10 SDK **10.0.26100.0** is installed. UE 4.26-era builds commonly expect something
  closer to 10.0.18362. Untested; if UBT complains about the SDK, look here first.

**Notes for the next agent**
- **`vswhere -requires <componentId>` cannot be trusted on this install.** It reported
  `...VC.14.29.16.11.x86.x64`, `...VC.v142.x86.x64` and `...Windows10SDK.19041` as ABSENT while a working
  v142 `cl.exe` sat on disk. Diagnose from `VC\Tools\MSVC\<ver>\bin\HostX64\x64\cl.exe --version` instead;
  a 19.29.x banner means v142. `-property` queries (installationPath, displayName, packages) were fine.
- C: needs re-measuring before the build. It was 25.8 GB free *before* the VS workload landed.

## 2026-09-06 — Level B prerequisite audit: go, but no C++ compiler on the machine (new issue #36)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** context only — no code touched

**What changed**
- `40-known-issues-and-gaps.md` — new **issue #36** `[ENV]`, recording the full audit: what is verified
  present, what is missing, one explicitly unverified assumption, and one cheap alternative not yet tried.
- `15-carla-testbed-plan.md` — new **"LEVEL B — prerequisite audit (2026-09-06)"** section before the shared
  phase plan; **Status** line changed from "Level A scaffold written, never executed" (stale — Level A has
  run) to "Level A runs; Level B is prerequisite-blocked"; the source-build risk row now carries the measured
  findings instead of generic advice.
- `04-current-state.md` — the "Level B" next-step item rewritten with the audit result; header date note
  updated.

**Why**
- The user asked for a go/no-go check before committing to a ~165 GB source build, then asked for it to be
  recorded so it is not re-derived next session. Level B is the top-priority item because it is the only
  level that can test vision.

**Contracts affected**
- None.

**Context files updated**
- `40-known-issues-and-gaps.md`, `15-carla-testbed-plan.md`, `04-current-state.md`, this file.

**Verified**
- **Epic/CARLA GitHub access works** — `git ls-remote` against the *private* `CarlaUnreal/UnrealEngine`
  returned HEAD `2ac0528`; branches `4.26` and `carla` both exist; `gh auth status` shows `syntherat` with
  `repo` scope. This was the plan's highest-likelihood risk and it is retired. Tag `0.9.16` exists on
  `carla-simulator/carla`.
- **No C++ toolchain at all** — `VC\Tools\MSVC\` under VS 2022 Professional 17.14 is empty and a recursive
  find for `cl.exe` returned nothing. `vswhere -requires` for the v141/v142 component returned empty. No
  `C:\Program Files (x86)\Microsoft Visual Studio\2019` directory.
- Disk via `Get-CimInstance Win32_LogicalDisk`: **C: 25.8 GB free, D: 306.5 GB free** (`wmic` is gone on
  Win11 26200 — use the CIM cmdlet).
- `make` = GNU Make **4.4.1** at `C:\ProgramData\chocolatey\bin\make`; CMake 3.31.11; Git 2.49.0; Python
  3.12.10 AMD64; Windows 8.1 SDK present; .NET 4.6.2 targeting pack present; Blender absent.
- 7-Zip **26.03** at `C:\Program Files\7-Zip\7z.exe`, on PATH — installed by the user during this session
  and re-checked after.
- Packaged CARLA at `D:\dev\pothole` measures **18.1 GB** and is independent of any source build.

**Assumed, NOT verified**
- That `clang` being absent does not matter on Windows (the belief is that CARLA builds LibCarla with MSVC
  and the clang requirement is Linux-only). **Not checked against the 0.9.16 Windows docs.** Confirm before
  relying on it.
- That installing the v142 component inside the VS 2022 installer is sufficient for UE 4.26's
  UnrealBuildTool to find a usable toolchain. This is the documented workaround but has not been exercised
  on this machine.

**Notes for the next agent**
- The original plan framed the Visual Studio question as a *version* mismatch. It is an *absence* — that
  correction is recorded in both #36 and the plan file (Rule 1).
- **C: is the tight resource, not D:.** The VS C++ workload and v142 toolset install to C:, which has 25.8 GB
  free and needs 7–10 GB of it. Everything else belongs on D:.
- **Untried and cheap:** whether a stock `static.prop.*` blueprint has concave collision usable as a real
  dip. Would give the camera a visible obstacle with no source build. Needs a running simulator, ~30 min,
  and overlaps with P0 item 3. Expect it to fail — most props are convex — but it is worth the half hour
  against a day of compiling.
- Writing Windows paths into these files from a Python heredoc **corrupted three of them** via octal escapes
  (`\7z` → BEL, `\bin` → backspace, `\2019` → `\x819`). They were caught with `cat -A` and repaired. Use raw
  strings or stage the text in a file.

## 2026-09-06 — The detector now fires on data it did not generate (resolves #35)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `pothole_detect_physics/detector_py/`, `carla_sim/`, context

**What changed**
- `detector_py/pothole_detection.py` — FREEFALL no longer resets on samples between rest and the impact
  gate; it waits, bounded by `max_air_time`.
- `carla_sim/scenario/impulse.py` — new `schedule_unload()` plus `_apply_at_point()`. Holds a downward force
  at the striking wheel to cancel the suspension reaction, modelling a wheel falling INTO a hole rather than
  being kicked downward.
- `carla_sim/config.py` — `UNLOAD_ENABLED`, `UNLOAD_TICKS = 19`, `UNLOAD_FORCE_SCALE = 3.2`, with the full
  tuning sweep recorded in comments.
- `carla_sim/scenario/drive_and_record.py` — uses unload *or* impulse, never both.

**Why**
- The sensor stage produced zero detections on every CARLA recording. Both defects had to be fixed; either
  alone leaves it silent.

**Contracts affected**
- **Contract #2** (`process_sample()` return): shape unchanged, **behaviour changed** — it now fires on
  continuously-rising rebounds that previously produced no event. Noted in `20-data-contracts.md`. No
  consumer required changes.

**Context files updated**
- `10-physics-sensor-pipeline.md`, `20-data-contracts.md`, `21-configuration-and-tuning.md`,
  `40-known-issues-and-gaps.md` (#35 resolved).

**Verified**
- CARLA: sensor stage fires, scores 0.90 and 0.74, on a strike whose trace shows drop → 6 samples of
  freefall → impact at 25.07.
- Synthetic regression: event recall **79.69 % → 80.73 %**, precision 100 %, spurious 0.
- `pytest integration/test_integration.py`: **9 passed, 1 skipped** — unchanged.
- Impulse-vs-unload and force-vs-duration sweeps recorded in `21-configuration-and-tuning.md`.

**Notes for the next agent**
- **The circularity was real and mechanical.** The old FSM could only fire on a discontinuity that
  `generate_dataset.py` wrote by hand. 33.3 % of CARLA pothole samples hit the old reset band versus 4.8 %
  of synthetic ones. Any future "the detector works" claim should say which data it works on.
- **Do not timestamp the event at the rebound peak.** It is more correct physically and it halves cascade
  recall, because the RandomForest filter then sees an ordinary-looking row. The detector fires more while
  the system detects less — invisible unless both stages are measured.
- `UNLOAD_FORCE_SCALE` and `UNLOAD_TICKS` are coupled; changing one alone reliably breaks one of the two
  gates. The sweep table shows why.
- Vision is still untestable in Level A. This fix validates the sensor stage only.

---

## 2026-09-06 — CARLA replay feeds the dashboard a vehicle (resolves #28, finds #35)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `integration/`, `pothole_map_ui/`, `carla_sim/out/`, context

**What changed**
- **New** `integration/carla_replay.py` — walks a recorded run's `gnss.csv` in time order, publishes
  **contract #12** (`vehicle_position.json`), and emits sensor-stage markers through
  `pave_connector.send_record_to_pave()`. Flags: `--speed`, `--reset`, `--loop`.
- `pothole_map_ui/app.js` — polls that file every 200 ms in CARLA mode, moves the car marker and calls
  `checkProximity()`. Three rendering bugs also fixed: `backgroundColor` at construction, a corrected
  `CARLA_STYLE`, and a double `fitBounds`.
- Exported `carla_sim/out/Town03_roads.geojson` (503 segments, 572 KB) from the live simulator.

**Why**
- The user asked to see the car driving the town and marking potholes. That was issue #28.

**Contracts affected**
- **New contract #12**, documented in `20-data-contracts.md`. Contract #7 unchanged — the replay goes through
  the existing single writer rather than writing the events file itself (Rule 3).

**Context files updated**
- `20-data-contracts.md` (#12), `14-integration-layer.md`, `02-repository-map.md`,
  `40-known-issues-and-gaps.md` (#28 resolved, #35 added).

**Verified**
- Replay end to end on `run_20260906_195639`: 40,000 GNSS fixes read, position file published, replay
  completes. Total run time **11 s** after batching the classifier.
- CARLA mode rendered in a real browser with a real API key: Town03's 503 segments drawn large on a dark
  ground. All three rendering bugs confirmed fixed **by screenshot**, having previously passed stub tests
  while being visibly wrong.
- Stub suite: **18/18**.

**Notes for the next agent**
- **#35 is the blocker now.** The replay reports **0 sensor-stage detections** on the first recording, and it
  is not a plumbing fault: the CARLA impulse overshoots the FSM's `|az| < 2.0` freefall window, hitting a
  median of −64.6 instead. Only 8 rows in 40,000 land inside that window. Lower `IMPULSE_DELTA_V` (try
  0.1-0.15) and re-record.
- **A wrong assumption corrected in the same session:** a Level A run was expected to confirm nothing
  *because vision cannot see a data-only pothole*. True, but not the operative reason — the sensor stage
  never fires, so vision is never reached. That mis-attribution would have sent someone to the wrong
  subsystem.
- `SensorSession.check_row()` is ~25 minutes per 40,000 rows because it calls `predict_proba` per row inside
  a fresh DataFrame. `carla_replay` batches it to 11 s. The orchestrator still pays the per-row cost — worth
  fixing there too, but it is a Rule 4 hot path and was left alone.
- Markers from a Level A replay are labelled `sensor (CARLA replay)`. Do not relabel them
  `hybrid (sensor+vision)` — there is no visible pothole for the camera to confirm.

---

## 2026-09-06 — Adopted stages A and B of the reviewed change log (finds #33, fixes #34)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `pothole_detect_physics/` (Model + detector_py), context

**What changed**
- **New** `Model/features.py` — shared feature construction so training and evaluation cannot drift apart:
  `ensure_event_ids`, `grouping_key`, `add_rolling_features`, `full_feature_columns`.
- `Model/train_ai_model.py` — rewritten. `GroupShuffleSplit` by event (item 4.1), optional `--rolling`
  features (item 4.3), and both the leaky and honest scores reported side by side. Stays a RandomForest.
  Writes the rolling model to a **separate** path.
- `Model/run_detector_on_dataset.py` — rewritten. Event-level scoring (item 4.2), row-level kept only for
  contrast, and a guard that refuses to run when the model's `n_features_in_` disagrees with the features
  built.
- `detector_py/generate_dataset.py` — emits `event_id` / `event_type`, seeded with `RANDOM_SEED = 42`, and
  clears labels under an overwriting speed breaker (#34).
- **Not changed:** the committed dataset, `pothole_ai_model.pkl`, and every integration module. The live
  cascade behaves exactly as before.

**Why**
- The user supplied `PAVE_v2_change_log_REVIEWED.docx` and asked for stages A and B. All four of the
  document's factual claims about the code were verified against source before any change.

**Contracts affected**
- **Contract #1** (`synthetic_pothole_dataset.csv` columns) gains `event_id` and `event_type`. Additive and
  optional: the committed CSV lacks them, `ensure_event_ids()` derives them, consumers select by name, and
  `carla_sim`'s `sensors.csv` stays column-compatible. `20-data-contracts.md` updated.
- Contract #3 (RandomForest feature vector) is **unchanged for the live model**. The rolling model is a
  separate artefact precisely because it would break it — see below.

**Context files updated**
- `20-data-contracts.md`, `10-physics-sensor-pipeline.md`, `40-known-issues-and-gaps.md` (#33, #34),
  `50-changelog.md`.

**Verified**
- Leakage is real and accuracy hides it. Same data, same model, two splits:

  | metric | leaky (row split) | honest (grouped) |
  |---|---|---|
  | accuracy | 99.41 % | 99.30 % |
  | **recall (pothole)** | **85.81 %** | **80.36 %** |
  | precision (pothole) | 93.53 % | 96.66 % |

- Rolling features, same honest split: recall **80.36 % → 96.23 %**, precision **96.66 % → 100 %**. The
  leaky-vs-honest recall gap collapses from 5.45 points to 0.59, i.e. the features carry real temporal
  signal rather than memorised event fragments.
- Event-level cascade scoring: **153 / 192 events, recall 79.69 %, precision 100 %, 0 spurious firings.**
  The row-level view of the same run reads TP 2040 / FP 28 / FN 321.
- The generator's three consistency invariants now hold: every `label == 1` row is tagged `pothole`, every
  speed-breaker row has `label == 0`, every background row has `event_id == -1`. Tested in an isolated copy
  so the committed dataset was never touched.

**Notes for the next agent**
- **#33 is the finding that matters, and it is not in the document.** The AI filter confirms 154 of 154
  physics candidates — it rejects nothing. So the classifier improvement from 80 % to 96 % recall moves
  end-to-end cascade recall by **zero**. Never quote the classifier's recall as a system number.
- The rolling model is at `Data/pothole_ai_model_rolling.pkl` and is **deliberately not wired in**.
  `integration/sensor_adapter.py` hardcodes the seven raw columns and builds a one-row DataFrame per sample,
  keeping no history, so a rolling model would be handed columns that do not exist. Wiring it in means
  teaching the adapter to buffer — a Rule 4 change, not done here.
- `Data/pothole_ai_model_grouped.pkl` is the raw-feature model retrained on the honest split. Also not
  promoted; `pothole_ai_model.pkl` is untouched so the 40-event demo stays reproducible.
- The committed dataset still carries the #34 mislabelling. It was left alone deliberately — regenerating
  breaks the pickled model, and the pre-seed file is unreproducible anyway.
- Stages C (depth honesty) and D (remove hand-shaped pulses, add refractory) were **not** done.

---

## 2026-09-06 — CARLA P0 complete; map exporter validated against a live simulator (finds #32)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `carla_sim/config.py`, context

**What changed**
- `carla_sim/config.py`: `WHEEL_POSITION_SCALE = 0.01`, `IMU_GRAVITY_AT_REST = 9.3483` — both measured, no
  longer `None`. `CARLA_TIMEOUT_S` 20.0 → 120.0.
- No other code changed. `export_map.py` and `impulse.py` ran as written.

**Why**
- The user installed CARLA 0.9.16. P0 was the designated next step and the gravity convention was the
  documented "could cost days" risk.

**Contracts affected**
- None.

**Context files updated**
- `21-configuration-and-tuning.md` (new CARLA section with the measured values and the 9.35-vs-9.81 gate
  offset), `30-setup-and-run.md` (working launch command + the `-quality-level=Low` warning),
  `40-known-issues-and-gaps.md` (#32), `15-carla-testbed-plan.md` (status header),
  `04-current-state.md` (step 3 done).

**Verified**
- `verify_setup.py` against live CARLA 0.9.16: **az = +9.3483 at rest** (positive — the FSM's assumption
  holds), wheel positions in **centimetres**, `add_impulse_at_location` **absent** (the `hasattr` fallback in
  `impulse.py` covers it), 400 Hz achievable at **0.235× real time**.
- Server 0.9.16 / client 0.9.16 — versions match.
- **`scenario/export_map.py` run against the live simulator** — previously stub-tested only. Produced
  `carla_sim/out/Town10HD_Opt_roads.geojson`, 209 KB: 200 LineString features, 3,761 coordinate points,
  82/200 flagged as junctions, every coordinate a `[lng, lat]` float pair, and the shipped `bounds` block
  independently re-checked to cover every point.
- The exported bounds are **lat -0.000618..0.001269, lng -0.001030..0.000988** — i.e. the town really does
  sit on (0, 0). The premise behind CARLA map mode is now confirmed by measurement, not by documentation.
- **Not verified:** CARLA mode has still never been *rendered*, because no Google Maps key is configured.
  `drive_and_record.py` has still never run.

**Notes for the next agent**
- See #32 before debugging any CARLA connection problem. `-quality-level=Low` crashes this build, the RPC
  port opens before the crash, and the resulting client error names the timeout rather than the crash. I
  raised `CARLA_TIMEOUT_S` on that false diagnosis; the comment in `config.py` says so rather than quietly
  keeping the number.
- `IMU_GRAVITY_AT_REST` is **9.35, not 9.81**, and the FSM gates are absolute constants. DROP is ~15 % harder
  to reach than the thresholds assume. Tune `IMPULSE_DELTA_V` against a real trace before concluding the
  impulse model is wrong.
- The exporter defaults to whichever town is loaded. Town10HD_Opt was exported because a `load_world` attempt
  had already crashed the simulator once; export Town03 (which `config.TOWN` names) once a run is recorded
  there.

---

## 2026-09-06 — orchestrator.py made runnable for the first time (fixes #31)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `integration/orchestrator.py`

**What changed**
- `import argparse` — it was used in `main()` and never imported.
- `run()` signature now matches what `main()` has always passed and what
  `14-integration-layer.md` has always documented: `run(limit, dataset_path=None, frame_provider=None)`.
- Frame and GPS now come from the provider object (`MockFrameProvider` by default, `CarlaFrameProvider`
  for `--carla-run`) instead of the module-level `get_mock_frame` / `simulate_gps`.
- `skipped_no_frame` initialised; new `skipped_no_gps` counter reported separately.
- Import swapped from `get_mock_frame, simulate_gps` to `MockFrameProvider`.

**Why**
- The CLI could not start at all. Discovered by actually running the documented command while CARLA
  downloaded. Defect 2 in particular blocked `--carla-run`, which is the project's own designated next step.

**Contracts affected**
- None. The `PotholeCandidateEvent` shape, `pave_events.json` and the provider interface are unchanged —
  the provider interface is now *used* rather than merely defined.

**Context files updated**
- `14-integration-layer.md` (documents `skipped_no_gps`, plus a note that the documented signature was
  aspirational until today), `40-known-issues-and-gaps.md` (#31, resolved), `04-current-state.md`.

**Verified**
- `python integration/orchestrator.py --help` renders all three flags.
- `pytest integration/test_integration.py -q` → **9 passed, 1 skipped**, unchanged by the fix.
- A live run over 40,000 synthetic rows produced real confirmed events in `integration/pave_events.json`,
  in contract #7 shape: `confidence` populated, `detected_by = "hybrid (sensor+vision)"` (not the
  mislabelled default from #19), `lat`/`lng` at the Bhopal mock origin.
- **Not verified:** the `--carla-run` branch. `CarlaFrameProvider` is constructed there and no recorded run
  exists yet, so that path is still untested end to end. It is now *reachable*, which it was not before.

**Notes for the next agent**
- The bug pattern is worth remembering: the feature existed in `main()`, in the printed summary, and in the
  context docs — everywhere except the function that had to implement it. Reading any one of those would
  have left you confident the feature worked.
- This is the first time the cascade has been observed running end to end: sensor FSM → RandomForest → gate
  → stand-in frame → YOLO → fusion → `pave_events.json`.
- The confidence numbers it emits are **not** accuracy. #18 (no frame↔sensor sync) and #30 (stage 1 inert)
  both still stand, and the frame paired with each event is unrelated to the row that triggered it.

---

## 2026-09-06 — Stand-ins swapped to dashcam frames; stage 1 found to be inert (resolves #29, finds #30)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** `pothole_detection_app/data/sample_images/` (gitignored content), context

**What changed**
- Replaced the 13 close-up Kerala positives with **16 dash-camera frames** from
  `Ryukijano/Pothole-detection-Yolov8` (roboflow export, 640x640, MIT-adjacent roboflow terms, YOLO labels
  shipped alongside). Kept the 4 Kerala `none_*` controls. Pool is now 20 images: 16 positive, 4 control.
- No source code changed.

**Why**
- #29: the close-ups were the wrong camera geometry and the stage-1 fallback was cropping the potholes out
  of frame. The user asked for the swap.

**Contracts affected**
- None.

**Context files updated**
- `40-known-issues-and-gaps.md` (#29 resolved with before/after numbers; #30 added), `CLAUDE.md`
  (two-stage claim corrected).

**Verified**
- Cascade at conf=0.35 on the new pool: **10/16** dashcam frames confirmed (was 1/13), **0/4** controls
  falsely flagged.
- **Localisation scored against the shipped YOLO labels:** 16/33 labelled potholes matched at IoU >= 0.3,
  per-frame best IoU typically 0.6-0.89. This is the check that distinguishes "fires the right number of
  times" from "fires in the right place".
- `road_seg.pt` run directly over all 16 frames at conf=0.05: **zero** `visible_road` predictions. It emits
  `roadside_object` on every frame, plus `vehicle`, `pedestrian`, `road_obstacle`, `shadow`.
- Stage-1 fallback fires on **20/20** images (fd-level capture).

**Notes for the next agent**
- **The headline is #30, not the swap.** The cascade has never used road segmentation. Every detection the
  system has produced came from the lower-60% crop. It looked fine because on dash-camera geometry that crop
  is a decent proxy for "the road" — a correct-looking result from the wrong mechanism.
- There is also a substring-matching defect in `_resolve_class_ids()`: `roadside_object` and `road_obstacle`
  both match the `"road"` include keyword. Fixing that **alone would make things worse**, turning an empty
  mask into a mask over roadside objects. Read #30 before touching it.
- 16/33 localisation is not a recall figure for the detector either — it is 16 stand-in frames against a
  checkpoint whose training set is unknown to us. #18 still blocks any real accuracy claim.

---

## 2026-09-06 — Vision cascade measured on the stand-in photos (finds #29)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** measurement only — no code changed

**What changed**
- No code. This entry records a measurement and its correction, so the next agent does not repeat it.

**Why**
- The 17 stand-in photos added earlier the same day had never been put through the detector. Adding images
  and assuming they work is exactly the pattern the verification-debt section warns about.

**Contracts affected**
- None.

**Context files updated**
- `40-known-issues-and-gaps.md` (new issue #29).

**Verified**
- Real `best.pt` + `road_seg.pt`, loaded on GPU, over all 17 images.
- Cascade at conf=0.35: **1/13** potholes confirmed, **0/4** clean-road false positives.
- Raw `best.pt` on the full frame at conf=0.05 finds boxes in **11/17** images; the full cascade in **3/17**.
  The stage-1 fallback is removing true positives on this imagery.
- `road_seg.pt` produced no usable mask on **17/17** images.
- mean-vs-max scoring in `vision_adapter` changes the verdict on exactly one image at conf=0.50 — ruled out
  as the cause.

**Notes for the next agent**
- **A correction to my own first pass, recorded because the wrong answer was convincing.** I initially
  measured the stage-1 fallback with `contextlib.redirect_stdout` and got 0/17 — "stage 1 is fine". That was
  an artefact: the warning is emitted through `logger`, not Python-level stdout, so the redirect captured
  nothing while the messages leaked to the terminal in plain sight. Re-measured with file-descriptor-level
  capture (`os.dup2`) it is 17/17. If you are counting log lines emitted by library code, capture at fd
  level or you will measure your own redirect.
- The stand-in images are close-up phone photos, not dash-camera frames. That mismatch, not model quality,
  is what produced 1/13. Do not quote that number as a recall figure.
- This affects only the non-CARLA demo path. `carla_frame_provider.py` supplies real timestamp-matched
  frames, so CARLA runs never touch this image pool.

---

## 2026-09-06 — Windows environment stood up; CARLA map mode added (decides #3, fixes #27, finds #28)
**Author:** Claude Opus 5 (Claude Code)
**Scope:** environment, `carla_sim/`, `pothole_map_ui/`, context

**What changed**
- New `carla_sim/scenario/export_map.py` — walks `world.get_map().get_topology()`, traces each segment via
  `waypoint.next(step)`, converts with `transform_to_geolocation()` and writes a GeoJSON `FeatureCollection`
  to `carla_sim/out/<Town>_roads.geojson`. Ships a `properties.bounds` block so the dashboard fits its
  viewport instead of guessing a zoom.
- `pothole_map_ui/app.js` — added a URL-selected **CARLA mode** (`?mode=carla&town=Town03`). Hoisted the
  existing dark tile theme to `REAL_STYLE`, added `CARLA_STYLE` (every Google feature off), and added
  `loadCarlaRoads()`. 412 → 497 lines.
- `pothole_detection_app/data/sample_images/` — 17 real road photos added (Kerala, from the MIT-licensed
  HF dataset `Arpitraj01/Pothole_classification`), named by severity: 4 `none`, 5 `low`, 4 `medium`,
  4 `severe`. The `none` images matter — an all-pothole pool cannot distinguish a working detector from one
  that always says yes.
- `pothole_map_ui/config.js` created from the example (still carries the placeholder key).
- A `venv/` on Python 3.12.10 with both dependency sets.

**Why**
- The user moved from a MacBook Air to the Windows machine, which is the only one that can run CARLA.
- The user asked for the CARLA town drawn on the dashboard, and explicitly for the real-world Google Maps
  path to keep working — hence a *mode*, not a replacement. This settles open decision #3 in
  `15-carla-testbed-plan.md`.

**Contracts affected**
- None. `PotholeGuard.reportDetection`, `pave_events.json` and the `sensors.csv` schema are all untouched.
  CARLA mode changes only the basemap layer, which is what keeps it inside Rule 3.

**Context files updated**
- `02-repository-map.md` (export_map.py), `13-map-ui.md` (CARLA mode section + new constants),
  `15-carla-testbed-plan.md` (decision #3 resolved), `30-setup-and-run.md` (Windows CUDA torch, pytest
  target, export command), `40-known-issues-and-gaps.md` (#27 → Resolved, #7 amended, #28 added),
  `04-current-state.md`, and `CLAUDE.md` (stale FileNotFoundError claim corrected).

**Verified**
- `torch 2.14.0+cu126`, `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 4060 Laptop GPU`.
- `pytest integration/test_integration.py` → **9 passed, 1 skipped** (was 8 passed / 2 skipped; a test that
  used to skip for missing deps now runs).
- **Issue #27 measured, not assumed:** `pothole_ai_model.pkl` loads as a `RandomForestClassifier` with
  **0** warnings under `joblib 1.6.0` + `numpy 2.5.2`, counted via `catch_warnings(record=True)` with
  `simplefilter('always')`. The 1602-warning figure was real for `joblib 1.5.3`; it is fixed upstream.
- `app.js` CARLA mode: **14/14** checks against a stubbed Google Maps API in Node — real mode still uses the
  tile style, still starts geolocation, draws no GeoJSON; CARLA mode blanks the basemap, skips geolocation,
  draws on both maps, fits the exported bounds, styles junctions distinctly, keeps `PotholeGuard` working,
  and fails loudly on a missing file. `node --check` clean; CRLF preserved.
- `export_map.py`: **12/12** checks against a stubbed `carla` module — including that coordinates come out
  `[lng, lat]` and not `[lat, lng]`, and that a cyclic lane graph hits `MAX_STEPS_PER_SEGMENT` instead of
  hanging.
- `carla-0.9.16-cp312-cp312-win_amd64` installed and imported. Every API `export_map.py` calls exists on
  the real client: `Client`, `GeoLocation`, `Map.get_topology`, `Map.transform_to_geolocation`,
  `Waypoint.next`. So the API surface is verified against the real library, not only against the stub.
- **Not verified:** `export_map.py` has never touched a running simulator, and CARLA mode has never been
  rendered by real Google Maps (no API key on this machine yet). Both test harnesses live in the session
  scratchpad, not the repo.

**Notes for the next agent**
- **`pip install -r pothole_detection_app/requirements.txt` gives you CPU-only torch on Windows.** PyPI
  bundles CUDA only in Linux wheels. Install from `https://download.pytorch.org/whl/cu126` first. The sole
  symptom otherwise is `cuda False`.
- **A bare `pytest` from the repo root runs ZERO tests.** `pothole_detection_app/test_detection.py` calls
  `sys.exit(1)` at import, which kills the collector with `INTERNALERROR`. This has presumably been masking
  the suite for a while. Run `pytest integration/test_integration.py`. Amended into issue #7.
- **New issue #28:** CARLA mode has no car marker and no proximity alerts, because there is no vehicle
  position feed. Deliberate — browser GPS would put the car in Bhopal while the roads sit near (0, 0). It
  means CARLA mode currently demonstrates detection and mapping but *not* the driver-alert half.
- opencv crossed to **5.0.0** and pandas to **3.0.5**. Neither is exercised by the integration suite beyond
  imports, so the GUIs and training scripts remain unverified against those majors.
- The `road_seg.pt` / `best.pt` models were never loaded this session. The GUI smoke tests from the previous
  handoff are still outstanding.
- **CARLA version correction:** the plan recommended 0.9.15, but there is no `cp312` wheel for it. On
  Python 3.12 the client must be **0.9.16**, and the downloaded simulator package has to match.
  `15-carla-testbed-plan.md` updated accordingly.
- CARLA mode still renders *through* Google Maps — it blanks the tiles but Google is the engine, so it
  needs the API key just as real-world mode does. If a keyless CARLA mode is ever wanted, the basemap
  choice is isolated in `initMaps()` and `loadCarlaRoads()`, which is the only place that would change.

---

## 2026-08-20 — Root README rewritten; handoff doc added
**Author:** Claude (Opus 5), via Claude Code
**Scope:** root `README.md`, new `context/04-current-state.md`, index and pointer updates

**What changed**
- **Rewrote the root `README.md`.** It now matches reality: correct folder names, all five subsystems
  including `integration/` and `carla_sim/`, both shipped models, the `config.js` key flow, a
  what-is-real-vs-simulated status table, per-event accuracy figures with the caveat, and a tuning
  section. Removed the model-performance table that presented upstream YOLO architecture ranges as
  though they measured `best.pt` (Rule 6).
- **Added `context/04-current-state.md`** — a mutable where-we-are document: the verification debt
  table, the ordered next steps, the open decisions, and a paste-able prompt for starting a fresh
  session. Distinct from `50-changelog.md`, which stays append-only history.
- Indexed it in `context/README.md` (subsystem table + routing) and inserted it into the boot sequence
  in `CLAUDE.md`.
- Added a gotcha to `CLAUDE.md`: `context/`, `CLAUDE.md` and `AGENTS.md` are gitignored on purpose, so
  `git status` reads clean despite uncommitted context edits.

**Why**
The user asked for an accurate README and a handoff they can resume from in another chat.

**Contracts affected**
None.

**Context files updated**
`02-repository-map.md` (README caveats section rewritten), `04-current-state.md` (new),
`README.md` (index + routing), `40-known-issues-and-gaps.md` (#15 now partially resolved), `CLAUDE.md`,
and this file.

**Verified**
- All cross-file links resolve.
- Every factual claim in the new README checked against the repo: folder names, both model files and
  sizes, the sklearn pin, the `--limit` flag, the `--watch` adapter invocation, the serve-from-root
  requirement, and the current test output.
- **Deliberate omission:** the root README does **not** reference `context/`, `CLAUDE.md` or
  `AGENTS.md`. Those are gitignored, so anyone cloning the repo would find broken references.

**Notes for the next agent**
- **Keep `context/` out of the root README.** It ships; the knowledge base does not.
- `04-current-state.md` is meant to be **rewritten**, not appended to. If it no longer describes the
  present, fix it — a stale current-state file is worse than none.
- `pothole_detection_app/README.md` is still stale and still carries the misleading performance table.
  It was left alone; rewriting it is a reasonable next tidy-up.

---

## 2026-08-20 — Missing sample images degrade instead of crashing (fixes #2 and #10)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `integration/frame_provider.py`, `integration/orchestrator.py`,
`integration/test_integration.py`, root `.gitignore`, new
`pothole_detection_app/data/sample_images/README.md`

**What changed**
- **`MockFrameProvider.get_frame()` returns `None`** when no stand-in images exist, matching
  `CarlaFrameProvider`, and warns once. `get_mock_frame()` keeps its raising contract, so the existing
  test stays meaningful.
- **Added discovery fallbacks.** `CANDIDATE_IMAGE_DIRS` searches `data/sample_images/`, then `input/`,
  then the `dataset_v3` and `dataset_v2` test/val splits, using the first that contains images and
  logging which once.
- **Sorted the image pool.** `Path.glob()` order is filesystem-dependent and the pool is indexed into,
  so the same `event_id` could pick different images on different machines.
- **Replaced `hash()` with MD5** for the index — fixes #10 in the same function.
- **Created `data/sample_images/` with a tracked `README.md`**, and narrowed the root `.gitignore` from
  the whole directory to its contents so the README ships while photos stay ignored.
- Orchestrator now prints an actionable note when *every* triggered row was skipped.
- Two regression tests: degradation-returns-None, and stability across processes.

**Why**
Fixes #2, which was the top item on the priority list: the orchestrator died on its first triggered
event on any fresh clone.

The framing matters. **I cannot supply road photographs**, and generating synthetic ones would be
worse than useless — YOLO would detect nothing, every event would be rejected, and the pipeline would
look like it worked while proving nothing (Rule 6). So the fix is not "add the images", it is:
their absence is now a clearly-reported degradation rather than a crash, anyone who already has a
dataset needs no setup at all, and the folder now explains what to put there and what a mock run does
and does not prove.

#10 came along for free: I was rewriting the indexing line anyway, and leaving a docstring that
promised reproducibility it could not deliver would have been sloppy.

**Contracts affected**
None. `MockFrameProvider.get_frame()` returning `None` brings it *into* line with the provider
interface `CarlaFrameProvider` already implemented and the orchestrator already handled.

**Context files updated**
`01-project-overview.md`, `02-repository-map.md`, `14-integration-layer.md`, `30-setup-and-run.md`,
`40-known-issues-and-gaps.md` (#2 and #10 to Resolved, priority renumbered), and this file.

**Verified by execution**
- No images anywhere: `get_frame()` returns `None` twice while warning **once**; `get_mock_frame()`
  still raises `FileNotFoundError` with the full searched-paths message.
- Images in a fallback dir: discovery finds exactly the 3 image files, excludes a `.txt`, and returns
  them sorted.
- `data/sample_images/` takes precedence over the fallback when both have images.
- **Cross-process determinism:** `get_mock_frame('fixed-event-id')` run in three subprocesses under
  `PYTHONHASHSEED` 0, 1 and 12345 returned the identical path every time. Under the old `hash()` this
  was not achievable.
- `git check-ignore`: `README.md` tracked, `probe.jpg` ignored.
- Full suite: **8 passed, 2 skipped** (the two new tests skip when no images are configured, which is
  the state of a fresh clone).

**Notes for the next agent**
- The two new tests **skip** on a bare clone. If you add sample images they start running — that is
  intended, not flaky.
- Discovery order is deliberate: `sample_images/` first so an explicit choice always beats a fallback.
- **This does not close #18.** The chosen image still has no relationship to the sensor event that
  triggered it, so `vision_score` and `final_confidence` from a mock run remain meaningless. Only real
  synced data (a vehicle, or `carla_sim/`) fixes that. The new README says so explicitly, because a
  demo that now runs cleanly end to end is exactly the situation where someone starts quoting its
  numbers.

---

## 2026-08-20 — Filtered GUI events now reach the map (fixes #19)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `integration/` (1 new, 1 modified), `pothole_detection_app/app/pothole_app_filtered.py`,
`pothole_map_ui/app.js`

**What changed**
- **New** `integration/filtered_gui_adapter.py` — translates the filtered GUI's `events.jsonl` into
  contract #7 and publishes it. One-shot, or `--watch` to follow the file while the GUI is still
  writing. Idempotent: seeds its seen-set from what is already in `pave_events.json`.
- `pave_connector.py`: extracted `send_record_to_pave(record)` so both producers share one writer and
  contract #7 stays defined in a single place. `send_to_pave(event)` now shapes the cascade record and
  delegates. Behaviour unchanged.
- `pothole_app_filtered.py`: `_detect_frame()` now returns the confidences of boxes that survive class
  and road-mask filtering (a fourth return value; both call sites updated), and the event record gains
  a `confidence` field — their mean.
- `app.js`: the poller now reads `detected_by` instead of hardcoding `'Both'`, and tolerates a `null`
  confidence.

**Why**
Fixes issue #19. The GUI already produced everything the map needed and wrote it to a file nothing
read.

Two things blocked a pure adapter, and both are worth recording because they shaped the design:

1. **`events.jsonl` had no confidence**, and contract #7 requires one. `potholes_detected` is a count,
   not a confidence — deriving one from it would have been inventing a number (Rule 6). The detector
   was already computing per-box confidences in `_detect_frame` and discarding them, so the fix was to
   stop discarding them. Framed as a defect in the GUI's own detection log, which it is: a log with a
   count but no measure of certainty is incomplete regardless of who consumes it. Not a change made to
   serve the integration layer (Rule 3).
2. **`app.js` ignored `detected_by`** and labelled everything `'Both'`. Publishing camera-only events
   through it would have had them claim to be sensor+vision on the dashboard — wrong, and invisibly so.

**Contracts affected** (Rule 4)
- **#9 (filtered-GUI record)** gains `confidence`. Optional — runs recorded before today have no such
  key and consumers must handle its absence.
- **#7 (`pave_events.json`)** unchanged in shape, but two consumer-side facts changed: `confidence` may
  now be `null`, and `detected_by` is now actually read. Both documented.
- Both mappings and the producer table are now in `20-data-contracts.md`.

**Context files updated**
`02-repository-map.md`, `03-architecture-dataflow.md`, `11-vision-pipeline.md`,
`14-integration-layer.md`, `20-data-contracts.md` (#7 and #9 both rewritten),
`30-setup-and-run.md`, `40-known-issues-and-gaps.md` (#19 to Resolved, priority renumbered), and this file.

**Verified by execution**
Adapter tested end to end against a fabricated `events.jsonl` containing a normal record, a legacy
record with no confidence, a record with no coordinates, and a truncated final line:
- Field mapping correct (`id`→`event_id`, `latitude`/`longitude`→`lat`/`lng`, `timestamp`→`created_at`).
- Every published record matches the contract #7 key set **exactly**.
- Legacy record published with `confidence: null` — not fabricated.
- Record without coordinates dropped, not placed at a guessed position.
- Truncated final line ignored — this happens for real during `--watch`, since the GUI appends
  line by line.
- Re-running published 0 and left the file at 2 records: **idempotent**.
- An event appended afterwards was picked up on the next pass.
- Both `run_dir` and a direct `events.jsonl` path resolve.
- A cascade record still writes correctly alongside GUI records through the shared writer.

`app.js`'s mapping checked in node across five event shapes (with/without confidence, with/without
`detected_by`, both missing): none throws, and the fallback labels read sensibly.

**Not verified**
The GUI itself was not launched — `cv2`, `ultralytics` and `torch` are not installed here — so the
`confidence` values now being logged are unexercised. The `_detect_frame` 4-tuple change is
parse-checked and both call sites updated, but not run. Worth confirming with one short video before
relying on it.

**Notes for the next agent**
- The `detected_by` vocabularies still disagree: the cascade writes `"hybrid (sensor+vision)"`, the
  adapter writes `"Image Model"` (the map's own `MODELS` wording). The dashboard displays whichever
  string it receives, so this is cosmetic — but align it if you touch either producer.
- `send_record_to_pave` is now the single writer for contract #7. New producers should go through it
  rather than writing the file directly.
- `pave_connector` is still a full read-modify-write per record and not concurrency-safe (issue #8).
  Running the adapter in `--watch` **while** the cascade is also publishing would race. One writer at a
  time.

---

## 2026-08-19 — Guarded download_road_model.py against clobbering the real model (fixes #12b)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `pothole_detection_app/scripts/download_road_model.py`

**What changed**
- The script now **refuses to overwrite an existing output file**, printing what is already there and
  why replacing it would be a downgrade, and exiting 1. Added `--output <path>` to write elsewhere and
  `--force` to overwrite deliberately; `--force` also writes a `.bak` copy first.
- **Moved the existence check before the download.** Previously it fetched the weights and then copied
  over the top. Now a refused run costs no network fetch and cannot half-finish.
- Moved `from ultralytics import YOLO` inside the function, below the check, so `--help` and the abort
  path work without ultralytics installed.
- **Corrected the docstring and console output.** It claimed the weights were "trained on
  Cityscapes/COCO and can segment roads". They are COCO-only and **COCO has no road class** — that
  false claim is precisely what made the script look safe to run. It now states plainly that this is a
  downgrade and points at the real training path.
- Replaced the non-ASCII status glyphs, which raise `UnicodeEncodeError` on a cp1252 Windows console
  when stdout is redirected.

**Why**
Fixes issue #12b, which this session created: adding the real `road_seg.pt` turned a previously
harmless script into a destructive one. It writes to `model/road_seg.pt` unconditionally, so a single
run would have replaced a 7-class visible-road model with a COCO model that cannot see roads at all —
silently, with detection quality degrading rather than failing.

**Contracts affected**
None.

**Context files updated**
`12-vision-training-scripts.md` (section rewritten), `30-setup-and-run.md` (command updated),
`40-known-issues-and-gaps.md` (#12b to Resolved, priority table renumbered), and this file.

**Verified by execution**
- Default run with `road_seg.pt` present: aborts, exit code 1, clear message.
- **Model SHA-256 byte-identical before and after** the abort
  (`b0330552c08c1945808e4f85f8ef5b3a0d75f6a25f0799b0198b493ab186c530`).
- `--help` works with ultralytics **not** installed.
- `--output <fresh path>` and `--force` both get past the guard and reach the ultralytics import,
  which fails here because the package is absent — that failure is the proof the check runs first.
- Directory listing after all cases: nothing created, nothing clobbered.
- **Not verified:** the actual download and copy path, since ultralytics is not installed. The backup
  and copy logic below the guard is unexecuted.

**Notes for the next agent**
- The guard protects `--output` too, not just the default path, so pointing it at another existing
  file is equally refused.
- If you genuinely need the COCO stand-in alongside the real model, `--output model/road_seg_coco.pt`
  is the intended route; nothing reads that filename automatically, so you would pass it explicitly to
  `create_two_stage_detector`.
- This whole issue is a reminder worth generalising: adding an asset can make an existing script
  dangerous without that script changing at all. Worth a thought whenever a placeholder becomes real.

---

## 2026-08-19 — Fixed the broken detector import and the training pipeline chain (#3, #4)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `pothole_detect_physics/` (1 script + README), `pothole_detection_app/quick_start.bat`

**What changed**

*#3 — `Model/run_detector_on_dataset.py`*
- Added a `sys.path.insert` for `detector_py/` before importing `PotholeDetector`, matching the
  pattern `integration/sensor_adapter.py` already uses. The script previously raised
  `ModuleNotFoundError` from any working directory.
- Corrected `pothole_detect_physics/README.md`, which listed the script under `detector_py/` and told
  you to run it from there. Folder tree, folder description and run command all updated.

*#4 — `quick_start.bat`*
- Inserted `merge_datasets.py` as step 2. The chain is now organize (dataset_v2), merge (dataset_v3),
  train (reads dataset_v3), evaluate, launch. Renumbered to 5 steps.
- `evaluate_model.py` is now called with `--data data\dataset_v3\data.yaml`. It defaults to dataset_v2, so the
  pipeline had been training on v3 and evaluating on v2 — the same mismatch one step later.
- Added explicit `if not exist ...data.yaml` checks after both dataset steps, plus a header comment
  pointing at the hardcoded `SOURCE_DIR`.

**Why**
Both were documented blockers: two workflows that fail outright as shipped.

The third-order find is the one worth remembering: **`organize_dataset.py` returns exit code 0 when
its source directory is missing or empty.** It prints an error and `return`s from `main()`, so
`if errorlevel 1` never fired and the batch marched on to training, which then failed with a
confusing "Data config not found" three steps from the actual cause. Adding `merge_datasets.py`
alone would not have fixed that; the file checks do.

**Contracts affected**
None.

**Context files updated**
`02-repository-map.md`, `10-physics-sensor-pipeline.md`, `12-vision-training-scripts.md`,
`30-setup-and-run.md`, `40-known-issues-and-gaps.md` (#3 and #4 to Resolved, priority table
renumbered, the physics half of stale-README issue #15 struck through), and this file.

**Verified**
- **#3 executed.** Replicated the script's path logic and confirmed `from pothole_detection import
  PotholeDetector` resolves and `PotholeDetector()` instantiates (state `IDLE`, `drop_threshold`
  6.81). Also confirmed the bare import *still fails* from `Model/`, proving the insert is
  load-bearing rather than incidental.
- **#4 structurally validated, not executed.** Checked `if (` block balance (depth returns to 0), no
  bare parens inside block echoes, no unescaped redirects in echo lines, and that the printed chain
  reads organize, merge, train, evaluate, launch. Cross-checked against the scripts themselves:
  `organize_dataset.TARGET_DIR` is dataset_v2, `merge_datasets.OUTPUT_DIR` is dataset_v3,
  `train_model.DATA_DIR` is dataset_v3.
- **Not executed:** running `quick_start.bat` end to end would start a 1-3 hour training run
  (Rule 8), and `run_detector_on_dataset.py` processes all 80,000 rows and opens a plot window.
  Neither was run in full.

**Notes for the next agent**
- `evaluate_model.py` still *defaults* to dataset_v2. Only the batch file passes `--data`. If you
  invoke it by hand after training, pass it yourself or you will evaluate on the wrong split.
- Running the scripts by hand still requires `merge_datasets.py` between organize and train — only
  the batch file was fixed, not the scripts' own defaults.
- The `Model/` vs `detector_py/` placement of `run_detector_on_dataset.py` was left alone
  deliberately (Rule 11). Moving it into `detector_py/` remains a defensible tidy-up; the README now
  documents where it actually is either way.

---

## 2026-08-19 — Fixed the `_conf` import trap (fixes #5)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `pothole_detection_app/app/` — three files

**What changed**
- `utils.py`: added `get_conf_threshold()`, the read-side counterpart to `set_conf_threshold()`. Its
  docstring explains why importing `_conf` is wrong, so nobody reverts it.
- `main_enhanced.py`: import no longer pulls `_conf`; **4** call sites now use `get_conf_threshold()`.
- `enhanced_utils.py`: import no longer pulls `_conf` **or** `_model`; **2** call sites use the accessor.
- Both import blocks carry a short comment naming the trap.

**Why**
Fixes issue #5. `from utils import _conf` binds the float **by value at import time**.
`set_conf_threshold()` rebinds `utils._conf`, which the importing module never sees — so the enhanced
GUI's confidence slider updated its label and the utils global, while `_detect_image`,
`_detect_video_worker` and `batch_process_images` all kept using 0.35 forever.

`_model` was the same trap: imported by value (as `None`), never used. Removed so nobody reaches for it
later and hits the identical bug. `load_model()` was already being called correctly.

**Contracts affected**
None. Internal to the vision subsystem; no cross-subsystem interface changed.

**Context files updated**
`11-vision-pipeline.md` (the warning replaced with the correct usage guidance, in both the utils section
and the troubleshooting table), `21-configuration-and-tuning.md` (confidence section and the constants
table), `40-known-issues-and-gaps.md` (#5 to Resolved, priority table renumbered), and this file.

**Verified**
- All three files parse.
- **Behaviourally verified against the real `utils.py`** with the heavy deps stubbed: drove
  `set_conf_threshold()` through 0.35 / 0.10 / 0.60 / 0.90 while reading through both patterns from a
  separate module. The old `from utils import _conf` stayed pinned at **0.35** the whole time; the
  accessor returned **0.10 / 0.60 / 0.90** correctly. That is the bug and the fix demonstrated side by
  side.
- AST check confirms neither shipped consumer imports `_conf` or `_model` any more, and both import
  `get_conf_threshold`.
- **Not verified:** the GUI was never launched. `cv2`, `ultralytics` and `torch` are not installed here,
  so the end-to-end "move the slider, see detection change" path is unconfirmed. Someone with the vision
  deps should run `python app/main_enhanced.py`, set confidence to 0.90, and confirm detections drop.

**Notes for the next agent**
- `utils.run_detection()` uses the bare `_conf` and that is **correct** — it lives in the same module and
  reads the live global. Only cross-module imports were broken.
- `pothole_app_filtered.py` was never affected; it owns a real `DoubleVar` and reads it live.
- The general rule now stated in the code: **never import a module global that another module rebinds.**
  Call an accessor. That applies to `_conf`, `_model`, and anything added later with the same shape.

---

## 2026-08-19 — scikit-learn pinned to the version that pickled the model (fixes #26)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `pothole_detect_physics/requirements.txt`

**What changed**
- Pinned `scikit-learn==1.7.2`, with a comment explaining why and warning that retraining requires
  updating the pin. `pandas`, `numpy`, `joblib` and `matplotlib` left unpinned — the ask was the
  sklearn pin, and pinning the rest has a wider blast radius.
- Downgraded the local environment from 1.9.0 to 1.7.2 to verify the pin actually resolves.

**Why**
Fixes issue #26. The committed `pothole_ai_model.pkl` records `_sklearn_version = 1.7.2`; loading it
under any other version raises `InconsistentVersionWarning`, which scikit-learn documents as possibly
producing invalid results. Every measurement in this repo, including the Stage-1 accuracy numbers,
was being produced through that mismatch.

**Contracts affected**
None.

**Context files updated**
`10-physics-sensor-pipeline.md` (pin noted beside the model output),
`30-setup-and-run.md` (verified version set updated), `40-known-issues-and-gaps.md` (#26 to Resolved,
#27 added, priority table renumbered), and this file.

**Verified**
- Version read **directly from the pickle bytes** — `_sklearn_version` at offset 408 is the string
  `1.7.2`. Not inferred from the warning text; `getattr(model, '_sklearn_version')` does not work
  because scikit-learn consumes it during `__setstate__`.
- `scikit-learn==1.7.2` installs on Python 3.12 with **no numpy downgrade** — it requires only
  `numpy>=1.22.0`, satisfied by the installed 2.5.1.
- Loaded the model with `InconsistentVersionWarning` escalated to an exception via
  `warnings.simplefilter("error", ...)`. It does **not** raise. `predict_proba` on a benign row still
  returns sane output (0.9976 / 0.0024).
- Full suite re-run under the pin: **7 passed, 1 skipped**, event recall 1.00, precision 1.00 —
  **identical to the results under 1.9.0**. So the mismatch had not been skewing anything; it is
  simply no longer a risk. Runtime was 122 s this run versus 62 s previously; not investigated, and
  more likely machine load than a version effect.

**Notes for the next agent**
- **If you retrain the model, update the pin.** `train_ai_model.py` writes whatever version you train
  with into the new `.pkl`, and a stale pin puts the warning straight back.
- **New issue #27:** unpickling emits ~1600 `DeprecationWarning`s from `joblib/numpy_pickle.py` —
  *"Setting the shape on a NumPy array has been deprecated in NumPy 2.5"*. One per array in the
  forest. Harmless today, but the load will break outright when numpy removes the API. Distinct from
  #26: that was scikit-learn's version check, this is joblib's pickle reader. Not fixed — the options
  are upgrading joblib once a release handles numpy 2.5, or pinning `numpy<2.5`.
- The vision subsystem does not use scikit-learn, so `pothole_detection_app/requirements.txt` is
  correctly untouched.

---

## 2026-08-19 — Stage-1 accuracy test rescored per event (fixes #25)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `integration/test_integration.py`

**What changed**
- `test_stage1_accuracy_against_real_labels` now scores **per event** instead of per sample.
  Contiguous `label==1` runs are grouped by a new module-level `group_label_events()` helper; a
  pothole counts as detected if the cascade triggered anywhere inside its block or within
  `EVENT_MATCH_TOLERANCE_S` (0.02 s) after it. Precision is the fraction of triggers landing in any
  such window.
- The tolerance is converted from seconds to samples using the **dataset's own timestep**, so it
  survives a change of sampling rate — CARLA runs may not be 400 Hz.
- Per-sample recall is still printed, explicitly labelled *"NOT a quality signal"* and annotated with
  its own ceiling, so baselines recorded before this change remain comparable.
- Added `test_label_event_grouping` — covers the grouping logic with no model load, so a change that
  breaks the metric fails fast rather than silently skewing the accuracy test.
- Added a **vacuity guard**: the test now asserts the analysed slice contains at least one labelled
  event. Previously, a `.head(N)` window with no potholes would have passed while reporting nothing.
- Updated the module docstring to describe event-level scoring.

**Why**
Fixes issue #25, raised in the previous entry. The old per-sample scoring compared a once-per-event
detector against a 12-sample label window, capping recall at 1/12 = 0.083 regardless of detector
quality — so a cascade catching every pothole reported `recall=0.08` and read as broken.

**Contracts affected**
None. Test-only change; no production code touched.

**Context files updated**
`00-RULES.md` (Rule 6 rewritten around the corrected figures), `14-integration-layer.md` (test section
rewritten with the measured table), `21-configuration-and-tuning.md` (tuning workflow now points at
event-level), `40-known-issues-and-gaps.md` (#25 moved to Resolved; accuracy-claims table updated),
and this file.

**Verified**
- Full suite re-run: **7 passed, 1 skipped, 61.8 s** (up from 6 passed — the new grouping test).
- Reported output: `events 3, detected 3, triggers 3, recall 1.00, precision 1.00`, with the
  per-sample line showing `0.08 (ceiling 0.083)`.
- **Metric discrimination verified separately** against seven synthetic scenarios: perfect detection,
  one missed event (0.67), all missed (0.00), a spurious trigger on clean road (precision 0.75), all
  triggers spurious (0.00), a trigger just past the block within tolerance (1.00), and one 9 samples
  past the block, outside the 8-sample tolerance (0.00). All scored as expected — the metric can
  register a regression rather than passing vacuously.
- `test_label_event_grouping` covers empty input, full-length runs, multiple runs, adjacent
  single-sample runs, a run touching the end of the array, and float labels as pandas supplies them.

**Notes for the next agent**
- **Quote event-level recall/precision.** The per-sample line exists only so older baselines stay
  comparable; the output says so, and Rule 6 now says so.
- The test still deliberately asserts no performance target — only sanity floors and the vacuity
  guard. Its purpose is a baseline to track while tuning `SENSOR_THRESHOLD`, not a gate. Current
  baseline: 3 events, recall 1.00, precision 1.00 on the first 2000 rows.
- Issue #26 is still open and still applies to these numbers: the `.pkl` was pickled under
  scikit-learn 1.7.2 and unpickles under 1.9.0 with `InconsistentVersionWarning`.

---

## 2026-08-19 — Test suite run; Stage-1 recall metric found to be misleading
**Author:** Claude (Opus 5), via Claude Code
**Scope:** documentation corrections — **no code changed.** Test dependencies installed.

**What changed**
- Installed `pandas 3.0.0`, `numpy 2.5.1`, `scikit-learn 1.9.0`, `joblib 1.5.3`, `pytest 9.1.1`.
- Ran `pytest integration/test_integration.py -v -s`: **6 passed, 1 skipped** (the skip is
  `test_frame_provider_is_deterministic`, which skips cleanly because `data/sample_images/` is absent
  — issue #2). The P3 changes did not break anything.
- **Corrected a claim I made in the 2026-08-19 knowledge-base entry**, which called the Stage-1
  recall figure the only trustworthy number in the repo. The precision is trustworthy; **the recall
  is structurally misleading.** Recorded as new issue #25.
- Added issue #26: the committed `.pkl` was pickled by scikit-learn 1.7.2 and now unpickles under
  1.9.0 with `InconsistentVersionWarning`.

**The finding**
The test prints `precision=1.00 recall=0.08`, which reads as a broken detector. It is not.

`sensor_triggered` fires on **one** sample (the impact). `label` marks **12** samples per pothole.
So sample-level recall cannot exceed 1/12 = 0.083 however good the detector is — and 0.08 is
essentially that ceiling.

Measured by hand on the first 2000 rows (3 pothole events, 36 labelled samples):

| Metric | Value |
|---|---|
| Sample-level recall (printed by the test) | 0.08 |
| Theoretical ceiling for a once-per-event detector | 0.083 |
| **Event-level recall** | **3/3 = 1.00** |
| Precision | 1.00 (TP=3, FP=0) |

The FSM fires on the 11th of each event's 12 labelled samples — exactly where `generate_dataset.py`
injects the impact block. **The cascade caught every pothole.**

**Contracts affected**
None.

**Context files updated**
`00-RULES.md` (Rule 6 corrected; Rule 2's stale "road_seg.pt does not ship" claim fixed),
`14-integration-layer.md` (test description corrected with the measured table),
`21-configuration-and-tuning.md` (warning when using it as a tuning signal),
`30-setup-and-run.md` (verified dependency versions, runtime),
`40-known-issues-and-gaps.md` (issues #25 and #26; accuracy-claims table rewritten), and this file.

**Verified**
- Full suite: 6 passed, 1 skipped, 65 s.
- The event-level analysis was computed directly from the dataset and the live `SensorSession`, not
  inferred — contiguous `label==1` runs grouped, trigger indices matched against each run.
- Confirmed all three events are 12 samples wide and every trigger lands inside its event.

**Notes for the next agent**
- **Do not quote the printed recall without explaining it.** Quote event-level recall, or say what the
  0.08 means. Rule 6 has been updated accordingly.
- Issue #25 has a concrete fix (group contiguous label runs, score per event). I did **not** apply it —
  changing test semantics is a decision for the user, and the ask was to run the suite, not change it.
- Issue #26 is quieter but nastier: every number above was produced with a version-mismatched pickle.
  They look self-consistent, but the model has not been validated against the sklearn that loads it.
  Pin the version or retrain.
- The suite takes ~65 s, dominated by the 2000-row Stage-1 test.

---

## 2026-08-19 — P3: real frame provider + orchestrator wiring
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `integration/` — one new module, two modified

**What changed**
- **New** `integration/carla_frame_provider.py` — `CarlaFrameProvider(run_dir)`. Timestamp-keyed
  lookup of camera frames and GNSS fixes over a recorded CARLA run. Nearest-frame selection with a
  tolerance, linearly interpolated GPS, miss counters, a `coverage()` diagnostic, and a
  `__main__` self-check that needs no CARLA.
- **Modified** `integration/frame_provider.py` — appended `MockFrameProvider`, an object wrapper
  around the existing two functions so the mock and the real provider are interchangeable. The
  functions themselves are **untouched**; `test_integration.py` calls them directly.
- **Modified** `integration/orchestrator.py` — `run()` gains `dataset_path` and `frame_provider`
  parameters (both defaulting to current behaviour), plus an argparse CLI with `--limit`,
  `--dataset` and `--carla-run`. A sensor-triggered row whose provider returns no frame is now
  skipped and counted rather than paired with an unrelated image.

**Why**
P3 of the CARLA plan. This is the piece that actually closes issue #18 — the mock keys frames on
`event_id`, which has no relationship to anything; the real provider keys on **time**, and a shared
timebase is the only thing that makes a frame and a sensor row belong together.

**Contracts affected**
None changed. `run()`'s new parameters are additive with defaults preserving existing behaviour.
`PotholeCandidateEvent` and `pave_events.json` are untouched.

**Context files updated**
`02-repository-map.md`, `14-integration-layer.md` (new provider section + updated `run()` signature),
`15-carla-testbed-plan.md` (P3 marked written), `30-setup-and-run.md` (CLI commands),
`40-known-issues-and-gaps.md` (#9 mitigated, #10 superseded on the CARLA path, #18 progress), and this file.

**Verified**
- All three modules parse.
- **`CarlaFrameProvider` tested end-to-end against a fabricated run directory** — no CARLA needed.
  Covered: out-of-order `frame_index.csv` sorted correctly (mean gap exactly 0.05 s); a frame listed
  but missing on disk dropped with a warning; nearest-frame selection at six positions including
  round-down, round-up, an exact tie resolving to the earlier frame, and past-the-last-frame within
  tolerance; out-of-range lookups returning `None` and incrementing `misses_no_frame` rather than
  fabricating; GPS interpolation exact to 1e-9 against a known linear track; GPS out-of-range
  returning `None`; interface parity with `MockFrameProvider`.
- Confirmed `test_integration.py` imports only `get_mock_frame` and `simulate_gps` from
  `frame_provider` and does not import `orchestrator` at all, so it is structurally unaffected.
- **Not verified:** the existing pytest suite was NOT run — `pandas` and `pytest` are not installed
  in this environment. The orchestrator changes are unexecuted; only parsing and the provider's own
  logic were tested. Nothing has been run against CARLA.

**Notes for the next agent**
- `DEFAULT_FRAME_TOLERANCE_S = 0.06` is deliberately tight. The camera runs at 20 Hz so worst-case
  nearest-frame error is 25 ms. **Do not widen it to reduce skips** — pairing a sensor event with a
  distant frame silently reintroduces the desynchronisation the class exists to remove. If skips are
  high, raise the camera rate instead.
- `get_frame()` returning `None` is a real answer ("the camera was not looking then"), not an error.
  The orchestrator skips those rows and reports the count.
- Run `python integration/carla_frame_provider.py <run_dir>` after any recording — it reports the
  max frame gap and warns when rows will fall through.
- Someone with `pandas` and `pytest` installed should run
  `pytest integration/test_integration.py -v -s` to confirm the suite still passes.

---

## 2026-08-19 — CARLA Level A scaffold written
**Author:** Claude (Opus 5), via Claude Code
**Scope:** new `carla_sim/` subsystem + context updates

**What changed**
- Created `carla_sim/` — the Level A testbed: potholes as data, jerk seeded with a physics impulse,
  no CARLA source build required.
  - `config.py` — every tunable. `WHEEL_POSITION_SCALE` and `IMU_GRAVITY_AT_REST` deliberately start
    as `None`; the recorder refuses to run until P0 measures them.
  - `verify_setup.py` — P0 as executable code: IMU gravity convention, wheel position units, impulse
    API shape, achievable tick rate.
  - `scenario/potholes.py` — registry, route placement, wheel-over detection, ground-truth labels.
  - `scenario/impulse.py` — the jerk model.
  - `scenario/route.py` — deterministic route + minimal waypoint follower.
  - `scenario/sensors.py` — sensor rig, per-sensor rates, sync-mode queues.
  - `scenario/drive_and_record.py` — main loop, writes contract #1.
  - `README.md`, `requirements.txt`, `.gitignore`.

**Why**
The user asked for the Level A scaffold after reviewing the plan in
[`15-carla-testbed-plan.md`](15-carla-testbed-plan.md). Level A is the on-ramp that needs no source
build; every piece written here carries forward to Level B unchanged.

**Contracts affected**
None changed. `sensors.csv` is written to be **column-identical** to contract #1
(`timestamp,ax,ay,az,gx,gy,gz,speed,label`) so the FSM, the RandomForest, fusion and the connector
consume it with no modification. That property is the whole point — do not "improve" the schema.

**Context files updated**
`15-carla-testbed-plan.md` (status + scaffold table), `02-repository-map.md` (new subsystem section),
`30-setup-and-run.md` (commands), `README.md` (index + routing), and this file.

**Verified**
- All 8 modules parse (`ast.parse`).
- **Pure-geometry logic unit-tested against a stubbed `carla` module** — `Pothole.contains` boundary
  behaviour at the radius; `PotholeTracker` firing exactly one entry event per crossing rather than
  one per tick; re-entry after leaving firing again; two potholes and four wheels resolving without
  double-fire; label window rising on entry and expiring at +50 ms; `write_ground_truth` round-trip.
- **Not verified — nothing has been run against a simulator.** CARLA is not installed on this
  machine. Every runtime claim (sensor attribute names, the impulse API, sync-mode behaviour,
  `save_to_disk`, blueprint ids) comes from the documented 0.9.15 API and is unconfirmed.

**Notes for the next agent**
- **Two API details were deliberately not asserted.** `WheelPhysicsControl.position` units are
  measured by `verify_setup.py` rather than assumed, and `impulse.py` detects whether
  `add_impulse_at_location` exists and otherwise falls back to `add_impulse` plus a manually
  computed angular impulse `r x J`. Both paths are physically equivalent.
- `IMPULSE_DELTA_V = 0.8` is a **starting guess**, not a tuned value. The right value drives `az`
  below the FSM's 6.81 drop threshold without launching the car.
- The design deliberately seeds **only the drop**; the suspension unloading produces the freefall
  reading and the spring re-compressing produces the impact spike. Scripting all three phases would
  reduce this to replaying `generate_dataset.py` and would prove nothing. Do not "simplify" it.
- Both entry points restore world settings in a `finally` block. Leaving the CARLA server in
  synchronous mode makes it appear frozen to every other client — keep that.
- Next: run P0, tune the impulse, then P3 — `integration/carla_frame_provider.py` plus a
  `dataset_path` parameter on `orchestrator.run()`.

---

## 2026-08-19 — CARLA simulation testbed plan recorded
**Author:** Claude (Opus 5), via Claude Code
**Scope:** documentation only — no code written, no `carla_sim/` created

**What changed**
- Added `context/15-carla-testbed-plan.md` — the full design for testing the pipeline inside CARLA:
  three fidelity levels for creating potholes with real collision depth, an 8-phase build plan, the
  proposed `carla_sim/` layout, P0 assumptions to verify, risks, and open decisions.
- Indexed it in `README.md` (subsystem table + routing table).
- Linked it from issue #18 in `40-known-issues-and-gaps.md` as the concrete plan to close that gap.

**Why**
The user asked whether the whole system could be tested in CARLA with real depth-carrying potholes, synced
camera + IMU detection, and the CARLA road network rendered in the map UI, then asked for the plan to be
kept in context. Also published as an artifact:
https://claude.ai/code/artifact/731a81d3-76b7-488a-baf7-d70a620cd5d4

**Contracts affected**
None. The plan is explicitly built around keeping contract #1 (sensor CSV columns) and contract #7
(`pave_events.json`) unchanged — that is what makes it cheap.

**Context files updated**
`15-carla-testbed-plan.md` (new), `README.md`, `40-known-issues-and-gaps.md`, and this file.

**Verified**
- Confirmed by reading source: `scripts/prepare_multiclass_seg_dataset.py` already consumes CARLA
  `rgb/` + `semantic/` renders, with `CARLA_ROAD=7`, `CARLA_PEDESTRIAN=4`, `CARLA_VEHICLE=10`, and derives
  its `shadow` class heuristically (dark + low-saturation pixels on road) since CARLA has no shadow tag.
- Confirmed the CARLA IMU channel set maps one-to-one onto contract #1's columns.
- **Not verified:** every claim about CARLA runtime behaviour comes from the documented API. CARLA is not
  installed here and nothing was run. The IMU gravity convention in particular is an assumption, and P0
  exists to test it.

**Notes for the next agent**
- **Status line in `15-*.md` says PROPOSED.** Update it the moment code lands (Rule 5).
- The highest-risk unknown is whether CARLA's IMU includes gravity. `PotholeDetector` assumes
  `az ~= 9.81` at rest; if CARLA differs, nothing ever triggers and it will present as a logic bug.
- Level A (scripted impulse) needs no CARLA source build and is the recommended on-ramp. Its key design
  point: script only the drop and let the suspension generate freefall and impact naturally, and use
  `add_impulse_at_location` at the wheel rather than `add_impulse` at the centre of mass, so the gyro
  channels carry real signal instead of staying flat.
- No decisions have been made yet — the four open questions at the end of `15-*.md` are still open.

---

## 2026-08-19 — Added the real road segmentation model (`road_seg.pt`)
**Author:** Claude (Opus 5), via Claude Code
**Scope:** `pothole_detection_app/` (model + .gitignore), plus context updates across 11 files

**What changed**
- Copied `model/road_seg.pt` (20,507,364 bytes) into `pothole_detection_app/model/` from the upstream
  repository at `D:/dev/PAVE/pothole-detection-app/pothole-detection-app/model/road_seg.pt`.
- Fixed `pothole_detection_app/.gitignore`: added `!model/road_seg.pt` after the `model/*` rule (line 24).
  The file already had `!model/road_seg.pt` at line 20, but `model/*` on line 24 re-ignored it and only
  `best.pt` was re-exempted afterwards. Git was silently refusing to track the model.

**Why**
The user supplied the original repository and asked for the road segmentation model to be taken from it.
Resolves issue #1 — the largest blocker in `40-known-issues-and-gaps.md`. Two-stage detection (the vision
subsystem's headline false-positive filter) was inert without it.

**What the model actually is**
Read directly from the checkpoint's pickled config, since ultralytics is not installed here:
- `yolo11s-seg`, task `segment`, ultralytics 8.4.22, saved 2026-03-18
- **7 classes — the `extended` preset:** `visible_road, vehicle, pedestrian, shadow, vegetation,
  roadside_object, road_obstacle`
- Trained on `data/visible_road_seg_public_full/data.yaml`, 60 epochs, patience 30, batch 10, imgsz 640
- **Not** the generic COCO model `scripts/download_road_model.py` fetches — this is the real thing.

**Contracts affected**
Contract #10 (YOLO label formats) extended with the shipped model's 7-class ordering. Both road-mask
implementations resolve classes **by substring match on the name**, not by index, so this ordering plus
the exact class names are now a live contract. Documented in `20-data-contracts.md`.

**Context files updated**
`CLAUDE.md`, `01-project-overview.md`, `02-repository-map.md`, `03-architecture-dataflow.md`,
`11-vision-pipeline.md`, `12-vision-training-scripts.md`, `14-integration-layer.md`,
`20-data-contracts.md`, `21-configuration-and-tuning.md`, `30-setup-and-run.md`,
`40-known-issues-and-gaps.md`, and this file.

**Verified**
- SHA-256 of source and destination match: `b0330552c08c1945808e4f85f8ef5b3a0d75f6a25f0799b0198b493ab186c530`.
- Provenance confirmed: the upstream repo's `best.pt` is byte-identical to ours
  (`8f999cc971cea221984f030b3dc62dbf07a1c8ca4273e78e07019b04291b23e7`), so it is the same source project.
- Architecture, class names and training args extracted from `best/data.pkl` inside the checkpoint zip.
- `git status` now lists the file as untracked rather than hiding it, confirming the .gitignore fix works.
- **Not verified:** the model was never loaded or run. `ultralytics`/`torch` are not installed in this
  environment and there is no venv. Nothing was inferred with it. Someone with the deps should run
  `python scripts/test_road_segmentation.py` to confirm mask quality.

**Notes for the next agent**
- **New issue #12b:** `scripts/download_road_model.py` writes to `model/road_seg.pt` and would now
  **overwrite this model** with a COCO one that has no `road` class. Do not run it until it is guarded.
- `TwoStageDetector`'s class resolution is correct here **only because of pass ordering**:
  `roadside_object` and `road_obstacle` both match the include substring `road`, get added to the mask,
  then get subtracted by the exclude pass. Reordering those two passes would silently admit roadside
  clutter as drivable road. Traced by hand in `11-vision-pipeline.md`; worth an actual test.
- **Correction to the previous entry:** it recorded `data/visible_road_seg_public_full/` (referenced by
  `test_detection.py`) as matching no directory any script produces. That was wrong — it is the real
  training-dataset directory for these weights, per the `data.yaml` path inside the checkpoint. Issue #7
  has been corrected in place.
- Resolved issues now move to a **Resolved** section in `40-known-issues-and-gaps.md` keeping their
  original numbers. Numbers are never reused or renumbered, so existing references stay valid.
- The vision half of the pipeline is now genuinely testable end to end — but issue #2 (missing
  `data/sample_images/`) still blocks the orchestrator, and issue #18 (no frame↔sensor sync) still means
  no fused accuracy number is meaningful.

---

## 2026-08-19 — Context knowledge base created
**Author:** Claude (Opus 5), via Claude Code
**Scope:** documentation only — no code, config, or data was modified

**What changed**
- Added `CLAUDE.md` at the repo root as the AI agent entry point, with the mandatory boot sequence and an immediate-gotchas section.
- Added `AGENTS.md` at the repo root — a short pointer to the same rules, for non-Claude tools that look for that filename.
- Created the `context/` knowledge base, 15 files:
  - `00-RULES.md` — twelve binding rules (boot sequence, subsystem boundaries, contract discipline, the mandatory doc-update rule, permission gates for heavy operations, style, reporting).
  - `README.md` — index, reading order, and a task-to-file routing table.
  - `01-project-overview.md` — what PAVE is, the four subsystems, a maturity table separating real from staged, and a glossary.
  - `02-repository-map.md` — every file and directory, annotated as committed / gitignored / referenced-but-missing.
  - `03-architecture-dataflow.md` — component diagram, the step-by-step cascade, the three transports, and the score at each hop.
  - `10-physics-sensor-pipeline.md` — the FSM, the depth and length derivations, the synthetic generator, the RandomForest and its limits.
  - `11-vision-pipeline.md` — `TwoStageDetector`, the road-mask algorithms, both GUIs, the shared utils.
  - `12-vision-training-scripts.md` — training presets, dataset preparation, evaluation, inference CLIs.
  - `13-map-ui.md` — maps, GPS, Haversine proximity, event polling, the `PotholeGuard` API, theme tokens.
  - `14-integration-layer.md` — design philosophy, each adapter, fusion mathematics, the connector, the test suite.
  - `20-data-contracts.md` — eleven boundary-crossing contracts specified in full.
  - `21-configuration-and-tuning.md` — every threshold in one table, with the effect of moving each.
  - `30-setup-and-run.md` — environments, first-run checklist, every command with its working-directory requirement, troubleshooting.
  - `40-known-issues-and-gaps.md` — 24 findings, categorised and severity-ranked, with a suggested fix order.
  - `50-changelog.md` — this file.

**Why**
The user asked for context files covering every element and mechanic of the project, so any AI assistant can understand it quickly, plus a rules system that agents must read first and that requires them to keep the context files current after making changes.

**Contracts affected**
None — documentation only.

**Context files updated**
All of them (created).

**Verified**
- Read every source file in the repo in full, except the two large Tkinter GUIs (`main_enhanced.py`, `pothole_app_filtered.py`), which were read structurally plus in full across their detection, threading, road-mask and output paths.
- Confirmed by direct inspection: `model/road_seg.pt` absent while `model/best.pt` is present (40.5 MB); no `__init__.py` anywhere in the repo; `data/sample_images/` absent; the CSV has 80,000 data rows with the documented nine columns.
- Every issue in `40-known-issues-and-gaps.md` was traced to specific lines. Fixes were **not** applied — the task was documentation.
- **Not verified by execution:** nothing was run. No tests, no orchestrator pass, no training, no GUI launch. All behavioural descriptions are from source reading, not observed runtime.

**Notes for the next agent**
- The highest-value structural gap is #18 (no frame↔sensor synchronisation). Until it is closed, **no vision-stage or end-to-end accuracy claim is meaningful**. The only trustworthy number in the repo is the Stage-1 precision/recall printed by `pytest integration/test_integration.py -v -s`.
- Two "stage" vocabularies collide: integration Stage 1/2 means *sensor/vision*; vision two-stage means *road segmentation/pothole detection*. Always say which.
- The fusion weights give the camera an absolute veto — with `vision_score = 0` the maximum `combined` is 0.4, below the 0.5 cut. Setting `SENSOR_WEIGHT >= FUSION_THRESHOLD` would destroy that property. Worth knowing before anyone tunes them.
- The three existing READMEs are stale in specific documented ways (`40-known-issues-and-gaps.md` §15). They were left untouched — rewriting them was not part of this task, and it is a reasonable follow-up.
- Nothing in `40-known-issues-and-gaps.md` was fixed. All 24 items are open.
