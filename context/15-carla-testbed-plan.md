# 15 — CARLA Simulation Testbed (LEVEL A — P0 COMPLETE 2026-09-06)

> **Status change 2026-09-06.** CARLA 0.9.16 is installed at `D:\dev\pothole` and running. `verify_setup.py` has been executed against it and both P0 values are now filled in (`IMU_GRAVITY_AT_REST = 9.3483`, `WHEEL_POSITION_SCALE = 0.01`). `scenario/export_map.py` has been run against the live simulator and produced a valid 200-segment GeoJSON. **Still unrun:** `drive_and_record.py` — no run has been recorded, so nothing downstream of it is verified.

**Scope:** the design for testing the whole PAVE pipeline inside the CARLA simulator — potholes with real
depth, camera + IMU detection, and the CARLA road network rendered in the map UI.
**Status:** 🟢 **LEVEL B BUILD PHASE COMPLETE AND VERIFIED (2026-09-08).** CARLA 0.9.16 builds from
source against a source-built UE 4.26, and **the CarlaUE4 editor opens and renders `Town10HD_Opt`** (4,013
actors). Engine + Program targets, LibCarla, the CarlaUE4 plugin (752/752 actions) and the PythonAPI wheel
all built. What remains for Level B is the **content** work (P1: Blender tiles → FBX → `make import`),
for which Blender is **already installed** (5.2.0 LTS and 4.4.1 on D:). Level A runs. Level C unstarted.
**Rule 6 applies:** do not describe any of this as working, measured, or validated until it is.
**Drafted:** 2026-08-19 · **Published plan:** https://claude.ai/code/artifact/731a81d3-76b7-488a-baf7-d70a620cd5d4

---

## Why this is worth doing

It closes [issue #18](40-known-issues-and-gaps.md) — no frame↔sensor synchronisation — which is the single
structural gap blocking every accuracy claim in the project. It also retires #2, #12 and #10 as side effects.

**The architecture already fits.** CARLA's sensors produce contract #1 natively:

| Contract #1 column | CARLA source |
|---|---|
| `timestamp` | `world.tick()` snapshot / sensor timestamp |
| `ax, ay, az` | `sensor.other.imu` → `accelerometer` |
| `gx, gy, gz` | `sensor.other.imu` → `gyroscope` |
| `speed` | `vehicle.get_velocity()` magnitude |
| `label` | computed from known pothole positions — **real ground truth for the first time** |

Because the columns match, **the physics FSM, the RandomForest, fusion, the connector and the map UI all run
unchanged.** The work is a recorder, not a new pipeline.

**Prior art already in this repo:** `scripts/prepare_multiclass_seg_dataset.py` consumes `<input>/rgb/` +
`<input>/semantic/` CARLA renders, and its constants (`CARLA_ROAD=7`, `CARLA_PEDESTRIAN=4`,
`CARLA_VEHICLE=10`) match CARLA 0.9.13+ exactly. Someone planned this before.

---

## The core problem: a pothole is negative space

Unreal collides with **surfaces**. A decal or texture fools the camera but the IMU feels nothing — and the
whole point is to feel it.

CARLA vehicles use Unreal's PhysX wheeled-vehicle model: each wheel is a **raycast** against collision
geometry driving a spring-damper suspension. Give the raycast a surface that drops away and the suspension
extends; give it a far edge and it slams. That is the `DROP → FREEFALL → IMPACT` signature the FSM hunts for.

### Three levels

| Level | Approach | Source build? | Fidelity |
|---|---|---|---|
| **A** | Scripted impulse + data-defined pothole locations | ❌ no | Drop is scripted; freefall and impact are physics-derived |
| **B** | Blender pothole tiles → CARLA props, spawned via Python API | ✅ yes | Fully geometry-derived |
| **C** | Holes cut into the road mesh of a custom packaged map | ✅ yes | Highest; no lip artifacts |

**Recommended progression: A to prove the plumbing, then B.** They are stages, not alternatives — Level A's
recorder, exporters, integration adapter and UI work all carry forward to B unchanged.

---

## LEVEL A — detail

Runs on a **stock packaged CARLA release** (`pip install carla`). No Unreal, no Blender, no source build.

### The insight that makes it work

Do **not** script all three phases. Script only the drop, and let the suspension produce the rest:

1. Apply a downward impulse at the wheel that is entering the pothole.
2. The body accelerates down; the suspension unloads. With the wheel unloaded the accelerometer's contact
   force falls toward zero — **that is a genuine freefall reading, not a scripted one.**
3. The spring re-compresses as the body falls back. **That is a genuine impact spike.**

So even in Level A, two of the three FSM phases come out of the physics engine rather than a script.

### Use `add_impulse_at_location`, not `add_impulse`

`add_impulse()` applies at the centre of mass and produces **no rotation**. The IMU also records `gx, gy, gz`,
so a COM impulse leaves the gyro channels flat — and the RandomForest would happily learn "flat gyro" as the
tell, producing a model that collapses the moment it sees a real one-sided wheel strike.

`add_impulse_at_location(impulse, wheel_world_location)` produces the roll and pitch a real one-sided hit
produces, so all six IMU channels carry signal.

### Pieces to build

| # | Piece | Notes |
|---|---|---|
| 1 | Pothole registry | List of road-surface `carla.Location`s + radius + depth/severity. Pick from `map.generate_waypoints()` so they land on real lanes. |
| 2 | Wheel-over test | Per tick, distance from each wheel to each pothole centre. `vehicle.get_physics_control().wheels[i].position` — **verify units, CARLA has returned centimetres here.** Vehicle-centre proximity is an acceptable first cut. |
| 3 | Impulse profile | Downward impulse at the wheel, magnitude scaled by severity and speed, over 1–3 ticks. Tune against the FSM's thresholds. |
| 4 | Recorder | Sync mode; IMU every tick, camera at 20 Hz via `sensor_tick`. Writes `sensors.csv`, `frames/`, `frame_index.csv`, `gnss.csv`, `ground_truth.json`. |
| 5 | Vision input | See below — the honest limitation of Level A. |

### The vision limitation

Level A has no 3D pothole for the camera to see. Options, least to most honest:

- **Composite a pothole texture into the RGB frame** at the projected 2D position. You know the 3D point and
  the camera intrinsics, so the warp is computable — and it yields auto-labelled boxes for free. Synthetic on
  synthetic, but it exercises the full cascade.
- **Feed the vision stage real pothole photos** keyed to the event, as `frame_provider` does today. Honest
  about what is being tested.
- **Accept a vision-stage no-op** and evaluate Stage 1 only.

**Level A validates the sensor stage and the plumbing. It does not validate vision.** That is Level B's job.

### What Level A delivers

- The full loop: CARLA → `sensors.csv` → orchestrator → `pave_events.json` → map UI.
- Real GPS via `transform_to_geolocation` — retires the two mock-GPS implementations.
- Real frame↔sensor sync via `frame_index.csv` — retires #18.
- Real ground-truth labels — makes `test_stage1_accuracy_against_real_labels` meaningful.
- Road network export for the UI.
- A basis for retuning the FSM thresholds and retraining the classifier.

**Effort:** ~2–4 days. **Blocked on:** nothing but a machine that runs CARLA.

### Scaffold status (2026-08-19)

Written and syntax-checked; the pure-geometry logic is unit-verified against a stubbed `carla`
module. Nothing has touched a real simulator.

| File | State |
|---|---|
| `carla_sim/config.py` | Written. `WHEEL_POSITION_SCALE` and `IMU_GRAVITY_AT_REST` are `None` until P0 measures them |
| `carla_sim/verify_setup.py` | Written — the executable form of P0 below |
| `carla_sim/scenario/potholes.py` | Written; **entry/label logic tested** |
| `carla_sim/scenario/impulse.py` | Written; adapts to whichever impulse API the build exposes |
| `carla_sim/scenario/route.py` | Written, untested |
| `carla_sim/scenario/sensors.py` | Written, untested |
| `carla_sim/scenario/drive_and_record.py` | Written, untested |

**P3 is also written** (2026-08-19): `integration/carla_frame_provider.py` plus `dataset_path` and
`frame_provider` parameters on `orchestrator.run()` and an argparse CLI. Its lookup logic **is
tested** against a fabricated run directory — no CARLA required, see the changelog.

Still to do for a working Level A: run P0, tune `IMPULSE_DELTA_V`, record a run, then point the
orchestrator at it.

---

---

## LEVEL B — prerequisite audit (2026-09-06)

**Level B is the top-priority item** ([`04-current-state.md`](04-current-state.md)) because it is the only
level that can test vision: Level A has no hole in the world, so the camera photographs clean tarmac.

Before committing to a ~165 GB source build, the toolchain was audited offline. **Verdict: go — and as of
2026-09-07 every toolchain prerequisite is met.** All four gaps were closed within a day. Blender remains
absent but gates mesh authoring (P1), not the build. Full detail, including the two traps worth keeping, is
[issue #36](40-known-issues-and-gaps.md) (now resolved). Summary:

| | Item | State |
|---|---|---|
| ✅ | Epic / CARLA GitHub access | **Verified against the private repo.** `CarlaUnreal/UnrealEngine` HEAD `2ac0528`; branches `4.26` and `carla` present; `gh` authed as `syntherat` with `repo` scope. This was the plan's "High likelihood" risk — it is retired |
| ✅ | Disk | D: 306.5 GB free vs ~165 GB needed. **Build on D:** — C: has only 25.8 GB |
| ✅ | CMake / Git / Python x64 | 3.31.11 / 2.49.0 / 3.12.10 AMD64 |
| ✅ | Windows 8.1 SDK, .NET 4.6.2 pack | present |
| ✅ | 7-Zip | installed 2026-09-06, 26.03, on PATH |
| ✅ | **C++ compiler** | **Installed by the user after the audit.** Desktop-development-with-C++ workload present (`Workload.NativeDesktop` 17.14.37314.3) |
| ✅ | **MSVC toolset** | v142 (`14.29.30133`) installed 2026-09-06, **but the engine actually built with v143 (`14.44.35207`) from a VS 2022 *BuildTools* instance** — see the correction below and issue #36 |
| ✅ | **`make` 3.81** | **Installed 2026-09-07.** `C:\GnuWin32\bin\make.exe` = GNU Make 3.81, Machine PATH position 0. NB: the chocolatey `make` 3.81 package is broken — see [issue #36](40-known-issues-and-gaps.md) for the working route |
| ✅ | Blender | **Installed** (found 2026-09-08): `D:\Program Files\Blender Foundation\Blender 5.2\` (5.2.0 LTS) and `Blender 4.4\` (4.4.1). Not on PATH — invoke by full path. Earlier rows in this file claimed it was absent; that was a search error, not reality |

### Engine built — 2026-09-07

`CarlaUnreal/UnrealEngine` branch **`carla`** (HEAD `e9d9e60c8`) cloned shallow to **`D:\dev\UnrealEngine_4.26`**;
`Setup.bat` pulled 11.5 GB of dependencies; `GenerateProjectFiles.bat` produced `UE4.sln`; and
`Build.bat UE4Editor Win64 Development` completed 4238/4238 actions in 5357 s (~89 min) with zero errors,
yielding `UE4Editor.exe` and 417 DLLs. Tree **104.51 GB**.

**INCOMPLETE as first recorded — corrected 2026-09-08.** That command builds only the *editor* target. The
engine's **Program** targets were missing, and without `ShaderCompileWorker.exe` the editor cannot compile a
single shader: it started, printed `Unable to launch .../ShaderCompileWorker.exe`, and exited. Checking that
`UE4Editor.exe` existed is what let this pass unnoticed. `ShaderCompileWorker`, `UnrealLightmass` and
`UnrealPak` were built separately on 2026-09-08 (157 actions for the first, minutes each, zero errors), after
which the editor launched and ran **13 ShaderCompileWorker processes** at full CPU. See
[issue #38](40-known-issues-and-gaps.md).

That compile settles all three toolchain unknowns — **but not the way first recorded.** **Windows 10 SDK
10.0.26100 is acceptable** and **absent `clang` is irrelevant on Windows**, both confirmed. The third was
recorded as "v142-inside-VS-2022 works"; **that is wrong.** UBT's log reads
`Using Visual Studio 2019 14.44.35228 toolchain (...\2022\BuildTools\VC\Tools\MSVC\14.44.35207)` — it used
**v143** from a **BuildTools** instance. Build CARLA with v143 to match.

**Two traps, both costly, both likely to recur on the CARLA build:**
- **Windows Defender throttles this tooling by ~4 orders of magnitude.** The dependency check ran at 38 KB/s / 3.4 % CPU until an exclusion was added for the engine tree, then **415 MB/s / 88 % CPU**. **Add an exclusion for the CARLA source tree before building it.** Symptom: low CPU, high "IO Other Operations/sec", little data moving.
- **`Setup.bat` can hang with its process alive** — frozen at 90 %, 0.00 MiB/s, 604 identical progress lines. Kill `GitDependencies` and re-run; it resumes and re-fetches only the remainder. Judge by the byte counter, not by whether the process exists.
- **CORRECTED 2026-09-07:** this previously blamed Git Bash / MSYS for mangling `cmd //c "Setup.bat"` into
  `'Setup.bat' is not recognized`. **That was wrong.** The real cause was
  `NoDefaultCurrentDirectoryInExePath=1`, inherited by every spawned shell, which stops cmd resolving
  executables in the current directory. Switching to PowerShell only appeared to fix it because absolute
  paths were used at the same time. Clear the variable; see the build recipe below.

### CARLA source build on this machine — the working recipe (2026-09-07)

`make PythonAPI` succeeds and produces
`PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl` (5.44 MB, containing a 17.67 MB
`carla/libcarla.cp312-win_amd64.pyd`). Getting there took **eight** distinct fixes. Reproduce with a
wrapper `.bat` that sets all of the following **before** invoking `make`:

```bat
set "NoDefaultCurrentDirectoryInExePath="                        rem #1 -- do this FIRST
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "UE4_ROOT=D:\dev\UnrealEngine_4.26"
set "GENERATOR="                                                 rem CARLA default; do NOT force NMake
set "MAKEFLAGS="
set "BOOST_BUILD_PATH=<dir containing the user-config.jam below>"
cd /d D:\dev\carla-source
make PythonAPI
```

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `'bootstrap.bat' is not recognized` although the file is present; `'vswhere.exe' is not recognized` from vcvarsall | **`NoDefaultCurrentDirectoryInExePath=1`** is inherited by every spawned shell. It stops cmd resolving executables in the CURRENT DIRECTORY, which CARLA's installers depend on — and it broke vcvarsall's own vswhere lookup | clear the variable. **This is the root cause of several other symptoms** |
| 2 | UBT/LibCarla ABI mismatch risk | the engine was built with **v143**, not v142 | use `BuildTools\...\vcvarsall.bat x64` with the **default** toolset (14.44) |
| 3 | `NMAKE : fatal error U1065: invalid option '='` | `make PythonAPI GENERATOR="..."` puts the assignment into `MAKEFLAGS`; **NMAKE reads MAKEFLAGS** | pass GENERATOR as an environment variable; clear `MAKEFLAGS` |
| 4 | b2 skips every object "for lack of `msvc-setup.nup`" | boost 1.84's `msvc.jam` only auto-detects Community/Professional/Enterprise under `%ProgramFiles(x86)%\...\2022\`. This machine has **BuildTools** | `user-config.jam` naming the compiler and setup script explicitly (below) |
| 5 | `dtype.cpp(101): error C2039: 'elsize' is not a member of '_PyArray_Descr'` | boost 1.84's **boost.numpy** predates the **NumPy 2** ABI break; numpy 2.5.1 is installed | build boost alone with `PYTHONNOUSERSITE=1` so numpy is invisible and boost_numpy is skipped. CARLA never uses boost_numpy. **Do not set this for the whole build** — setuptools/wheel/pip are also user-site here |
| 6 | osm2odr: `Generator "NMake Makefiles" does not support platform specification` | `BuildOSM2ODR.bat:108` passes `-A x64` **unconditionally**; only the VS generator accepts it | use CARLA's default `Visual Studio 17 2022` generator |
| 7 | `LNK1120: 50 unresolved externals`, all `xercesc_3_2` | `NMake Makefiles` is **single-config**, so `cmake --build . --config Release` was ignored and Xerces built **Debug** (`xerces-c_3D.lib`) | same fix as #6 — the VS generator is multi-config and honours `--config Release`, producing `xerces-c_3.lib` |
| 8 | LibCarla compiles single-threaded | nmake has no parallel build | same fix as #6 — MSBuild parallelises |

**Note how much of this cascades from one wrong turn.** #3, #6, #7 and the serial build were all consequences
of forcing `GENERATOR="NMake Makefiles"`, which was itself only introduced because #1 made CMake's Visual
Studio generator unable to find a compiler. Once #1 was fixed the workaround should have been reverted
immediately; leaving it in cost three failed builds.

`user-config.jam` for #4:
```
using msvc : 14.3
  : "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
  : <setup>"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"
  ;
```

**Verify by artifact, never by exit code** — see [issue #37](40-known-issues-and-gaps.md).

**Correction to this plan's earlier framing.** The Environment table below and the risk table both assumed the
Visual Studio question was a *version* problem. At audit time it was worse — there was no C++ toolchain on the
machine at all. **Both were resolved by the user later the same day**, so the Visual Studio question is now
closed; see [issue #36](40-known-issues-and-gaps.md) for the measured state and for the `vswhere` caveat.

### Stock props cannot substitute for carved geometry — measured 2026-09-07

**Question:** could a stock `static.prop.*` from the packaged 0.9.16 release give the camera a real
depression, letting Level B's vision test happen without the source build?

**Answer: no. Not one of the 99 props has a usable cavity.** The source build is unavoidable.

**Method.** Probed the live simulator (Town10HD_Opt) with `world.cast_ray`. Pass 1 spawned all 99
`static.prop.*` blueprints and recorded bounding boxes. Pass 2 took the 50 pothole-scale ones (footprint
≥ 0.30 m so a wheel could engage, ≤ 3.0 m so it is a road feature, height 0.15–2.5 m), spawned each on flat
ground, and cast a 9×9 ray grid down through it, recording the highest hit above ground per cell.

**Result — collision volumes are sealed convex hulls.** Every visually-open container (`bin`, `container`,
`clothcontainer`, `box01`–`03`, all `plantpot*`, `trashcan04/05`) returned **depth 0.00 at hit-fraction
1.00**: rays land on one uniform top surface and never enter the opening. `open_interior_cells` was **0 for
every solid prop**. The nonzero depths are hull artifacts, not cavities — `warningaccident` /
`warningconstruction` 0.98 m is an A-frame whose rays pass *between the legs* (hit 0.78); `doghouse` 0.28 m
is a sloping roof; `trashbag` 0.27, `glasscontainer` 0.19, `trashcan01/03` 0.15, `trafficcone02` 0.14,
`barrel` 0.11 are curvature — a cone tapers, a barrel domes. The only props with interior gaps are
see-through, not wheel-catching: `shoppingtrolley` (7 % hit), `plasticbag` (1 %), `plastictable` (rays
between legs).

**Inference, flagged as such:** convex-hull collision is deduced from the ray pattern, not read from the
assets' collision flags. A 100 % hit rate across the mouth of an open bin admits no other explanation, but
it was not confirmed at the asset level.

**Also settled — the geometric argument.** Even a prop with a perfect cavity could not make a true pothole:
the road surface remains, so a prop sits *on* it. The best case is rim-then-cavity — the wheel climbs a lip,
drops inside, climbs out — which contaminates the signature with an entry bump and still does not look like
a hole in tarmac to a camera. This is the same lip artifact the Level B risk table already flags for
Blender tiles, which is why Level C (holes cut into the road mesh) remains the highest-fidelity option.

**P0 item 3 is verified by the same run:** prop spawning at a map spawn point works, **99/99 blueprints
spawned successfully**, physics disabled, all destroyed cleanly.

### P1 — pothole assets: STARTED 2026-09-08

`carla_sim/assets/` now generates the Level B tiles. Four FBX plus a CARLA manifest, all measured.

| Tile | Bowl depth (measured) | Bowl radius | UCX bodies |
|---|---|---|---|
| `PotholeTile_Shallow` | 3.9 cm | 22 cm | 13 |
| `PotholeTile_Medium` | 6.9 cm | 28 cm | 13 |
| `PotholeTile_Deep` | 10.9 cm | 34 cm | 13 |
| `PotholeTile_FlatControl` | 0.0 cm (control) | — | 1 |

All measure **160.0 cm** across, i.e. the FBX scale-100 conversion is correct.

**Files**
- `carla_sim/assets/generate_pothole_meshes.py` — parametric generator, runs headless in Blender.
- `carla_sim/assets/verify_pothole_meshes.py` — re-imports each FBX and **measures** it. A clean generator
  run is not evidence; run this too.
- `carla_sim/assets/fix_package_paths.py` — repairs the asset paths CARLA writes into
  `PavePotholes.Package.json`. **Run after every import.**
- `carla_sim/assets/fbx/` — the four FBX plus `PavePotholes.json` (CARLA `Import.py` props format:
  `name` / `source` / `tag`, landing at `/Game/PavePotholes/Static/static/<name>`).

**Two decisions that follow directly from measurements in this file**

1. **Collision is authored explicitly as `UCX_` convex bodies — 12 rim wedges plus a floor slab — instead of
   letting UE hull the mesh.** The 2026-09-07 prop probe established that CARLA props get sealed convex
   collision: 0 of 99 stock props had a usable cavity, every open-topped container reading as solid. A
   single hull would turn these tiles into bumps. The verifier checks no collision body encloses the space
   above the bowl floor.
2. **A flat control tile exists and should be driven alongside the others.** Because a prop sits ON the road
   and cannot cut into it, *the deepest hole a tile can present equals its own thickness* — a 10.9 cm
   pothole necessarily means a 10.9 cm lip to climb first. This is the "tile edge lips" risk in the table
   below, and it is why Level C is rated higher fidelity. The control has the same footprint and ramp with
   no bowl, so the lip's own contribution to the IMU signature can be measured and subtracted rather than
   assumed away.

**Regenerate / re-verify**
```bash
BL="D:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
"$BL" --background --python carla_sim/assets/generate_pothole_meshes.py
"$BL" --background --python carla_sim/assets/verify_pothole_meshes.py
```

**Known bias, documented rather than tuned out** (per the risk table): bowl radii are >= 0.10 m because
CARLA's raycast wheels have zero width and would drop into holes a real tyre bridges. The generator refuses
to emit a bowl narrower than that.

**Imported into CARLA — 2026-09-08.** Four `.uasset` files under
`Unreal/CarlaUE4/Content/PavePotholes/Static/static/`, plus
`Config/PavePotholes.Package.json` registering them as props. Both import commandlets reported
`0 error(s)`.

**Exactly one `.uasset` per tile** (not one per mesh in the FBX), which is the evidence that UE consumed the
13 `UCX_` bodies as collision rather than importing them as separate static meshes.

The import route is **not** `make import` — that silently does nothing and returns 0. Use:

```bat
cd /d D:\dev\carla-source
python Util\BuildTools\Import.py
python <repo>\carla_sim\assets\fix_package_paths.py
```

The fixup is mandatory, not optional: `Import.py` registers each prop at
`<fbx_basename>.<fbx_basename>`, but an FBX carrying `UCX_` bodies has multiple root nodes and imports as
`<fbx_basename>_<mesh_name>`, so the registered path points at an asset that does not exist and the prop
fails to spawn with no error anywhere. Full detail: [issue #40](40-known-issues-and-gaps.md).

**VERIFIED IN-ENGINE — 2026-09-08.** `carla_sim/assets/probe_tile_collision.py` spawned each tile in a live
simulator and ray-probed it, the same method that measured the 99 stock props.

| blueprint | rim | floor | **depth measured** | depth authored | hit% |
|---|---|---|---|---|---|
| `static.prop.potholetile_deep` | 0.120 | 0.010 | **0.110 m** | 0.11 | 0.85 |
| `static.prop.potholetile_medium` | 0.080 | 0.010 | **0.070 m** | 0.07 | 0.85 |
| `static.prop.potholetile_shallow` | 0.050 | 0.010 | **0.040 m** | 0.04 | 0.85 |
| `static.prop.potholetile_flatcontrol` | 0.080 | 0.080 | **0.000 m** | 0.00 | 1.00 |

**The bowls are genuinely empty in CARLA's physics.** This is exactly what all 99 stock props failed: every
open-topped container there returned depth 0.00 at hit-fraction 1.00, rays landing on a convex lid that does
not visually exist. These return their true authored depth at hit-fraction 0.85 — rays entering the bowl.

The flat control reading **0.000** is the negative control: it rules out the probe manufacturing depth out of
tile thickness or ramp geometry.

Import evidence, independent of the probe: the commandlet logged **40** `Triangulating mesh UCX_... for
collision model` lines = 13 + 13 + 13 + 1, exactly the authored decomposition. UE consumed every collision
body.

**Still NOT established** (Rule 6 — do not overstate this):
- That a *wheel* actually drops in. Ray geometry is necessary, not sufficient; CARLA's raycast wheels have
  zero width and their suspension response is a separate question.
- That the FSM fires on the resulting signature. That needs a recorded drive (P2).
- The lip is real and unavoidable at Level B: the Deep tile presents an 11 cm hole *behind a 12 cm ramp*.
  **Drive `PotholeTile_FlatControl` alongside every experiment** and subtract its contribution, or every
  detection is confounded by the tile's own edge.

---

### CARLA source build on this machine — the working recipe (2026-09-07)

`make PythonAPI` succeeds and produces
`PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl` (5.44 MB, containing a 17.67 MB
`carla/libcarla.cp312-win_amd64.pyd`). Getting there took **eight** distinct fixes. Reproduce with a
wrapper `.bat` that sets all of the following **before** invoking `make`:

```bat
set "NoDefaultCurrentDirectoryInExePath="                        rem #1 -- do this FIRST
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "UE4_ROOT=D:\dev\UnrealEngine_4.26"
set "GENERATOR="                                                 rem CARLA default; do NOT force NMake
set "MAKEFLAGS="
set "BOOST_BUILD_PATH=<dir containing the user-config.jam below>"
cd /d D:\dev\carla-source
make PythonAPI
```

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `'bootstrap.bat' is not recognized` although the file is present; `'vswhere.exe' is not recognized` from vcvarsall | **`NoDefaultCurrentDirectoryInExePath=1`** is inherited by every spawned shell. It stops cmd resolving executables in the CURRENT DIRECTORY, which CARLA's installers depend on — and it broke vcvarsall's own vswhere lookup | clear the variable. **This is the root cause of several other symptoms** |
| 2 | UBT/LibCarla ABI mismatch risk | the engine was built with **v143**, not v142 | use `BuildTools\...\vcvarsall.bat x64` with the **default** toolset (14.44) |
| 3 | `NMAKE : fatal error U1065: invalid option '='` | `make PythonAPI GENERATOR="..."` puts the assignment into `MAKEFLAGS`; **NMAKE reads MAKEFLAGS** | pass GENERATOR as an environment variable; clear `MAKEFLAGS` |
| 4 | b2 skips every object "for lack of `msvc-setup.nup`" | boost 1.84's `msvc.jam` only auto-detects Community/Professional/Enterprise under `%ProgramFiles(x86)%\...\2022\`. This machine has **BuildTools** | `user-config.jam` naming the compiler and setup script explicitly (below) |
| 5 | `dtype.cpp(101): error C2039: 'elsize' is not a member of '_PyArray_Descr'` | boost 1.84's **boost.numpy** predates the **NumPy 2** ABI break; numpy 2.5.1 is installed | build boost alone with `PYTHONNOUSERSITE=1` so numpy is invisible and boost_numpy is skipped. CARLA never uses boost_numpy. **Do not set this for the whole build** — setuptools/wheel/pip are also user-site here |
| 6 | osm2odr: `Generator "NMake Makefiles" does not support platform specification` | `BuildOSM2ODR.bat:108` passes `-A x64` **unconditionally**; only the VS generator accepts it | use CARLA's default `Visual Studio 17 2022` generator |
| 7 | `LNK1120: 50 unresolved externals`, all `xercesc_3_2` | `NMake Makefiles` is **single-config**, so `cmake --build . --config Release` was ignored and Xerces built **Debug** (`xerces-c_3D.lib`) | same fix as #6 — the VS generator is multi-config and honours `--config Release`, producing `xerces-c_3.lib` |
| 8 | LibCarla compiles single-threaded | nmake has no parallel build | same fix as #6 — MSBuild parallelises |

**Note how much of this cascades from one wrong turn.** #3, #6, #7 and the serial build were all consequences
of forcing `GENERATOR="NMake Makefiles"`, which was itself only introduced because #1 made CMake's Visual
Studio generator unable to find a compiler. Once #1 was fixed the workaround should have been reverted
immediately; leaving it in cost three failed builds.

`user-config.jam` for #4:
```
using msvc : 14.3
  : "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
  : <setup>"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"
  ;
```

**Verify by artifact, never by exit code** — see [issue #37](40-known-issues-and-gaps.md).

**Correction to this plan's earlier framing.** The Environment table below and the risk table both assumed the
Visual Studio question was a *version* problem. At audit time it was worse — there was no C++ toolchain on the
machine at all. **Both were resolved by the user later the same day**, so the Visual Studio question is now
closed; see [issue #36](40-known-issues-and-gaps.md) for the measured state and for the `vswhere` caveat.

### Stock props cannot substitute for carved geometry — measured 2026-09-07

**Question:** could a stock `static.prop.*` from the packaged 0.9.16 release give the camera a real
depression, letting Level B's vision test happen without the source build?

**Answer: no. Not one of the 99 props has a usable cavity.** The source build is unavoidable.

**Method.** Probed the live simulator (Town10HD_Opt) with `world.cast_ray`. Pass 1 spawned all 99
`static.prop.*` blueprints and recorded bounding boxes. Pass 2 took the 50 pothole-scale ones (footprint
≥ 0.30 m so a wheel could engage, ≤ 3.0 m so it is a road feature, height 0.15–2.5 m), spawned each on flat
ground, and cast a 9×9 ray grid down through it, recording the highest hit above ground per cell.

**Result — collision volumes are sealed convex hulls.** Every visually-open container (`bin`, `container`,
`clothcontainer`, `box01`–`03`, all `plantpot*`, `trashcan04/05`) returned **depth 0.00 at hit-fraction
1.00**: rays land on one uniform top surface and never enter the opening. `open_interior_cells` was **0 for
every solid prop**. The nonzero depths are hull artifacts, not cavities — `warningaccident` /
`warningconstruction` 0.98 m is an A-frame whose rays pass *between the legs* (hit 0.78); `doghouse` 0.28 m
is a sloping roof; `trashbag` 0.27, `glasscontainer` 0.19, `trashcan01/03` 0.15, `trafficcone02` 0.14,
`barrel` 0.11 are curvature — a cone tapers, a barrel domes. The only props with interior gaps are
see-through, not wheel-catching: `shoppingtrolley` (7 % hit), `plasticbag` (1 %), `plastictable` (rays
between legs).

**Inference, flagged as such:** convex-hull collision is deduced from the ray pattern, not read from the
assets' collision flags. A 100 % hit rate across the mouth of an open bin admits no other explanation, but
it was not confirmed at the asset level.

**Also settled — the geometric argument.** Even a prop with a perfect cavity could not make a true pothole:
the road surface remains, so a prop sits *on* it. The best case is rim-then-cavity — the wheel climbs a lip,
drops inside, climbs out — which contaminates the signature with an entry bump and still does not look like
a hole in tarmac to a camera. This is the same lip artifact the Level B risk table already flags for
Blender tiles, which is why Level C (holes cut into the road mesh) remains the highest-fidelity option.

**P0 item 3 is verified by the same run:** prop spawning at a map spawn point works, **99/99 blueprints
spawned successfully**, physics disabled, all destroyed cleanly.

### P1 — pothole assets: STARTED 2026-09-08

`carla_sim/assets/` now generates the Level B tiles. Four FBX plus a CARLA manifest, all measured.

| Tile | Bowl depth (measured) | Bowl radius | UCX bodies |
|---|---|---|---|
| `PotholeTile_Shallow` | 3.9 cm | 22 cm | 13 |
| `PotholeTile_Medium` | 6.9 cm | 28 cm | 13 |
| `PotholeTile_Deep` | 10.9 cm | 34 cm | 13 |
| `PotholeTile_FlatControl` | 0.0 cm (control) | — | 1 |

All measure **160.0 cm** across, i.e. the FBX scale-100 conversion is correct.

**Files**
- `carla_sim/assets/generate_pothole_meshes.py` — parametric generator, runs headless in Blender.
- `carla_sim/assets/verify_pothole_meshes.py` — re-imports each FBX and **measures** it. A clean generator
  run is not evidence; run this too.
- `carla_sim/assets/fix_package_paths.py` — repairs the asset paths CARLA writes into
  `PavePotholes.Package.json`. **Run after every import.**
- `carla_sim/assets/fbx/` — the four FBX plus `PavePotholes.json` (CARLA `Import.py` props format:
  `name` / `source` / `tag`, landing at `/Game/PavePotholes/Static/static/<name>`).

**Two decisions that follow directly from measurements in this file**

1. **Collision is authored explicitly as `UCX_` convex bodies — 12 rim wedges plus a floor slab — instead of
   letting UE hull the mesh.** The 2026-09-07 prop probe established that CARLA props get sealed convex
   collision: 0 of 99 stock props had a usable cavity, every open-topped container reading as solid. A
   single hull would turn these tiles into bumps. The verifier checks no collision body encloses the space
   above the bowl floor.
2. **A flat control tile exists and should be driven alongside the others.** Because a prop sits ON the road
   and cannot cut into it, *the deepest hole a tile can present equals its own thickness* — a 10.9 cm
   pothole necessarily means a 10.9 cm lip to climb first. This is the "tile edge lips" risk in the table
   below, and it is why Level C is rated higher fidelity. The control has the same footprint and ramp with
   no bowl, so the lip's own contribution to the IMU signature can be measured and subtracted rather than
   assumed away.

**Regenerate / re-verify**
```bash
BL="D:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
"$BL" --background --python carla_sim/assets/generate_pothole_meshes.py
"$BL" --background --python carla_sim/assets/verify_pothole_meshes.py
```

**Known bias, documented rather than tuned out** (per the risk table): bowl radii are >= 0.10 m because
CARLA's raycast wheels have zero width and would drop into holes a real tyre bridges. The generator refuses
to emit a bowl narrower than that.

**Imported into CARLA — 2026-09-08.** Four `.uasset` files under
`Unreal/CarlaUE4/Content/PavePotholes/Static/static/`, plus
`Config/PavePotholes.Package.json` registering them as props. Both import commandlets reported
`0 error(s)`.

**Exactly one `.uasset` per tile** (not one per mesh in the FBX), which is the evidence that UE consumed the
13 `UCX_` bodies as collision rather than importing them as separate static meshes.

The import route is **not** `make import` — that silently does nothing and returns 0. Use:

```bat
cd /d D:\dev\carla-source
python Util\BuildTools\Import.py
python <repo>\carla_sim\assets\fix_package_paths.py
```

The fixup is mandatory, not optional: `Import.py` registers each prop at
`<fbx_basename>.<fbx_basename>`, but an FBX carrying `UCX_` bodies has multiple root nodes and imports as
`<fbx_basename>_<mesh_name>`, so the registered path points at an asset that does not exist and the prop
fails to spawn with no error anywhere. Full detail: [issue #40](40-known-issues-and-gaps.md).

**NOT yet verified: that the collision is genuinely concave in-engine.** UE consuming the UCX bodies is
necessary but not sufficient — nothing has yet confirmed the bowl is empty in the cooked asset. The decisive
test is the same `world.cast_ray` grid probe that measured all 99 stock props on 2026-09-07, run against a
spawned tile in a live simulator. Until that passes, **do not describe these tiles as working holes** (Rule 6).

### P2 — Level B recording harness: DONE and MEASURED 2026-09-08

**The sensor stage fires on fully physics-derived pothole events.** No scripted impulse anywhere: a wheel
falls into real geometry and the drop, freefall and impact all come out of the physics engine. This is what
Level A structurally could not do.

**The experiment.** Two runs, identical seed (42), route, tile positions, tick count (24,000 ≈ 60 s) and
wheel-over events. The ONLY difference is whether the tiles have cavities:

| | BOWLS | FLAT CONTROL |
|---|---|---|
| tiles spawned | 8 | 8 |
| wheel-over events | 15 | 15 |
| labelled samples | 315 | 315 |
| az max | 1616.9 | 380.8 |
| samples below DROP gate (6.81) | 398 | 284 |
| samples above IMPACT gate (19.81) | 50 | 19 |
| \|az\| > 1000 **after settling** | **8** | **0** |
| **FSM detections** | **7** | **1** |
| on a labelled pothole | 7 | 1 |
| elsewhere (false positives) | **0** | **0** |

**7 detections with cavities, 1 without — so the bowls cause the detections, not the lip.** Everything else
was held constant, which is the whole point of shipping a flat control tile. Zero false positives in both.

The supporting numbers point the same way rather than resting on the detector alone: with cavities the signal
crosses the IMPACT gate 50 times vs 19, peaks at 1617 vs 381, and produces 8 post-settling \|az\| > 1000
samples vs **zero**. Every genuine large impact in the recording comes from a cavity.

**The lip is not free, though.** The control still produced 1 detection and 284 DROP-band samples. The bare
tile edge does register — it is simply much weaker than the cavity. **Quote every future Level B figure
against its control**, never on its own.

**On the huge az value:** both runs contain an identical `az = -149443.4` at **row 1, t = 0.0025 s,
speed 0.19 m/s**. That is the already-documented spawn settling transient (see `04-current-state.md`:
"row 1 carries az = -157477 -- the vehicle settling at spawn. Skip the first few rows."), NOT a tile
artifact. It is identical in both runs because it is the same spawn. Excluding it is what leaves the clean
8-vs-0 split above.

**Files**
- `carla_sim/scenario/tiles.py` — spawns tiles at the Level A pothole locations, yaw-aligned to the lane;
  severity maps onto the three real depths; `control=True` swaps in the flat tile.
- `carla_sim/scenario/drive_and_record.py` — `--level A|B` and `--control`. At Level B the impulse is
  skipped entirely; `PotholeTracker` still runs because it produces the ground-truth labels.
- `carla_sim/analyse_run.py` — runs the sensor stage over a recording and reports detections **with the az
  envelope**, because "no detections" and "the wheel never fell far enough" are different problems.

**Reproduce**
```bash
PY=venv/Scripts/python.exe
$PY carla_sim/scenario/drive_and_record.py --town "" --level B --potholes 8 --ticks 24000 --seed 42 \
    --no-camera --out carla_sim/out/levelB_bowls
$PY carla_sim/scenario/drive_and_record.py --town "" --level B --control --potholes 8 --ticks 24000 \
    --seed 42 --no-camera --out carla_sim/out/levelB_control
```

**NOT established yet (Rule 6)**
- **No vision result.** These runs used `--no-camera`. The camera has still never seen a pothole, which is
  the reason Level B exists at all. That is the next step: record WITH frames and run the vision stage.
- No end-to-end accuracy figure; issue #18 remains open.
- 7 detections over 15 wheel-over events is not a recall figure — wheel-overs pair up (two wheels, one
  pothole), so the denominator is nearer 8. Do not quote a rate from this run.

## Phase plan (applies to A and B alike)

| Phase | Work | Effort |
|---|---|---|
| **P0** | Verify assumptions — see below. **Do not skip.** | ½ day |
| **P1** | Pothole assets (Level B only): Blender tiles → FBX → `make import` → `make package` | **STARTED 2026-09-08** — 4 FBX generated and measured; import pending. See the P1 section above |
| **P2** | Recording harness — new `carla_sim/` subsystem | **DONE 2026-09-08.** Level B recording works; 7 detections with cavities vs 1 with the flat control. See the P2 section above |
| **P3** | Wire in: `integration/carla_frame_provider.py` + a `dataset_path` param on `orchestrator.run()` | 1 day |
| **P4** | Recalibrate FSM thresholds; retrain the RandomForest on CARLA data | 1–2 days |
| **P5** | Close the vision domain gap — auto-labelled fine-tuning | 2–4 days |
| **P6** | Export the CARLA road network to the map UI | 2–3 days |
| **P7** | Live streaming mode (optional) | 2–3 days |

### Proposed layout

```
carla_sim/                          ← new subsystem, kept out of existing folders (Rule 3)
├── assets/                         Blender sources + exported FBX (Level B)
├── scenario/
│   ├── potholes.py                 registry + placement
│   ├── drive_and_record.py         sync-mode loop, sensors, logging
│   └── routes.py                   fixed waypoint routes for repeatability
├── export/
│   ├── export_road_network.py      waypoints/topology → GeoJSON
│   └── export_vision_data.py       rgb/ + semantic/ for the existing seg script
└── out/run_<ts>/
    ├── sensors.csv                 ← contract #1, column-identical
    ├── frames/<frame>.jpg
    ├── frame_index.csv             frame → timestamp   (closes #18)
    ├── gnss.csv
    └── ground_truth.json           pothole positions + wheel-over windows
```

---

## P0 — assumptions to verify before anything else

### 1. IMU gravity convention ⚠ highest risk in the plan
`PotholeDetector` assumes `az ≈ 9.81` at rest, `< 6.81` for DROP, `> 19.81` for IMPACT. If CARLA's IMU
reports gravity-free values or a flipped Z axis, **nothing will ever trigger** and the cause will look like a
logic bug for days.

**Test:** park a vehicle on flat road, read `sensor.other.imu`, print `accelerometer`. Record the result here.

### 2. Achievable tick rate
400 Hz means `fixed_delta_seconds = 0.0025`, below CARLA's comfortable range and requiring substepping
config. See the open decision below.

### 3. Prop spawning — ✅ VERIFIED 2026-09-07
**99/99 `static.prop.*` blueprints spawn successfully** at a map spawn point in the packaged 0.9.16 build
(physics disabled, all destroyed cleanly). Measured during the prop-cavity probe recorded in the Level B
section above — which also established that **no stock prop can substitute for carved geometry**.

---

## Known risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| CARLA source build friction (UE4.26, ~165 GB, painful on Windows) | High | **Toolchain complete 2026-09-07 (#36):** disk, Epic GitHub access, **v142** (cl.exe 19.29.30159) and **GNU Make 3.81** all verified. Nothing blocks the clone. Follow official docs exactly — version drift is the usual failure |
| `best.pt` detects little or nothing in CARLA (domain gap) | High | Expected. Auto-label from known 3D positions + camera intrinsics, then fine-tune. **Archive `best.pt` first** — training overwrites it |
| **CARLA IMU defaults to zero noise** | Certain | Set `noise_accel_stddev_*` and gyro equivalents. Training on noiseless data yields a classifier that solves a fake-easy problem |
| Raycast wheels have zero width and drop into holes a real tire would bridge | Medium | Author widths ≥ ~20 cm; document the bias rather than tuning it out |
| Sampling rate vs physics stability at 0.0025 s | Medium | See open decision |
| Tile edge lips reading as phantom bumps (Level B) | Medium | Bevel edges; run a control with flat tiles |
| Sim results being read as real-world results | Certain | They validate the *pipeline*, not real-world detector performance. Keep the distinction in every report |

---

## Map UI: the geo-reference decision

`world.get_map().generate_waypoints(2.0)` and `.get_topology()` give the full road network as polylines;
`transform_to_geolocation()` converts CARLA metres to lat/lng, so **contract #7 stays byte-identical.**

But CARLA towns are geo-referenced near (0, 0) — the Gulf of Guinea. Overlaying them on Google tiles puts the
roads in the ocean; anchoring them to a real city puts CARLA geometry over a mismatched real street grid.

**Proposed:** add a **CARLA mode** to the dashboard — hide the base tiles, draw the extracted GeoJSON network
over the existing dark ground. Same markers, same proximity logic, same `PotholeGuard` API (Rule 3); only the
basemap changes.

---

## Environment

| Item | Recommendation |
|---|---|
| CARLA | **0.9.16** on this machine. *Superseded 2026-09-06:* 0.9.15 was the recommendation, but PyPI ships no `cp312` wheel for it — on Python 3.12 the only installable client versions are 0.9.16 and an ancient 0.9.5. `carla-0.9.16-cp312-cp312-win_amd64` is installed and imports; `Client`, `GeoLocation`, `Map.get_topology`, `Map.transform_to_geolocation` and `Waypoint.next` all exist on it. The simulator package must match the client version. 0.10.x moved to UE5 and changed the asset pipeline |
| GPU | 8 GB VRAM or better — simulator and YOLO inference run together |
| Disk | ~165 GB for a source build (Level B/C). The packaged release is far smaller but cannot take custom props |
| Blender | 5.2.0 LTS and 4.4.1 installed at `D:\Program Files\Blender Foundation\`. **Export FBX at `global_scale=1.0`, i.e. in METRES — NOT scale 100.** CARLA's Import.py forces `bConvertSceneUnit=1` and converts units itself; scale 100 makes it happen twice and tiles land 100x oversized (issue #41) |
| Unreal | 4.26 from Epic's **CARLA fork**, not stock |

---

## Open decisions — awaiting the user

1. **Level A first, or straight to B?** (User asked for Level A detail on 2026-08-19; not yet committed.)
2. **400 Hz** to keep the existing model untouched, or **100 Hz** with retuned air-time gates and a retrained
   classifier? At 100 Hz, `min_air_time = 0.01 s` is a single sample — degenerate, as
   [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md) already flags.
3. ~~**Map rendering:** CARLA mode with no basemap, or anchor a town to a real lat/lng?~~ **DECIDED 2026-09-06 -- CARLA mode**, built and shipped. The user asked for the town drawn *and* for the real-world Google path to keep working, so it is a URL-selected mode (`?mode=carla`) rather than a replacement. Exporter: `scenario/export_map.py`. Renderer: `loadCarlaRoads()` in `app.js`. See [`13-map-ui.md`](13-map-ui.md).
4. **Replay or live?** Replay fits the current architecture directly. Live is where the cascade design pays
   off, since only a small share of rows ever trigger YOLO.

---

## For the agent picking this up

- Nothing here is built. Update this file's **Status** line the moment that changes (Rule 5).
- Claims about CARLA behaviour come from the documented API, **not from execution**. Verify in P0.
- `carla_sim/` is a new subsystem — do not reach into `pothole_detect_physics/` or `pothole_detection_app/`
  to make it work. Adapt in `integration/` (Rule 3).
- Keeping contract #1 column-identical is what makes this cheap. Do not "improve" the CSV schema.
- When P3 lands, add `carla_sim/` to [`02-repository-map.md`](02-repository-map.md) and the new adapter to
  [`14-integration-layer.md`](14-integration-layer.md).
