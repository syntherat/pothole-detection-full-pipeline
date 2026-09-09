# 04 — Current State & Handoff

**Scope:** where the project actually stands, what was done most recently, and what to do next.
**This file is mutable** — unlike [`50-changelog.md`](50-changelog.md), which is append-only history,
this one gets rewritten to reflect the present. Update it when the situation changes.
**Last updated:** 2026-09-09 (second session) — the issue #30 diagnostic was **half-answered from the
checkpoint itself** and the other half is staged but unrunnable here; see "Item 3" below. Earlier that day:
patent disclosure prepared; prior-art search found that the vision architecture is anticipated (Xing & Yang, Feb 2026) and that two VIT applications cover adjacent ground. See the patent section below. No code changed. Level B P3 remains the technical state: `best.pt` confirms **0/45** on real pothole frames (peak 0.302 vs a 0.35 gate), and fine-tuning is still the critical path.

---

## Read this first if you are resuming

**The headline: the physics detector now fires on data it did not generate.** As of the morning of
2026-09-06 it fired only on `generate_dataset.py`'s hand-written pulses. It now detects potholes in CARLA
recordings, and the reason it previously could not was a genuine defect, not a tuning gap. See issue #35 —
it is the most important thing in the repository right now.

**What is still not true:** there is no validated end-to-end accuracy figure for anything. **Vision has now
been tested and it fails**: on 45 frames containing real CARLA potholes, `best.pt` confirmed **0**, peaking
at 0.302 against a 0.35 gate. The camera has finally seen a pothole; the model does not recognise it. See
the P3 changelog entry and `carla_sim/out/levelB_vision/vision_results.csv`.

Boot order: [`00-RULES.md`](00-RULES.md) → [`README.md`](README.md) → this file → whichever subsystem you
are touching.

---

## Where this is running now

**The Windows machine**, `D:\dev\PAVE\pothole-detection-full-pipeline`. Earlier sessions ran on a MacBook
Air, which could not run CARLA; its memory did not travel.

| | |
|---|---|
| Python | 3.12.10, venv at `venv/` |
| torch | **2.14.0+cu126, `cuda True`**, RTX 4060 Laptop 8 GB |
| CARLA | **0.9.16** simulator at `D:\dev\pothole`, client wheel matches |
| Other | ultralytics 8.4.142, **opencv 5.0.0**, **pandas 3.0.5**, numpy 2.5.2, scikit-learn 1.7.2, joblib 1.6.0 |
| Disk | D: ~300 GB free, C: ~23 GB free |

**On Windows, install torch FIRST from PyTorch's CUDA index.** The requirements file alone yields a CPU-only
wheel; the only symptom is `cuda False`. See [`30-setup-and-run.md`](30-setup-and-run.md).

**A bare `pytest` from the repo root runs ZERO tests** — `pothole_detection_app/test_detection.py` calls
`sys.exit(1)` at import and kills the collector (issue #7). Use:
```bash
python -m pytest integration/test_integration.py -q     # 9 passed, 1 skipped
```

---

## What works, with numbers

Everything here was measured on this machine on 2026-09-06, not inferred.

| Capability | Evidence |
|---|---|
| Full cascade, synthetic data | **40 confirmed events**, mean confidence 0.767 |
| Event-level scoring | **153/192 events, recall 79.7 %, precision 100 %, 0 spurious** |
| Detector on CARLA | **7 sensor-stage detections** over 9 potholes driven |
| CARLA P0 | `IMU_GRAVITY_AT_REST = 9.3483` (positive — the FSM's assumption holds), wheel units **cm** |
| Map exporter | Town03 **503 segments**, Town10HD_Opt **200**, validated against the live simulator |
| Dashboard CARLA mode | Renders the town; **car drives the route and proximity alerts fire at 25 m** |
| Vision on dashcam frames | 10/16 confirmed, **0/4** false positives, IoU 0.6–0.89 against shipped labels |

**Which simulator to start — read this before recording anything.** There are two, and they are not
interchangeable:

| | `D:\dev\pothole` (stock 0.9.16 package) | `D:\dev\carla-source` (source build, UE4Editor) |
|---|---|---|
| Level A (scripted impulse) | **yes** | yes |
| **Level B (tile props)** | **NO — blueprints do not exist** | **yes, the only option** |
| start cost | seconds | editor launch, then press Play |

The PAVE tiles are imported into the *source* build's content
(`carla-source/Unreal/CarlaUE4/Content/PavePotholes/`). The stock package has 99 `static.prop.*`
blueprints and **none** of them are `potholetile_*`. Pointing a Level B run at it fails with
`IndexError: blueprint 'static.prop.potholetile_medium' not found` **after** loading the map and placing
the potholes, so it looks like a scenario bug rather than a wrong-simulator bug. `make package` has never
been run — there is no `Dist/`, so there is no cooked Level B build to use instead. (Cost this session a
wasted launch, 2026-09-08.)

**Starting the Level B editor (`make launch`) needs a VS x64 environment.** `make launch` re-runs Setup,
which looks for `cl.exe` and dies with *"Can't find Visual Studio compiler (cl.exe) ... You are not using
Visual Studio x64 Native Tools Command Prompt"* if the shell has not sourced it. A plain terminal — or any
shell this agent spawns — is not that prompt. Both fixes are needed, in this order:

```bat
set "NoDefaultCurrentDirectoryInExePath="
call "C:\Program Files (x86)\Microsoft Visual Studio2\BuildTools\VC\Auxiliary\Buildcvars64.bat"
cd /d D:\dev\carla-source
make launch
```

With the targets already built this reaches "Launching Unreal Editor..." in seconds; the wait is the editor
itself, not a rebuild. **The editor then has to be put into Play by hand** — a recording cannot start until
it is, because the RPC server only serves during PIE. (Found 2026-09-08.)

**Reproduce the CARLA demo (LEVEL A ONLY — see the table above):**
```bash
# 1. CARLA, at DEFAULT quality -- -quality-level=Low crashes this build (issue #32)
cd D:\dev\pothole && ./CarlaUE4.exe -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000
# 2. record  (--town "" reuses the loaded world; load Town03 from Python, not the CLI)
python carla_sim/scenario/drive_and_record.py --town "" --ticks 24000 --no-camera
# 3. serve the dashboard from the REPO ROOT, then replay
python -m http.server 5500
python integration/carla_replay.py carla_sim/out/run_<ts> --speed 3 --reset --loop
# 4. open  http://localhost:5500/pothole_map_ui/index.html?mode=carla&town=Town03
```

---

## What to do next, in order

### 1. Level B — carve real geometry  ⭐ the only way to test vision
**Level B is WORKING as far as the sensor stage (2026-09-08).** Tiles are real cavities and the FSM fires
on physics-derived drops; the flat control confirms the bowls cause it. **Vision remains untested — the
camera has still never seen a pothole.** The CARLA source build is complete and verified: the editor renders
`Town10HD_Opt` with 4,013 actors. Engine, LibCarla, CarlaUE4 plugin and the PythonAPI wheel all
build from source. What remains for Level B is the content work, not the build.

**Done:**
- **Engine:** `CarlaUnreal/UnrealEngine` branch `carla` (HEAD `e9d9e60c8`) at `D:\dev\UnrealEngine_4.26`.
  `Build.bat UE4Editor Win64 Development` — 4238/4238 actions, ~89 min, zero errors. 104.51 GB.
  **Plus the Program targets, which that command does NOT build** — `ShaderCompileWorker`, `UnrealLightmass`,
  `UnrealPak`, built separately 2026-09-08. Without `ShaderCompileWorker` the editor cannot compile any
  shader and exits at startup ([issue #38](40-known-issues-and-gaps.md)).
- **CARLA source:** `carla-simulator/carla` tag `0.9.16` (HEAD `294096e`) at `D:\dev\carla-source`.
- **Content:** 20.09 GB pack, byte-exact, extracted to 43,359 files / 23.36 GB.
- **`make PythonAPI`:** produces `PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl` (5.44 MB,
  containing a 17.67 MB `carla/libcarla.cp312-win_amd64.pyd`). Zero compile or link errors.
- Toolchain: **v143** (`cl.exe` 19.44.35228) from the VS 2022 **BuildTools** instance, GNU Make 3.81,
  CMake 3.31.11, 7-Zip 26.03, Epic GitHub access. D: ~175 GB free.

**Correction to earlier entries in this file:** the toolchain is **v143, not v142**. UBT's log names
`...\2022\BuildTools\VC\Tools\MSVC\14.44.35207`. The v142 component installed on 2026-09-06 was probably
never needed. Building LibCarla with v142 would be ABI-incompatible with the plugin.

**The CARLA build needs eight specific fixes** — full table in
[`15-carla-testbed-plan.md`](15-carla-testbed-plan.md) under "the working recipe". The one to know:
**`NoDefaultCurrentDirectoryInExePath=1` is inherited by every spawned shell** and breaks bare-command
resolution *and* vcvarsall's vswhere lookup. Clear it first; several other symptoms disappear with it.

**Never trust `make`'s exit code here** — [issue #37](40-known-issues-and-gaps.md). `BuildOSM2ODR.bat` and
`BuildPythonAPI.bat` both print success banners and return 0 after failing. Verify `dist/` contains the
`.whl`.

**Next, in order:**
1. **Level B P1 + P2 DONE (2026-09-08).** Tiles authored, imported, verified in-engine as real cavities
   (0.110 / 0.070 / 0.040 m, control 0.000), and **the sensor stage now fires on fully physics-derived
   events** — no scripted impulse anywhere.
   **The controlled result:** two runs, same seed/route/tile positions/15 wheel-over events, differing only
   in whether the tiles have cavities → **7 FSM detections with bowls, 1 with the flat control, 0 false
   positives in both.** The cavities cause the detections, not the tile lip. Detail and caveats:
   [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md) P2.

2. **P3 DONE (2026-09-08) — the vision run happened, and the model failed it.**
   `carla_sim/out/levelB_vision`: 1,200 frames, camera on, same seed and tiles. Its sensor stage reproduces
   the bowls run (7 detections, same timestamps to within 5 ms), so **frames and sensor rows describe the
   same holes — issue #18 is closed.**
   Scored over 45 picked frames (15 events x 2.0/1.0/0.5 s lead): **0/45 confirmed at conf=0.35**, 25/45
   detect something at conf=0.05, peak **0.302**. Mean confidence rises with proximity
   (0.045 -> 0.091 -> 0.103), which suggests the model is faintly seeing the real holes rather than firing
   on texture — **but box-vs-tile spatial correspondence was NOT verified.**
   Raw per-frame numbers: `carla_sim/out/levelB_vision/vision_results.csv`.

3. **NEXT — fine-tune on CARLA frames. This is now the critical path.**
   The documented fallback is no longer a contingency. Auto-label from the known 3D tile positions plus
   camera intrinsics (1,200 frames, exact coordinates — labels are derivable, not hand-drawn), then
   fine-tune. **Archive `best.pt` before any training run.**
   The projection code doubles as the check on whether today's 0.05-threshold boxes actually sit on the
   tiles, so write that first — it answers a real question either way.
   **Do not lower the confidence gate to manufacture detections.** No false-positive rate has been measured
   for vision on CARLA frames, so detections at 0.25-0.30 mean nothing about precision (Rule 6).

4. Then P4 (retune the FSM / retrain the classifier on CARLA data) and `make package` for a cooked build.

**Always run the flat control alongside any Level B experiment.** A prop sits on the road, so every tile is
a hole behind a ramp of the same height; the control still produced 1 detection and 284 DROP-band samples.
A Level B number quoted without its control does not separate the cavity from the edge.

**The build phase is done and verified — do not redo it.** If `make launch` is ever re-run, remember:
- The splash parks at 95 % for a long time. On this machine asset discovery alone took **32 minutes**
  (`LogAssetRegistry: Asset discovery search completed in 1920.5251 seconds`). Not a hang.
- **Read `Unreal/CarlaUE4/Saved/Logs/CarlaUE4.log`** for real progress — it is timestamped and names the
  current phase. Do not infer from CPU counters.
- Answer **"Not Now"** to "Project file is out of date. Would you like to update it?" — letting UE rewrite
  `CarlaUE4.uproject` is a needless way to break the build.

**Why Level B is still the top item:** the sensor stage is validated, vision is not, and Level A
definitionally cannot test vision — there is no hole for the camera to see. This is the only remaining path
to an end-to-end accuracy number, which is issue #18, the oldest open gap in the project.

**The prop shortcut is closed (2026-09-07).** All 99 `static.prop.*` blueprints were ray-probed: none has a
usable cavity, and no prop can make a true hole regardless, since the road surface stays put. That run also
verified P0 item 3 (99/99 props spawn).

**Read issue #35 before starting.** Level B would NOT have fixed the FSM defect — a real hole produces the
same continuously-rising rebound the old FSM discarded. That fix was needed regardless and is already in.

### 2. Tune the detection rate on CARLA  (~1 hour)
7 detections across 9 potholes driven, and detections arrive in pairs (two wheels, same pothole), so roughly
4–5 distinct potholes of 9. Weaker severities do not clear the impact gate. `UNLOAD_FORCE_SCALE` and
`UNLOAD_TICKS` are **coupled** — the sweep table is in [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md).

### 3. Smoke-test the two GUIs  (~30 min)  ← STILL OUTSTANDING
Neither `best.pt` nor `road_seg.pt` has been loaded through a GUI on this machine. Dependencies are
installed, so this is now just launching them.

### 4. Stages C and D of the reviewed change log
From `PAVE_v2_change_log_REVIEWED.docx` (the user's document; **A and B are done**):
- **C — depth honesty.** Report engaged depth, stop claiming true geometric depth. A contract change: it
  touches `sensor_adapter.py`, `schema.py`, `fusion.py`, `orchestrator.py` and the map UI (Rules 3 and 4).
- **D — remove the hand-shaped pulses** from `generate_dataset.py`, add a 150 ms refractory window.
  **Do this one carefully:** the FSM's thresholds were tuned to those pulses, and #35 showed how deep that
  coupling runs.

---

## Patent disclosure — prepared 2026-09-09, and what the prior-art search found

The PAVE inventive disclosure form was completed this session. Working files are in `docs/`
(gitignored); the filled form itself lives outside the repo. **Two findings from the prior-art search
matter to the project, not just to the filing.**

**1. The vision architecture is not novel as such.** Xing & Yang, *Journal of Imaging*, February 2026
disclose the same arrangement PAVE uses — a YOLO-family detector run on unmasked frames, with road
segmentation applied afterwards to filter the results — differing only in that they retain a detection
by bounding-box area overlap (τ=0.5) rather than by a centroid point. It was published **before**
`two_stage_detection.py` was written (first committed 2026-08-06). Their reported result is a ~33.7 %
false-positive reduction. **Worth reading if the road mask is ever revisited** — it is the same idea,
measured.

**2. FOUR VIT documents occupy adjacent ground — three of them GRANTED.**
Corrected and expanded 2026-09-09 after searching InPASS directly; PATENTSCOPE had shown only two, and
had not shown that any were granted.

**Claim sets were subsequently read** (InPASS serves full specifications). The abstracts overstated the
threat in every case:

| | Status | What the CLAIMS cover | Verdict |
|---|---|---|---|
| **IN 202441017108** (VIT Chennai) | GRANTED | **Radar / LiDAR** at multiple mount points; **Kalman / Bayesian / Dempster-Shafer** fusion *aligning viewpoints* into one road-surface representation | 🟢 spatial registration fusion — wrong sensors, wrong fusion |
| **IN 202441016273** (VIT Chennai) | GRANTED | Camera **activated only below a pre-set speed**; **classification CNN** (conv → pool → FC); compare to stored dataset | 🟢 no detector, no road constraint |
| **IN 202241069806** (VIT University) | GRANTED | **IMU only, no camera.** Cloud aggregation; **clustering for GPS inaccuracy**; per-pothole confidence | 🟢 for the filing · 🔴 **occupies issue #43** |
| **IN 202441059963** (VIT Chennai) | pending | Accelerometer → **triggers image capture** → image analysed → GPS → severity | 🔴 anticipates limitation (ii) — already conceded in §10.3 |

The applicant on this filing is VIT Bhopal. All four were disclosed in the form. **The three granted
ones are distinguishable at claim level; only the pending one reaches PAVE's claims, and only on the
limitation the disclosure already concedes.**

**What survives as distinguishing:** the road-region constraint on the visual result, replacement of
the detection result set (so the constraint reaches fusion rather than only the display), and the
fusion invariant that `SENSOR_WEIGHT < FUSION_THRESHOLD` makes vision necessary for confirmation. That
last one is the strongest single element and is recited in the claim — **if the fusion weights are
ever retuned so that `SENSOR_WEIGHT >= FUSION_THRESHOLD`, the claim stops reading on the system.**
Noted in [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md).

**Claim-level reading also largely rehabilitated the fusion claim.** `US9626763B1` (Lytx) determines
the pothole *from sensor data* and only **stores** video afterwards — vision is not an input, and the
inventive core is a driving-avoidability / visibility **rating**. `US10967862B2` (Uber/Aurora) claims a
**network computing system** distributing map labels between fleet vehicles — no camera, no fusion in
claim 1. `IN 202641059359`'s claim 1 is a single unlimited sentence that would not survive examination,
though **its disclosure still counts against novelty** — claims govern infringement, disclosure governs
novelty, and those are different questions. **Nothing read at claim level combines two independent
modality verdicts numerically against a threshold.**

**One claim set remains unread:** `IN 202541089617` — InPASS returns *"Claims: Please see the
attachment"*, a PDF it does not serve as text. Its abstract describes **early** fusion (raw data merged
before one classifier), architecturally distinct from PAVE's late decision fusion, but that should be
confirmed from the attachment.

**InPASS was searched on 2026-09-09** — the user entered each captcha, and the searches and result
extraction were driven programmatically. **1,103 documents examined** across the granted and pending
collections. Full detail in `docs/INPASS-FINDINGS.md`. Three method traps worth knowing:

1. **Ticking Published + Granted returns GRANTED ONLY.** The pending set is a separate collection
   (`PublicationConnection`) reachable only by ticking Published alone. An abstract search for
   `pothole` returns 23 granted and **262 pending** — searching "both" sees 8 % of the field.
2. **The two collections format IPC codes differently** — `G06V20/56` on granted records,
   `G06V0020560000` on pending ones.
3. **The InPASS IPC field returned 0 results in both formats** for a code that provably exists on
   indexed records. The IPC sweep — the only vocabulary-independent search — remains **unrun**, and is
   the largest gap. A commercial database is needed.

Google Patents does not index India at all (verified: a "pothole detection" query returns 49,289
results across CN/US/JP/KR/WO/EP/DE and zero IN). Espacenet is behind Cloudflare bot verification and
was not searched.

---

## Open decisions — for the user, not an agent

1. **Which GUI is the product?** (issue #23) `main_enhanced.py` and `pothole_app_filtered.py` share no code
   and have two different road-mask implementations. Nobody has chosen.
2. **Is the system "two-stage"?** (issue #30) `road_seg.pt` predicts `visible_road` on **0/16** dashcam
   frames; every detection so far came from the lower-60 % crop fallback. Either fix it, drop the stage, or
   describe the system accurately — but do not claim two-stage in a writeup until one of those happens.
   **Still open.** The patent disclosure took the third route: it describes both mask-derivation paths as
   co-equal embodiments and attaches no performance claim to either, so nothing was asserted that #30
   contradicts. That resolved the *writeup* problem, not the underlying one — the checkpoint still never
   fires `visible_road`. See #30 for how the wording was handled.
3. **400 Hz or 100 Hz?** Unchanged. 400 keeps the model untouched; 100 needs retuned air-time gates.
4. ~~Map rendering for CARLA~~ — **DECIDED.** CARLA mode, `?mode=carla`, real-world path untouched.

---

## Priority issues

| | Issue | Note |
|---|---|---|
| 1 | **#18** frame↔sensor sync | Still the reason no accuracy figure exists. Level B is the path |
| 2 | **#30** stage 1 is inert | `road_seg.pt` never fires; the cascade is single-stage plus a crop |
| 3 | **#33** the AI filter rejects nothing | It confirmed 154/154 physics candidates. Improving the classifier from 80 %→96 % recall moved end-to-end recall by **zero** |
| 4 | **#34** committed dataset mislabels 4.2 % of positives | Fixed in the generator; the committed CSV still has it |
| 5 | **#7** `test_detection.py` aborts pytest | Blocks the whole suite from collecting |

Resolved 2026-09-06: **#27** (joblib/numpy — no pin needed), **#28** (vehicle feed), **#31** (orchestrator
could not start), **#35** (detector did not transfer). Environment gotcha: **#32** (`-quality-level=Low`
crashes CARLA, and the crash masquerades as a client timeout).

---

## Traps that cost time today — do not re-learn these

- **A CARLA crash looks like a timeout.** The RPC port opens *before* the shader crash, so the client
  connects and then blocks. Check `Get-Process CarlaUE4*` before touching `CARLA_TIMEOUT_S` (#32).
- **`sensors.csv` row 1 carries `az = -157477`** — the vehicle settling at spawn. Skip the first few rows.
- **Town03's own road geometry produces 4 sub-threshold dips per 8,000 rows with no potholes at all.**
  Verified with `--potholes 0`. They look exactly like strikes; measure against that baseline, not zero.
- **Do not timestamp a pothole event at the rebound peak.** More correct physically, and it halves cascade
  recall while the detector fires *more* (#35).
- **Accuracy hides leakage.** The grouped-split fix moved accuracy 0.11 points and recall **5.45**. ~97 % of
  rows are negative. Read recall.
- **Capture library log output at the file-descriptor level.** A `redirect_stdout` count of the stage-1
  fallback read 0/17 while the messages were visibly leaking to the terminal; it is 17/17 (#29).

---

## Patent work — where it stands, and what is left

**The disclosure form is text-complete.** `PAVE PATENT DOC-5.docx` (in the user's Downloads, not the
repo) carries §6B, §6E, §8 claim (c) as amended, §9, §10.3, §11, §12, the workflow bullets and all 12
figures. The Smart Traffic template remnants are gone. Both colleague sign-offs (§6A physics, §6C
fusion) are in and approved.

**Three things remain, none of them repo code:**

| | Item | Owner |
|---|---|---|
| 1 | Pull `IN 202541089617`'s claims from the IPO e-register file wrapper — InPASS will not serve the PDF | patent agent |
| 2 | Narrow claim (a) to late decision fusion with the necessity invariant — draft language is in the report | patent agent |
| 3 | **Resolve issue #30** so limitation (iii) is demonstrated rather than only disclosed | **this repo** |
| 4 | Name the two pothole datasets in §12 — the repo records them only as "kaggle and roboflow" | whoever built `dataset_v3` |

**Item 3 is the only engineering task. Half of it is now answered; the other half is blocked on a
missing dataset.**

**Answered, without running anything.** `road_seg.pt` is a zip, and its `data.pkl` was disassembled
with `pickletools` to recover the arguments and metrics the training run wrote into it. Two of the
three loading/preprocessing hypotheses die immediately:

- **The checkpoint is not inert.** It records mask mAP50 **0.302** and box mAP50 **0.348** on the val
  split of `visible_road_seg_public_full` at the saved epoch. It segmented that data.
- **There is no `imgsz` mismatch.** It trained at **640**, which is also ultralytics' predict default,
  which is what `get_road_mask()` gets by not passing one.
- Recall is **0.28–0.32** even in-distribution, so this is a weak model regardless.

Full table, curves and caveats: issue #30 in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md).

**Blocked.** Those figures are **7-class aggregates**, and issue #30 turns on the `visible_road` row
specifically — a model strong on `vehicle` and `roadside_object` and blind to `visible_road` prints
exactly that table. `pothole_detection_app/scripts/validate_road_seg.py` (new, 2026-09-09) produces the
per-class row plus a predict-mode confidence sweep and the in-distribution fallback rate. **It has not
been run and cannot be run from a machine without the dataset.**

**What the next agent has to do first: find the split.** The checkpoint names
`D:\epics\pothole_detect_app\data\visible_road_seg_public_full\` — note `D:\epics\`, an **older
root** than today's `D:\dev\PAVE\`. It is gitignored, absent on the MacBook, and its presence on the
Windows box is **unverified**. If it is gone, rebuilding means re-downloading Cityscapes + ACDC + IDD +
Mapillary through `prepare_visible_road_public_dataset.py` — hours, registered accounts, and a Rule 8
conversation before starting.

Either outcome is still useful: a fix makes limitation (iii) demonstrable, a confirmed domain gap
justifies the geometric-prior embodiment already disclosed in §6B.4.

Deliverable for the agent: **`docs/PAVE-Prior-Art-Search-Report.docx`**.

---

## Starting a fresh chat

**For patent work**, paste this:

> Continuing the PAVE patent filing. Read `context/00-RULES.md`, `context/README.md` and
> `context/04-current-state.md` first, then `docs/PRIOR-ART-REGISTER.md` and
> `docs/PAVE-Prior-Art-Search-Report.docx`.
>
> State: the disclosure form is text-complete with all figures; both colleague sign-offs are in. A
> prior-art search covering InPASS (1,103 Indian documents), PATENTSCOPE, Google Patents, USPTO and
> the literature is done, and six of the closest references were read at claim level. Claim (b)
> (quarter-car physics) is the strongest; claim (a) was largely cleared once claims rather than
> abstracts were read; limitation (ii) of claim (c) is anticipated by IN 202441059963 and is
> conceded in §10.3.
>
> **Do not re-run the searches.** What remains is [your goal].

**For engineering work**, paste something like this:

> Continuing the PAVE pothole detection project at `D:\dev\PAVE\pothole-detection-full-pipeline` on Windows.
> Read `context/00-RULES.md`, `context/README.md` and `context/04-current-state.md` first, then tell me what
> you think the next step is.
>
> Context: CARLA 0.9.16 is installed at `D:\dev\pothole` and working. The sensor stage now detects potholes
> in CARLA recordings — see issue #35, which is the important one. Vision is still untested because Level A
> has no visible pothole. I want to [your goal].

`CLAUDE.md` loads automatically and points at the same place, but naming this file gets to the present state
directly rather than the full architecture tour.
