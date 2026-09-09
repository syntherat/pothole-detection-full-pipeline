# 30 — Setup & Run

**Scope:** environments, installs, and every runnable command with its working-directory requirements.
**Last verified against:** commit `3c586b8` (2026-08-19). Primary dev platform: Windows 11.
**Rule 8 applies to training, dataset generation, and full orchestrator runs — ask before running those.**

---

## Environments

Two independent dependency sets. They can share one venv, but do not have to.

| Set | File | Weight | Needed for |
|---|---|---|---|
| Sensor | `pothole_detect_physics/requirements.txt` | Light — pandas, numpy, scikit-learn, joblib, matplotlib | Physics subsystem |
| Vision | `pothole_detection_app/requirements.txt` | Heavy — ultralytics, torch, torchvision, opencv-python, pillow, numpy, lxml, tqdm | Vision subsystem |

**The integration layer needs both**, plus `pytest` for the tests.

Not declared but needed by some scripts: `matplotlib` and `seaborn` for `evaluate_model.py`, `pyyaml` for `merge_datasets.py`.

**Verified working combination (2026-08-19, Windows, Python 3.12.10)** — the integration suite passes
7/1 on: `pandas 3.0.0`, `numpy 2.5.1`, **`scikit-learn 1.7.2`**, `joblib 1.5.3`, `pytest 9.1.1`.

`scikit-learn` is **pinned to 1.7.2** in `pothole_detect_physics/requirements.txt` because that is the
version that pickled `pothole_ai_model.pkl`. Do not float it — a different version loads the model with
`InconsistentVersionWarning`, which scikit-learn documents as possibly producing invalid results. If you
retrain the model, update the pin to match.

Loading still emits ~1600 `DeprecationWarning`s from joblib under numpy 2.5 — harmless today, will break
when numpy removes the API. See issue #27 in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md).

The `.bat` files expect a venv at `pothole_detection_app/venv/`.

### One venv for everything (recommended)

```bash
python -m venv venv
```

Activate — PowerShell:
```bash
.\venv\Scripts\Activate.ps1
```

**On Windows, install torch FIRST and from PyTorch's CUDA index.** The `torch>=2.0.0` line in
`pothole_detection_app/requirements.txt` resolves to a **CPU-only** wheel on Windows -- PyPI only bundles CUDA
in the Linux wheels. Installing from the requirements file alone gives you a torch that cannot see the GPU,
and the only symptom is `torch.cuda.is_available()` returning `False`:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Then both sets -- vision first, so the deliberate `scikit-learn==1.7.2` pin settles last:
```bash
pip install -r pothole_detection_app/requirements.txt -r pothole_detect_physics/requirements.txt pytest
```

Verified 2026-09-06 on Python 3.12.10 + RTX 4060 Laptop: `torch 2.14.0+cu126` (`cuda True`),
`ultralytics 8.4.142`, `opencv 5.0.0`, `pandas 3.0.5`, `numpy 2.5.2`, `scikit-learn 1.7.2`, `joblib 1.6.0`.
Note opencv and pandas both crossed a major version since these docs were written.

**Run pytest against `integration/` specifically**, not the repo root:
```bash
python -m pytest integration/test_integration.py -q
```
`pothole_detection_app/test_detection.py` is a *script*, not a pytest module -- it calls `sys.exit(1)` at
import time, which aborts pytest's collector with `INTERNALERROR` and runs **zero** tests. See issue #7.

Python 3.10+ recommended (the code uses `X | None` unions and `set[int]` generics, which need 3.9/3.10).

---

## First-run checklist

Do these before expecting anything end-to-end to work:

- [ ] Install both dependency sets.
- [ ] `pothole_detection_app/model/best.pt` — already committed. ✔
- [ ] `pothole_detection_app/model/road_seg.pt` — present (20.5 MB). Two-stage road filtering is active. ✔
- [ ] `pothole_detection_app/data/sample_images/` — add a handful of road/pothole photos (see the README
      there). Optional: the provider also falls back to `input/` and the dataset test/val splits, and a run
      with no images anywhere now skips and reports rather than crashing.
- [ ] `pothole_map_ui/config.js` — copy from `config.example.js`, add a Google Maps key.
- [ ] Serve the UI over HTTP, not `file://`.

---

## Sensor subsystem

**Working directory matters here** — `Model/run_detector_on_dataset.py` imports `pothole_detection` by bare name while living in a different folder from it.

Regenerate the dataset — **overwrites the committed 12.5 MB CSV, Rule 8**:
```bash
python pothole_detect_physics/detector_py/generate_dataset.py
```

Retrain the classifier — **overwrites the committed .pkl, Rule 8**:
```bash
python pothole_detect_physics/Model/train_ai_model.py
```

Grouped-by-event split is now the default, and both the leaky and honest scores are printed. Add the
physics-aware rolling features (writes a **separate** model file, does not replace the live one):
```bash
python pothole_detect_physics/Model/train_ai_model.py --rolling
```

Standalone detector demo — the import was fixed 2026-08-19, so this now works from anywhere:
```bash
python pothole_detect_physics/Model/run_detector_on_dataset.py --no-plot
```

Scoring is **per event**, which is the unit that matters — a pothole spans ~12 rows, so row-wise scoring
counted one object up to twelve times. `--no-plot` skips the matplotlib window. To evaluate the rolling
model the features must match how it was trained, or the script refuses to run:
```bash
python pothole_detect_physics/Model/run_detector_on_dataset.py --no-plot --rolling \n    --model pothole_detect_physics/Data/pothole_ai_model_rolling.pkl
```

Processes all 80,000 rows and opens a matplotlib window at the end.

---

## Vision subsystem

All commands below run from `pothole_detection_app/`.

### GUIs

Filtered app — realtime video, class filters, per-event metadata (this is the one `run_app.py` launches):
```bash
python run_app.py
```

Enhanced app — batch processing, export, performance monitor:
```bash
python app/main_enhanced.py
```

### Headless

Smoke test (expects `input/` images — absent on a fresh clone):
```bash
python test_detection.py
```

Single image (relative model path, so run from `pothole_detection_app/`):
```bash
python scripts/predict_script.py path/to/image.jpg
```

Batch video with road segmentation:
```bash
python scripts/predict_videos.py --vids-dir ./input --output-dir ./output/videos --conf 0.35
```

Fetch a stopgap segmentation model. **This refuses to run while `model/road_seg.pt` exists**, which it does in this repo — the shipped model is a real 7-class visible-road model and the download is a COCO one with no `road` class. Use `--output` to keep both, or `--force` to overwrite deliberately:
```bash
python scripts/download_road_model.py --output model/road_seg_coco.pt
```

### Training — Rule 8, ask first

Archive the current weights before training; `train_model.py` overwrites `model/best.pt` on success.

```bash
python scripts/train_model.py --model small --hyperparams baseline
```

```bash
python scripts/evaluate_model.py --data data/dataset_v3/data.yaml
```

Windows batch pipeline (organize → train → evaluate → launch enhanced GUI):
```bash
quick_start.bat
```

`quick_start.bat` now runs 5 steps: organize (v2), merge (v3), train, evaluate, launch. Each dataset step is followed by an explicit file check, because `organize_dataset.py` exits 0 even when it finds nothing. Set its hardcoded `SOURCE_DIR` first (issue #14) or step 1 will produce nothing.

---

## Integration layer

Run from the **repo root** — `python integration/orchestrator.py` puts `integration/` on `sys.path[0]`, which is how the bare-name imports resolve.

⚠ **`__main__` calls `run()` with no limit — 80,000 rows with a YOLO pass per triggered row. Rule 8.** Prefer a bounded run:

```bash
python -c "import sys; sys.path.insert(0,'integration'); import orchestrator; orchestrator.run(limit=2000)"
```

Or via the CLI:
```bash
python integration/orchestrator.py --limit 2000
```

Against a recorded CARLA run — real time-synced frames and real GNSS instead of the mocks:
```bash
python integration/orchestrator.py --dataset carla_sim/out/run_<ts>/sensors.csv --carla-run carla_sim/out/run_<ts>
```

Inspect a run's frame/GPS coverage without running the cascade (needs no CARLA):
```bash
python integration/carla_frame_provider.py carla_sim/out/run_<ts>
```

Full run, no limit (only when you mean it — it now warns first):
```bash
python integration/orchestrator.py
```

Tests — `-s` is what shows the Stage-1 precision/recall numbers. Takes about 65 s; the Stage-1
accuracy test runs the FSM and `predict_proba` over 2000 rows:
```bash
pytest integration/test_integration.py -v -s
```

Tests skip gracefully when the `.pkl` or sample images are missing, so a fresh clone still gets a green run for the plumbing tests.

Publish a filtered-GUI video run to the map (translates `events.jsonl` into contract #7):
```bash
python integration/filtered_gui_adapter.py pothole_detection_app/output/video_detect_<ts>
```

Follow a run live, while the GUI is still writing:
```bash
python integration/filtered_gui_adapter.py pothole_detection_app/output/video_detect_<ts> --watch
```

Confirmed events append to `integration/pave_events.json`. Deleting that file is safe and resets the map's
history. Re-running the adapter is idempotent — it seeds from what is already published.

---

## CARLA testbed (Level A)

Scaffold only — **never executed.** See [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md) and
`carla_sim/README.md`.

Needs a running CARLA simulator plus the client library:
```bash
pip install -r carla_sim/requirements.txt
```

**Launching the simulator (Windows, verified 2026-09-06):**
```bash
cd D:\dev\pothole
./CarlaUE4.exe -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000
```

⚠ **Do not pass `-quality-level=Low`.** On CARLA 0.9.16 with an RTX 4060 it crashes the simulator during
startup with `LowLevelFatalError [Line: 139] Shader compilation failures are Fatal`. The failure is
especially misleading from a script's point of view: the RPC port opens *before* the crash, so a client
connects happily and then blocks until timeout on a dead process, producing "time-out while waiting for the
simulator, make sure the simulator is ready" — which points at the timeout rather than the crash. If you see
that message, check whether `CarlaUE4` is still in the process list before touching `CARLA_TIMEOUT_S`.

Passing a bare town name (`./CarlaUE4.exe Town03`) is silently ignored on this build — it boots
`Town10HD_Opt` regardless. Use `client.load_world("Town03")` from Python instead.

**Always run P0 first.** It measures the IMU gravity convention and the wheel-position units, and
prints two values to paste into `carla_sim/config.py`:
```bash
python carla_sim/verify_setup.py
```

Record a run:
```bash
python carla_sim/scenario/drive_and_record.py --no-camera --ticks 40000
```

Export the town's road network for the dashboard's CARLA mode (needs the simulator running, but not a
recorded run):
```bash
python carla_sim/scenario/export_map.py
```
Writes `carla_sim/out/<Town>_roads.geojson` and prints the dashboard URL to open.

The recorder refuses to start while `config.WHEEL_POSITION_SCALE` is `None`, by design — guessing it
wrong means wheel-over detection silently never fires.

---

## Map UI

Create the config once:
```bash
cp pothole_map_ui/config.example.js pothole_map_ui/config.js
```

Then edit `config.js` and set `window.GOOGLE_API_KEY` to a real key. **Never commit it** (Rule 7).

Serve from the **repo root**, so that `../integration/pave_events.json` resolves:
```bash
python -m http.server 5500
```

Then open `http://localhost:5500/pothole_map_ui/index.html`.

VS Code Live Server works too — open the folder at the repo root, not at `pothole_map_ui/`.

**Why the root matters:** `app.js` fetches `../integration/pave_events.json`. Served from `pothole_map_ui/`, that path escapes the served tree and 404s forever.

Geolocation needs `https://` or `localhost`. A LAN IP over plain HTTP will not get GPS.

---

## The full end-to-end demo

1. Create `pothole_detection_app/data/sample_images/` and drop in several road photos, at least some with potholes.
2. Create `pothole_map_ui/config.js` with a valid Maps key.
3. Serve the repo root: `python -m http.server 5500`.
4. Open `http://localhost:5500/pothole_map_ui/index.html` and allow location access.
5. In another terminal, run a bounded orchestrator pass (the `-c` command above).
6. Confirmed events append to `pave_events.json`; the map picks them up within 3 seconds as red markers with toasts.

If you have no sample images and just want to see the UI work, click **+ Simulate Detection** — it needs no backend at all.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No stand-in images found. Searched: ...` (warning, not a crash) | Add photos to `data/sample_images/` — see the README there. The run continues and reports how many events it skipped |
| `ModuleNotFoundError: pothole_detection` | Fixed 2026-08-19 — the script inserts `detector_py/` on `sys.path` itself. If it recurs, that insert has been removed |
| `ModuleNotFoundError: schema` | Running an `integration/` module without its directory on `sys.path`. Invoke `orchestrator.py` as a script, or insert the path |
| `Road segmentation model not found` in logs | **No longer expected** — `road_seg.pt` ships with the repo. If you see this, the file is missing or corrupt and detection has dropped to unfiltered single-stage |
| Confidence slider does nothing (enhanced GUI) | The `_conf` import trap — see [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md) |
| Map shows "GOOGLE_API_KEY missing" | `config.js` not created |
| `Could not poll pave_events.json` every 3 s | Opened via `file://`, or not served from the repo root |
| `ERROR: Data config not found` when training | `dataset_v3` does not exist. Run `merge_datasets.py` |
| Ultralytics downloads weights on first run | Normal — base checkpoints are fetched on demand |
| CUDA out of memory | Lower `batch` in the preset, or `--device cpu` |
| OpenCV write failures on non-ASCII paths | Use `utils._imwrite_unicode`, not `cv2.imwrite` |
| Tkinter missing on Linux | `apt install python3-tk` |
