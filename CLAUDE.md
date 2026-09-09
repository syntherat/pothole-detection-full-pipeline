# CLAUDE.md — Entry Point for AI Agents

> **PAVE** — hybrid pothole detection & driver-alert system (EPICS project, VIT Bhopal).
> Sensor/physics detection + YOLO vision detection + live map dashboard, joined by an integration layer.

---

## 🚦 STOP — Read this before doing anything

You are working in a repository that maintains its own **context system** under [`context/`](context/).

**Mandatory boot sequence for every session, every agent:**

1. Read **[`context/00-RULES.md`](context/00-RULES.md)** — the binding rules. They are not optional.
2. Read **[`context/README.md`](context/README.md)** — the index; it tells you which context files matter for your task.
3. Read **[`context/04-current-state.md`](context/04-current-state.md)** — where things stand right now,
   what the next step is, and which changes are written but never executed.
4. Read the specific context file(s) for the subsystem you are about to touch.
5. Do the work.
6. **Update the affected context files** and append an entry to
   [`context/50-changelog.md`](context/50-changelog.md) *in the same change*. This is Rule 5 and it is not skippable.

If you only read one thing, read `context/00-RULES.md`.

---

## Quick orientation

| Folder | Subsystem | Language | Context file |
|---|---|---|---|
| `pothole_detect_physics/` | Sensor / IMU physics + RandomForest classifier | Python | [`context/10-physics-sensor-pipeline.md`](context/10-physics-sensor-pipeline.md) |
| `pothole_detection_app/` | YOLO vision detector + Tkinter GUIs + training pipeline | Python | [`context/11-vision-pipeline.md`](context/11-vision-pipeline.md), [`context/12-vision-training-scripts.md`](context/12-vision-training-scripts.md) |
| `pothole_map_ui/` | PAVE map dashboard (Google Maps, GPS, proximity alerts) | HTML/CSS/JS | [`context/13-map-ui.md`](context/13-map-ui.md) |
| `integration/` | Cascade orchestrator that wires all three together | Python | [`context/14-integration-layer.md`](context/14-integration-layer.md) |

**The one-sentence data flow:**
sensor CSV row → physics FSM + RandomForest → (if both agree) mock frame + mock GPS → YOLO two-stage vision → weighted fusion → `integration/pave_events.json` → map UI polls the file every 3 s → red marker + proximity alert.

Full detail: [`context/03-architecture-dataflow.md`](context/03-architecture-dataflow.md).

---

## Things that will bite you immediately

These are documented in full in [`context/40-known-issues-and-gaps.md`](context/40-known-issues-and-gaps.md). The short version:

- `model/road_seg.pt` is present (added 2026-08-19) — a 7-class `yolo11s-seg` visible-road model. **Corrected 2026-09-06: two-stage detection is NOT live.** The checkpoint loads and runs, but predicts `visible_road` on 0/16 dash-camera test frames even at conf=0.05, so the road mask is always empty and the cascade always falls back to the lower-60% crop at `two_stage_detection.py:153`. It is still the silent single-stage fallback — measured, see issue #30.
- `integration/frame_provider.py` prefers `pothole_detection_app/data/sample_images/` (gitignored) but falls back through five more candidate directories, warning once if all are empty rather than raising. **Corrected 2026-09-06** — this file previously claimed a `FileNotFoundError` on the first triggered event, which stopped being true when the fallback chain landed. The real failure mode is quieter and worse: the cascade runs, the vision stage no-ops, and `final_confidence` comes from the sensor vote alone while still looking like a complete run. 17 stand-in road photos were added 2026-09-06, so it is populated on this machine.
- `pothole_map_ui/config.js` is gitignored. Copy `config.example.js` → `config.js` and add a Google Maps key, or the map renders an error box.
- The map UI must be served over **http://**, not `file://`, or the `pave_events.json` poll fails on CORS.
- Several paths are hardcoded to a specific Windows machine (`C:\Users\palso\OneDrive\Desktop\VITB\...`) and are dead.
- **`context/`, `CLAUDE.md` and `AGENTS.md` are gitignored on purpose.** `git status` reads clean even with uncommitted context edits — that is expected, not a bug. Never propose committing them, and never reference them from the root `README.md`, which ships to people who will not have them.

---

## Repo facts

- Platform: Windows (primary dev machine). Shell examples assume PowerShell or Git Bash.
- Two separate Python dependency sets — `pothole_detect_physics/requirements.txt` (light: pandas/sklearn) and `pothole_detection_app/requirements.txt` (heavy: torch/ultralytics). The integration layer needs **both**.
- No build system for the UI. Plain files, no bundler, no npm.
- License is proprietary ("All Rights Reserved") — see `pothole_detection_app/LICENSE` and the root README.
