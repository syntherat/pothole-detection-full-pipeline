# 00 — RULES (BINDING)

> **Read this file first, in full, before any other file in this repository.**
> These rules apply to Claude, Claude Code, Cursor, Copilot, Codex, Gemini, and any other AI agent
> or automated contributor operating on this repo. They also apply to human contributors who use
> those tools. They override default agent behaviour and habits.

---

## Rule 0 — Boot sequence

Before your first tool call that reads or writes project code, you must have read:

1. This file (`context/00-RULES.md`).
2. [`context/README.md`](README.md) — the index.
3. Every context file listed in the index as **relevant to the subsystem you are touching**.

If a task spans two subsystems (e.g. changing the event schema, which touches integration *and* the map UI), read the context files for **both**, plus [`context/20-data-contracts.md`](20-data-contracts.md).

Do not skim. These files exist so you do not have to re-derive the system from source every session — but they are only worth reading if you actually read them.

---

## Rule 1 — Context files are the source of truth for *intent*; code is the source of truth for *behaviour*

- When the context files and the code disagree about **what the code does**, the code wins — and you must **fix the context file** as part of your change, and note the correction in the changelog.
- When the context files and the code disagree about **what the code is supposed to do** (design intent, thresholds chosen deliberately, mocks that are placeholders), the context file wins — flag the code as a possible bug rather than silently "fixing" the docs to match.
- Never delete a documented constraint because it is inconvenient. Raise it.

---

## Rule 2 — Verify before you assert

The context files are a snapshot maintained by humans and agents. Before you rely on a specific fact — a file path, a function name, a threshold value, a CLI flag — **check it in the source**. Especially:

- File and directory paths (several in this repo are stale or machine-specific).
- Model weight files (both `best.pt` and `road_seg.pt` now ship with the repo — this was not true before 2026-08-19, so older notes may say otherwise).
- Dataset directories (most are gitignored and absent on a fresh clone).

If you cannot verify something, say so explicitly in your response rather than presenting it as fact.

---

## Rule 3 — Respect subsystem boundaries

The four subsystems are deliberately decoupled. The **only** sanctioned coupling points are:

| Boundary | Mechanism | Defined in |
|---|---|---|
| physics to integration | `PotholeDetector.process_sample()` return dict | `integration/sensor_adapter.py` |
| vision to integration | `TwoStageDetector.detect_potholes()` returning YOLO `Results` | `integration/vision_adapter.py` |
| integration to map UI | `integration/pave_events.json` (polled file) | `integration/pave_connector.py`, `pothole_map_ui/app.js` |
| any model to map UI | `window.PotholeGuard.reportDetection(...)` | `pothole_map_ui/app.js` |

**Do not** add new cross-subsystem imports, shared globals, or direct file reaches that bypass these. If a new coupling point is genuinely needed, add it *as an adapter* in `integration/` and document it in [`14-integration-layer.md`](14-integration-layer.md) and [`20-data-contracts.md`](20-data-contracts.md).

**Do not** edit `pothole_detect_physics/` or `pothole_detection_app/` internals to satisfy the integration layer. Adapt in the adapter. Those two folders are independently runnable projects and must stay that way.

---

## Rule 4 — Changing a contract is a multi-file change

If you touch any of the following, you must update **every** consumer and the corresponding contract doc:

- `integration/schema.py` — the `PotholeCandidateEvent` fields
- `integration/pave_connector.py` — the JSON written to `pave_events.json`
- `pothole_map_ui/app.js` — `pollPaveEvents()` / `PotholeGuard` signatures
- `pothole_detect_physics/detector_py/pothole_detection.py` — the `process_sample()` return dict
- The `synthetic_pothole_dataset.csv` column set
- `pothole_app_filtered.py` — the `events.jsonl` / per-event JSON record shape

Contract docs live in [`20-data-contracts.md`](20-data-contracts.md). A contract change with a stale doc is an incomplete change.

---

## Rule 5 — Update the context files in the same change (NOT OPTIONAL)

**Whenever you change code, configuration, data shapes, or behaviour, you must update the affected context files before you report the task as done.**

Checklist — for every change, ask these and answer them in your final response:

- [ ] Did a **file get added, renamed, moved, or deleted**? Update [`02-repository-map.md`](02-repository-map.md)
- [ ] Did the **flow between subsystems** change? Update [`03-architecture-dataflow.md`](03-architecture-dataflow.md)
- [ ] Did **subsystem-internal behaviour** change? Update the relevant `1x-*.md`
- [ ] Did a **data shape / JSON / CSV / dataclass** change? Update [`20-data-contracts.md`](20-data-contracts.md)
- [ ] Did a **threshold, constant, or tunable** change? Update [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md)
- [ ] Did a **run command, dependency, or setup step** change? Update [`30-setup-and-run.md`](30-setup-and-run.md)
- [ ] Did you **fix or discover a bug/gap/mock**? Update [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md)
- [ ] **Always:** append an entry to [`50-changelog.md`](50-changelog.md)

The changelog entry format is defined at the top of that file. Every entry must record *what changed*, *why*, *which context files were updated*, and *what a future agent needs to know*.

A code change with no context update is treated as an **incomplete change**, regardless of whether the code works.

---

## Rule 6 — Do not invent, do not silently fill gaps

This repo is a prototype with real, deliberate placeholders — mocked GPS, mocked frames, a synthetic dataset, a demo poll-a-file bridge. They are marked as such in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md).

- **Do not** replace a documented mock with a "real" implementation unless the user asked for exactly that.
- **Do not** invent accuracy numbers, benchmarks, or model metrics. The README's performance tables are *typical ranges from upstream sources*, not measured results for this repo's weights. Never quote them as measured.
- **Do not** fabricate dataset contents. Most `data/` subfolders are gitignored and absent on a fresh clone.
- If asked "how accurate is the system", the honest answer is: the only measurement available today is
  Stage-1 detection against a **synthetic** dataset. `test_stage1_accuracy_against_real_labels` scores
  **per event** and measured **recall 1.00, precision 1.00** over 3 events on 2026-08-19. The same test
  also prints a per-sample recall line (0.08) purely for continuity with older baselines — it caps at
  1/event-width and is **not** a quality signal. Quote the event-level numbers.
- Still true regardless: that covers **Stage 1 only, on synthetic data**. There is no vision-stage or
  end-to-end accuracy figure, and there cannot be until issue #18 is closed with real synced data.

---

## Rule 7 — Secrets and machine-specific paths

- `pothole_map_ui/config.js` holds a real Google Maps API key and is **gitignored**. Never commit it, never print the key, never inline a key into `app.js` or `index.html`.
- Root `.gitignore` already excludes `.env`, `*.key`, `config.js`, and `integration/pave_events.json`. Do not weaken these.
- Do not add new absolute paths tied to one machine. The repo already has several (documented in [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md)); adding more makes it worse. Use `Path(__file__).resolve().parent...` anchoring, which is the established pattern everywhere else.

---

## Rule 8 — Heavy operations require explicit permission

Do not start any of the following on your own initiative. Ask first, state the expected cost, and wait:

- **Model training** (`scripts/train_model.py`, `scripts/train_multiclass_road_seg.py`) — hours of GPU time.
- **Dataset generation** (`detector_py/generate_dataset.py`) — overwrites the committed 12 MB CSV, which changes every downstream number.
- **Model retraining** (`Model/train_ai_model.py`) — overwrites the committed 4.7 MB `.pkl`.
- **Full orchestrator runs** (`integration/orchestrator.py` with no `limit`) — 80,000 rows, each triggered row running a YOLO forward pass.
- **Downloading model weights** (`scripts/download_road_model.py`) — network fetch of pretrained weights.
- Deleting anything in `model/`, `Data/`, or `output/`.

For orchestrator experimentation, always use `run(limit=N)` with a small `N` first.

---

## Rule 9 — Style and conventions

Match the surrounding code; do not impose a house style.

- **Python:** stdlib `pathlib` anchored on `Path(__file__).resolve().parent`, f-strings, module-level `logger = logging.getLogger(__name__)`, docstrings on public functions. Type hints are used in newer files (`integration/`, `two_stage_detection.py`) and absent in older ones (`pothole_detect_physics/`) — follow the file you are in.
- **JS:** vanilla ES6+, no framework, no bundler, no npm. Section headers use the banner comment style. Do not introduce a build step.
- **CSS:** all colours and fonts go through the CSS custom properties in `:root` in `styles.css`. Never hardcode a hex value that a token already covers.
- **Tkinter:** the two GUI apps have their own separate colour constant blocks at the top of each file; keep them internally consistent per file. They are intentionally not shared.
- Comments explain *why*, not *what*. The existing codebase mostly follows this — keep it.

---

## Rule 10 — Reporting

In your final response for any task that changed the repo, state plainly:

1. What you changed (files, behaviour).
2. Which context files you updated and the changelog entry you added.
3. What you verified vs. what you assumed.
4. Anything you left undone, and why.

If tests failed or a step was skipped, say so with the output. Do not report completion for partial work.

---

## Rule 11 — When the rules do not cover it

Prefer the smallest change that satisfies the request. Do not refactor adjacent code, do not "clean up" files you were not asked to touch, do not restructure folders. If you believe a larger change is warranted, describe it and ask.

If a rule here conflicts with an explicit instruction from the user, the user wins — but say which rule you are setting aside and why.
