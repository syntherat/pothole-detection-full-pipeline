# Context Index

This directory is the project's **machine-readable knowledge base**. It exists so that an AI agent
(or a new human contributor) can understand PAVE completely without reading every line of source.

**Start with [`00-RULES.md`](00-RULES.md). It is binding.**

---

## Reading order

### Always
| # | File | What it gives you |
|---|---|---|
| 0 | [`00-RULES.md`](00-RULES.md) | **Binding rules.** Boot sequence, boundaries, the mandatory doc-update rule. |
| 1 | [`01-project-overview.md`](01-project-overview.md) | What PAVE is, who built it, the goal, the vocabulary. |
| 2 | [`02-repository-map.md`](02-repository-map.md) | Every file and folder, what it is, whether it exists on a fresh clone. |
| 3 | [`03-architecture-dataflow.md`](03-architecture-dataflow.md) | How the four subsystems connect; the end-to-end cascade. |
| 4 | [`04-current-state.md`](04-current-state.md) | **Where things stand right now** — verification debt, what to do next, open decisions. Mutable; rewrite it as reality changes. |

### Subsystem deep dives — read the one you are touching
| # | File | Covers |
|---|---|---|
| 10 | [`10-physics-sensor-pipeline.md`](10-physics-sensor-pipeline.md) | `pothole_detect_physics/` — IMU state machine, depth/length physics, synthetic dataset, RandomForest. |
| 11 | [`11-vision-pipeline.md`](11-vision-pipeline.md) | `pothole_detection_app/` runtime — YOLO detection, two-stage road masking, both Tkinter GUIs. |
| 12 | [`12-vision-training-scripts.md`](12-vision-training-scripts.md) | `pothole_detection_app/scripts/` — dataset prep, training presets, evaluation, batch prediction. |
| 13 | [`13-map-ui.md`](13-map-ui.md) | `pothole_map_ui/` — Google Maps dashboard, GPS tracking, proximity alerts, event polling. |
| 14 | [`14-integration-layer.md`](14-integration-layer.md) | `integration/` — the cascade orchestrator, adapters, fusion, PAVE connector, tests. |
| 15 | [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md) | **LEVEL A SCAFFOLD, NEVER RUN.** `carla_sim/` — CARLA testbed: potholes with real depth, synced camera + IMU, CARLA road network in the map UI. |

### Cross-cutting reference
| # | File | Covers |
|---|---|---|
| 20 | [`20-data-contracts.md`](20-data-contracts.md) | Every schema that crosses a boundary: dataclasses, JSON, JSONL, CSV columns. |
| 21 | [`21-configuration-and-tuning.md`](21-configuration-and-tuning.md) | Every threshold, constant and tunable in one table, with what happens when you move it. |
| 30 | [`30-setup-and-run.md`](30-setup-and-run.md) | Environments, installs, and every runnable command in dependency order. |
| 40 | [`40-known-issues-and-gaps.md`](40-known-issues-and-gaps.md) | Bugs, mocks, dead paths, missing files. Read before debugging anything. |
| 50 | [`50-changelog.md`](50-changelog.md) | Append-only log. **You write here on every change.** |

---

## Task-to-file routing

| If you are asked to... | Read (beyond the "Always" set) |
|---|---|
| Tune detection sensitivity / reduce false positives | `21`, then `10` (sensor side) or `11` (vision side) |
| Change what a confirmed event contains | `20`, `14`, `13` |
| Make the map show something new | `13`, `20` |
| Retrain or evaluate the YOLO model | `12`, `30`, and **Rule 8** |
| Retrain the sensor classifier / regenerate the CSV | `10`, `30`, and **Rule 8** |
| Wire in real GPS or a real camera | `14`, `40` (the mocks are documented there), `20` |
| Fix "it crashes when I run X" | `40` first, then `30`, then the subsystem file |
| Add a new detection stage | `03`, `14`, `20`, and **Rule 3** |
| Understand accuracy claims | `40` (§ Accuracy claims), `10`, and **Rule 6** |
| Resume after a break / start a fresh session | `04` first — it names the next step and the open decisions |
| Work on the CARLA simulation testbed | `15` first, then `14`, `20`, and **Rule 6** — the Level A scaffold exists but has never been executed |

---

## This directory is not in version control

The root `.gitignore` excludes `context`, `CLAUDE.md` and `AGENTS.md`. **This is deliberate** — confirmed
by the project owner on 2026-08-19. The knowledge base is local to the working machine and does not travel
with a clone.

Consequences to keep in mind:

- `git status` will read **clean** even when you have edited context files. That is expected, not a bug.
- Rule 5 still applies in full. These files are still the project's knowledge base; they are simply kept
  outside the tracked repo.
- Do not `git add -f` them, and do not propose un-ignoring them.

---

## Maintaining this directory

- One file per concern. If a file grows past roughly 400 lines, split it and update this index.
- Numbering: `0x` foundational, `1x` subsystem, `2x` cross-cutting reference, `3x` operational, `4x` problems, `5x` history. Keep new files in that scheme.
- Every file starts with a one-line statement of scope and a "last verified against" note naming the commit or date.
- When you add a context file, add it to the tables above **and** to the routing table.
- Do not duplicate content between files. Link instead. Duplicated facts drift apart.
