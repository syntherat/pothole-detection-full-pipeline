# AGENTS.md

This repository maintains its own context system for AI agents. It applies to **every** agent and
assistant — Claude, Cursor, Copilot, Codex, Gemini, Aider, and any other tool or automated contributor.

## Read these first, in this order

1. **[`context/00-RULES.md`](context/00-RULES.md)** — the binding rules. Not optional.
2. **[`context/README.md`](context/README.md)** — the index; it routes you to the right files for your task.
3. The context file(s) for the subsystem you are about to touch.

[`CLAUDE.md`](CLAUDE.md) at the repo root carries the same entry point with a short orientation section.

## The rule people most often skip

**Rule 5:** when you change code, configuration, data shapes, or behaviour, you must update the
affected files under `context/` and append an entry to
[`context/50-changelog.md`](context/50-changelog.md) **in the same change**.

A code change that leaves the context files stale is treated as an incomplete change, regardless of
whether the code works.

## Project in one line

**PAVE** — hybrid pothole detection: an IMU physics + RandomForest sensor stage gates a YOLO vision
stage, and confirmed detections are surfaced on a live Google Maps driver dashboard.

Four subsystems: `pothole_detect_physics/` (sensor), `pothole_detection_app/` (vision),
`pothole_map_ui/` (dashboard), `integration/` (the cascade that joins them).
