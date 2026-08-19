"""
integration/filtered_gui_adapter.py

Bridges the filtered GUI's detection log to the PAVE map.

`pothole_app_filtered.py` already produces everything the map needs while it
processes a video -- an id, GPS, a timestamp, a detection count and, since
2026-08-20, a confidence -- but it writes them in its own shape to

    pothole_detection_app/output/video_detect_<ts>/metadata/events.jsonl

which is incompatible with the `pave_events.json` the dashboard polls. This
module is the translation, and nothing else changes: the GUI keeps its shape,
the map keeps its contract (Rule 3).

    events.jsonl                    pave_events.json
    ------------------------------  ---------------------------------
    id                          ->  event_id
    latitude / longitude        ->  lat / lng
    confidence                  ->  confidence
    timestamp                   ->  created_at
    (constant)                  ->  detected_by = "Image Model"
    potholes_detected               dropped -- not part of contract #7

Usage:
    # one-shot: publish everything in a finished run
    python integration/filtered_gui_adapter.py <video_detect_dir>

    # live: follow the file while the GUI is still writing it
    python integration/filtered_gui_adapter.py <video_detect_dir> --watch

Both shapes are specified in context/20-data-contracts.md (#7 and #9).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pave_connector import send_record_to_pave, EVENTS_FILE  # noqa: E402

logger = logging.getLogger(__name__)

# These events come from the camera alone -- no sensor stage was involved. The
# map's own vocabulary for provenance is 'Image Model' / 'Math Model' / 'Both'
# (see MODELS in pothole_map_ui/app.js), so use its word rather than inventing one.
DETECTED_BY = "Image Model"

DEFAULT_POLL_INTERVAL_S = 2.0


def find_events_file(target: Path) -> Path:
    """
    Accept either a run directory or the jsonl itself, so both of these work:

        .../output/video_detect_20260820_101500
        .../output/video_detect_20260820_101500/metadata/events.jsonl
    """
    target = Path(target)
    if target.is_file():
        return target

    candidate = target / "metadata" / "events.jsonl"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"No events.jsonl under {target}. Expected either the file itself or a "
        "video_detect_<ts> directory containing metadata/events.jsonl. The GUI only "
        "writes it for video runs that produced at least one detection."
    )


def read_events(path: Path) -> list[dict]:
    """
    Parse events.jsonl, one JSON object per line.

    Malformed lines are skipped rather than fatal: in --watch mode we may read
    while the GUI is midway through appending, so a truncated final line is
    expected and will parse fine on the next poll.
    """
    events = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping unparsable line %d (likely a partial write).", lineno)
    return events


def to_pave_record(event: dict) -> dict | None:
    """
    Translate one events.jsonl record into contract #7 shape.

    Returns None if the record cannot be placed on a map -- missing coordinates
    make it useless to the dashboard, and guessing them would be worse than
    dropping it.
    """
    lat = event.get("latitude")
    lng = event.get("longitude")
    event_id = event.get("id")

    if event_id is None or lat is None or lng is None:
        logger.warning("Skipping record without id/latitude/longitude: %s",
                       event.get("id", "<no id>"))
        return None

    # Runs recorded before confidence was added carry none. Pass it through as
    # None rather than fabricating a number (Rule 6); the dashboard renders the
    # label without a confidence in that case.
    confidence = event.get("confidence")

    return {
        "event_id": event_id,
        "lat": float(lat),
        "lng": float(lng),
        "confidence": float(confidence) if confidence is not None else None,
        "detected_by": DETECTED_BY,
        "created_at": event.get("timestamp"),
    }


def publish(events_file: Path, seen: set[str]) -> int:
    """Publish every event not already in `seen`. Returns how many were sent."""
    sent = 0
    for event in read_events(events_file):
        event_id = event.get("id")
        if event_id is None or event_id in seen:
            continue

        record = to_pave_record(event)
        seen.add(event_id)  # mark even on failure, so we do not retry a bad record forever
        if record is None:
            continue

        send_record_to_pave(record)
        sent += 1
        conf = record["confidence"]
        logger.info("published %s  (%.6f, %.6f)  conf=%s",
                    record["event_id"], record["lat"], record["lng"],
                    f"{conf:.2f}" if conf is not None else "n/a")
    return sent


def already_published() -> set[str]:
    """
    event_ids already in pave_events.json.

    Seeding from this makes the adapter idempotent -- re-running it on the same
    run does not append the same events again. The dashboard also dedupes, but
    only per page load, and letting the file accumulate duplicates would be
    sloppy either way.
    """
    if not EVENTS_FILE.exists():
        return set()
    try:
        with EVENTS_FILE.open("r", encoding="utf-8") as f:
            return {e.get("event_id") for e in json.load(f) if e.get("event_id")}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read existing %s (%s) -- may republish events.",
                       EVENTS_FILE.name, e)
        return set()


def sync(target: Path, watch: bool = False,
         interval: float = DEFAULT_POLL_INTERVAL_S) -> int:
    """One-shot publish, or follow the file until interrupted. Returns total sent."""
    events_file = find_events_file(target)
    logger.info("Reading %s", events_file)

    seen: set[str] = already_published()
    if seen:
        logger.info("%d event(s) already in %s -- skipping those.", len(seen), EVENTS_FILE.name)

    total = publish(events_file, seen)
    logger.info("Published %d event(s).", total)

    if not watch:
        return total

    logger.info("Watching for new events every %.1fs -- Ctrl-C to stop.", interval)
    try:
        while True:
            time.sleep(interval)
            new = publish(events_file, seen)
            if new:
                total += new
                logger.info("Published %d new event(s), %d total.", new, total)
    except KeyboardInterrupt:
        logger.info("Stopped. %d event(s) published this session.", total)

    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish filtered-GUI detection events to the PAVE map."
    )
    parser.add_argument("target", type=Path,
                        help="A video_detect_<ts> directory, or an events.jsonl file.")
    parser.add_argument("--watch", action="store_true",
                        help="Keep following the file while the GUI writes to it.")
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL_S,
                        help=f"Poll interval for --watch (default {DEFAULT_POLL_INTERVAL_S}s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    try:
        sync(args.target, watch=args.watch, interval=args.interval)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
