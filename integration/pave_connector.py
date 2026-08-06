"""
Simplest possible bridge to PAVE for a prototype: write confirmed events
to a shared JSON file. Add a small poll loop in app.js that reads this
file and calls addPothole(lat, lng, locationName, detectedBy) for any
new events.

Later, swap this for a real API call (e.g. POST to a Flask endpoint)
without changing anything else in the orchestrator.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

EVENTS_FILE = Path(__file__).resolve().parent / "pave_events.json"


def send_to_pave(event) -> None:
    events = []
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE, "r") as f:
            events = json.load(f)

    events.append({
        "event_id": event.event_id,
        "lat": event.lat,
        "lng": event.lng,
        "confidence": event.final_confidence,
        "detected_by": "hybrid (sensor+vision)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)