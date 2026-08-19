"""
carla_sim/scenario/potholes.py

The pothole registry: where they are, when a wheel is inside one, and the
ground-truth label window.

Level A has no geometry -- a pothole is a coordinate, a radius and a severity.
Level B replaces `place_along_route` with real spawned prop meshes; everything
else in this module stays as it is, which is the point of keeping it separate.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import carla

logger = logging.getLogger(__name__)


@dataclass
class Pothole:
    """One pothole. `severity` 0-1 scales the impulse; it is not a depth in metres."""

    pothole_id: str
    x: float
    y: float
    z: float
    radius_m: float
    severity: float
    route_distance_m: float

    def location(self) -> carla.Location:
        return carla.Location(x=self.x, y=self.y, z=self.z)

    def contains(self, point: carla.Location) -> bool:
        """2D test -- z varies with road grade and would only add false negatives."""
        dx = point.x - self.x
        dy = point.y - self.y
        return (dx * dx + dy * dy) <= (self.radius_m * self.radius_m)


def place_along_route(route: list[carla.Waypoint], num: int, spacing_m: float,
                      radius_m: float, lateral_offset_m: float,
                      severity_range: tuple[float, float],
                      seed: int) -> list[Pothole]:
    """
    Place potholes on the route the vehicle will actually drive.

    Offset laterally from the lane centre so one side of the car hits at a time
    -- a both-wheels strike produces a symmetric jolt with almost no roll, which
    is both unrealistic and a much easier signal than the real thing.

    Sides alternate so the run contains left-wheel and right-wheel events.
    """
    rng = random.Random(seed)
    potholes: list[Pothole] = []

    # Route waypoints are 2 m apart (see route.build_route).
    step = max(1, int(spacing_m / 2.0))
    start = step  # leave room to reach target speed before the first event

    for i in range(num):
        index = start + i * step
        if index >= len(route):
            logger.warning("Route exhausted after %d potholes (wanted %d).", i, num)
            break

        waypoint = route[index]
        transform = waypoint.transform
        right = transform.get_right_vector()
        side = 1.0 if i % 2 == 0 else -1.0

        centre = transform.location
        potholes.append(Pothole(
            pothole_id=f"PH-SIM-{i:04d}",
            x=centre.x + right.x * lateral_offset_m * side,
            y=centre.y + right.y * lateral_offset_m * side,
            z=centre.z,
            radius_m=radius_m,
            severity=rng.uniform(*severity_range),
            route_distance_m=index * 2.0,
        ))

    logger.info("Placed %d potholes along the route.", len(potholes))
    return potholes


class PotholeTracker:
    """
    Per-tick wheel-over detection and ground-truth labelling.

    Tracks which potholes are currently 'occupied' so that a single crossing
    fires one impulse rather than one per tick while the wheel is inside.
    """

    def __init__(self, potholes: list[Pothole], label_window_s: float):
        self.potholes = potholes
        self.label_window_s = label_window_s
        self._active: set[str] = set()
        self._label_until: float = -1.0
        self.hits: list[dict] = []

    def update(self, sim_time: float, wheel_positions: list[carla.Location]) -> list[tuple[Pothole, carla.Location]]:
        """
        Returns the (pothole, wheel_position) pairs that were ENTERED this tick.

        Only entries are returned -- continuing to sit inside a pothole is not a
        new event, and re-firing the impulse every tick would launch the car.
        """
        entered: list[tuple[Pothole, carla.Location]] = []
        occupied: set[str] = set()

        for pothole in self.potholes:
            for wheel in wheel_positions:
                if not pothole.contains(wheel):
                    continue
                occupied.add(pothole.pothole_id)
                if pothole.pothole_id not in self._active:
                    entered.append((pothole, wheel))
                    self._label_until = sim_time + self.label_window_s
                    self.hits.append({
                        "pothole_id": pothole.pothole_id,
                        "sim_time": sim_time,
                        "severity": pothole.severity,
                        "wheel_x": wheel.x,
                        "wheel_y": wheel.y,
                    })
                    logger.info("[%8.3fs] HIT %s  severity=%.2f",
                                sim_time, pothole.pothole_id, pothole.severity)
                break  # one wheel per pothole per tick is enough

        self._active = occupied
        return entered

    def label(self, sim_time: float) -> int:
        """The `label` column of contract #1: 1 inside a hit window, else 0."""
        return 1 if sim_time <= self._label_until else 0


def write_ground_truth(path: Path, potholes: list[Pothole],
                       tracker: PotholeTracker, town: str, seed: int) -> None:
    """
    Ground truth for the run. This is what makes CARLA data worth more than the
    synthetic CSV: the labels are derived from known geometry, not from the same
    generator that produced the signal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "town": town,
        "seed": seed,
        "potholes": [asdict(p) for p in potholes],
        "hits": tracker.hits,
        "hit_count": len(tracker.hits),
        "placed_count": len(potholes),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    missed = len(potholes) - len({h["pothole_id"] for h in tracker.hits})
    logger.info("Ground truth: %d/%d potholes actually driven over (%d missed).",
                len(potholes) - missed, len(potholes), missed)
