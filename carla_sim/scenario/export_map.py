"""
carla_sim/scenario/export_map.py

Exports the CARLA town's road network as GeoJSON so the map dashboard can draw
it instead of Google's base tiles.

WHY THIS EXISTS
---------------
CARLA towns are geo-referenced near (0, 0) -- the Gulf of Guinea. Feeding CARLA
GNSS straight into the dashboard puts every pothole marker in the Atlantic, on
top of an empty ocean tile. The two ways out were to anchor the town over a real
city (markers land in buildings, because CARLA's streets are not that city's
streets) or to draw CARLA's own roads and hide the base tiles. This is the
second: `context/15-carla-testbed-plan.md` calls it "CARLA mode".

Coordinates stay in real lat/lng throughout -- `transform_to_geolocation()` does
the conversion -- so contract #7 (`PotholeGuard.reportDetection`) and the events
file are untouched. Only the basemap changes. That is Rule 3.

STATUS: written against the documented CARLA 0.9.x API but NEVER RUN against a
simulator. Treat every runtime claim here as unverified until it is.

Usage:
    python carla_sim/scenario/export_map.py
    python carla_sim/scenario/export_map.py --town Town05 --step 4.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import carla
except ImportError:
    sys.exit(
        "carla module not found.\n"
        "  pip install -r carla_sim/requirements.txt\n"
        "and make sure the CARLA simulator itself is running."
    )

import config  # noqa: E402

logger = logging.getLogger(__name__)

# A topology segment is one road between two junctions. Walking it in `step_m`
# increments traces the lane centreline. Too fine and the GeoJSON balloons for
# no visual gain at dashboard zoom; too coarse and curves become polygons.
DEFAULT_STEP_M = 2.0

# Guard against a malformed lane graph looping forever. At 2 m steps this caps a
# single segment at 10 km, which is far longer than any road in any CARLA town.
MAX_STEPS_PER_SEGMENT = 5000


def _walk_segment(start, end, step_m: float) -> list:
    """
    Waypoints along one topology segment, from `start` to `end`.

    `next()` returns several waypoints at a junction. Preferring the candidate
    that shares road_id and lane_id keeps us on the segment we were asked to
    trace instead of wandering into a branch, which would draw a road that does
    not exist.
    """
    points = [start]
    current = start
    end_loc = end.transform.location

    for _ in range(MAX_STEPS_PER_SEGMENT):
        if current.transform.location.distance(end_loc) <= step_m:
            break

        candidates = current.next(step_m)
        if not candidates:
            break

        same_lane = [
            wp for wp in candidates
            if wp.road_id == current.road_id and wp.lane_id == current.lane_id
        ]
        current = same_lane[0] if same_lane else candidates[0]
        points.append(current)

    points.append(end)
    return points


def _to_geojson_line(carla_map, waypoints) -> list[list[float]]:
    """
    GeoJSON coordinate list for `waypoints`. GeoJSON is [lng, lat], NOT [lat, lng]
    -- getting this backwards renders the whole town rotated into the wrong
    hemisphere, and it looks plausible enough at a glance to waste an afternoon.
    """
    coords: list[list[float]] = []
    for wp in waypoints:
        geo = carla_map.transform_to_geolocation(wp.transform.location)
        coords.append([geo.longitude, geo.latitude])
    return coords


def export(world, out_path: Path, step_m: float) -> dict:
    carla_map = world.get_map()
    topology = carla_map.get_topology()
    logger.info("Town %s: %d topology segments", carla_map.name, len(topology))

    features = []
    lat_min = lng_min = float("inf")
    lat_max = lng_max = float("-inf")

    for start, end in topology:
        waypoints = _walk_segment(start, end, step_m)
        coords = _to_geojson_line(carla_map, waypoints)
        if len(coords) < 2:
            continue

        for lng, lat in coords:
            lat_min, lat_max = min(lat_min, lat), max(lat_max, lat)
            lng_min, lng_max = min(lng_min, lng), max(lng_max, lng)

        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "road_id": start.road_id,
                "lane_id": start.lane_id,
                "is_junction": bool(start.is_junction),
            },
        })

    if not features:
        raise RuntimeError(
            "No road segments exported. get_topology() returned nothing usable -- "
            "is a town actually loaded?"
        )

    # The dashboard fits its viewport to these rather than guessing a zoom, so
    # they ship inside the file rather than being recomputed in JS.
    collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "town": carla_map.name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step_m": step_m,
            "segment_count": len(features),
            "bounds": {
                "lat_min": lat_min, "lat_max": lat_max,
                "lng_min": lng_min, "lng_max": lng_max,
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collection), encoding="utf-8")
    return collection["properties"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", default=None,
                        help=f"Town to load first. Default: use whatever is loaded (config.TOWN is {config.TOWN})")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_M,
                        help=f"Metres between traced points (default {DEFAULT_STEP_M})")
    parser.add_argument("--out", default=None,
                        help="Output path (default: carla_sim/out/<town>_roads.geojson)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
    client.set_timeout(config.CARLA_TIMEOUT_S)

    if args.town:
        logger.info("Loading %s ...", args.town)
        world = client.load_world(args.town)
    else:
        world = client.get_world()

    town_name = world.get_map().name.split("/")[-1]
    out_path = Path(args.out) if args.out else config.OUT_DIR / f"{town_name}_roads.geojson"

    meta = export(world, out_path, args.step)

    print("\n" + "=" * 68)
    print("ROAD NETWORK EXPORTED")
    print("=" * 68)
    print(f"  town      : {meta['town']}")
    print(f"  segments  : {meta['segment_count']}")
    print(f"  bounds    : lat {meta['bounds']['lat_min']:.6f} .. {meta['bounds']['lat_max']:.6f}")
    print(f"              lng {meta['bounds']['lng_min']:.6f} .. {meta['bounds']['lng_max']:.6f}")
    print(f"  written   : {out_path}")
    print(f"  size      : {out_path.stat().st_size / 1024:.0f} KB")
    print("\nOpen the dashboard in CARLA mode:")
    print(f"  http://localhost:5500/pothole_map_ui/index.html?mode=carla&town={town_name}")
    print("=" * 68)


if __name__ == "__main__":
    main()
