"""
carla_sim/scenario/tiles.py

Level B pothole placement: spawns real geometry on the road instead of scripting
a jerk at a coordinate.

Level A (scenario/impulse.py) fakes the cause and lets the suspension produce the
effect -- see context/15-carla-testbed-plan.md. Level B removes the script
entirely: a tile with a genuine cavity sits on the road, the wheel raycast finds
no surface where it expects one, and the drop, freefall and impact are ALL
physics-derived.

The tiles are authored by carla_sim/assets/generate_pothole_meshes.py and were
verified in-engine on 2026-09-08 -- ray-probed to their authored depths of
0.110 / 0.070 / 0.040 m, with the flat control at 0.000. See issue #41 for the
FBX scale trap that verification caught.

TWO THINGS TO KEEP IN MIND WHEN READING RESULTS FROM THIS MODULE

1. A prop sits ON the road; it cannot cut into it. So each tile is a hole
   *behind a ramp of the same height* -- the Deep tile is an 11 cm cavity
   entered over a 12 cm lip. `PotholeTile_FlatControl` exists precisely to
   measure what that lip contributes on its own. Run it. Any detection rate
   quoted without the control is confounded by the tile's own edge.
2. CARLA's raycast wheels have zero width, so they engage holes a real tyre
   would partly bridge. The plan's instruction is to document that bias rather
   than tune it out.
"""

import logging

import carla

logger = logging.getLogger(__name__)

# Blueprint ids CARLA derives from the imported prop names (lower-cased).
TILE_SHALLOW = "static.prop.potholetile_shallow"
TILE_MEDIUM = "static.prop.potholetile_medium"
TILE_DEEP = "static.prop.potholetile_deep"
TILE_FLAT_CONTROL = "static.prop.potholetile_flatcontrol"

# Bowl geometry per variant, for anything that needs the tile's real opening
# rather than Pothole.radius_m (which is the ROUTE pothole's 0.35 m marker, not
# the cavity -- projecting labels from it would size every box wrong).
# SOURCE OF TRUTH is VARIANTS in carla_sim/assets/generate_pothole_meshes.py.
# Mirrored here because that script lives outside any package and is not
# importable; if you change the meshes, change this too.
TILE_GEOMETRY = {
    TILE_SHALLOW:      {"bowl_radius": 0.22, "bowl_depth": 0.04, "thickness": 0.05},
    TILE_MEDIUM:       {"bowl_radius": 0.28, "bowl_depth": 0.07, "thickness": 0.08},
    TILE_DEEP:         {"bowl_radius": 0.34, "bowl_depth": 0.11, "thickness": 0.12},
    TILE_FLAT_CONTROL: {"bowl_radius": 0.00, "bowl_depth": 0.00, "thickness": 0.08},
}
TILE_SIZE = 1.60  # square tile edge, same source of truth as above.

# Severity is a 0-1 scalar on Pothole, not a depth. Map it onto the three real
# depths so a run contains a spread of event strengths, as Level A did.
SEVERITY_TO_TILE = (
    (0.55, TILE_SHALLOW),   # severity <  0.55
    (0.80, TILE_MEDIUM),    # severity <  0.80
    (2.00, TILE_DEEP),      # otherwise
)


def tile_for_severity(severity: float) -> str:
    for threshold, blueprint_id in SEVERITY_TO_TILE:
        if severity < threshold:
            return blueprint_id
    return TILE_DEEP


def spawn_tiles(world: "carla.World", potholes, control: bool = False,
                manifest_out: list | None = None) -> list:
    """
    Spawn one tile per pothole, aligned to the lane it sits in.

    `control=True` swaps every tile for the flat control, which is how you
    measure the lip's own contribution: same run, same route, same speeds, no
    cavities. Diff the two recordings.

    Returns the spawned actors; the caller owns their destruction.
    """
    library = world.get_blueprint_library()
    carla_map = world.get_map()
    spawned = []
    failures = 0

    for pothole in potholes:
        blueprint_id = TILE_FLAT_CONTROL if control else tile_for_severity(pothole.severity)
        location = pothole.location()

        # Take yaw from the lane so the tile's ramp faces along the direction of
        # travel. A tile rotated across the lane presents its edge as a kerb.
        waypoint = carla_map.get_waypoint(location, project_to_road=True)
        yaw = waypoint.transform.rotation.yaw if waypoint is not None else 0.0

        # The mesh origin is at road level: the body extends 2 cm below (buried,
        # so the knife edge does not z-fight with the road) and `thickness`
        # above. Spawning at the road surface is therefore correct as-is.
        z = waypoint.transform.location.z if waypoint is not None else pothole.z
        transform = carla.Transform(
            carla.Location(x=location.x, y=location.y, z=z),
            carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

        actor = world.try_spawn_actor(library.find(blueprint_id), transform)
        if actor is None:
            failures += 1
            continue
        # Static scenery: without this the tile is a rigid body the car can shove.
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        spawned.append(actor)

        # Record the transform we ACTUALLY spawned at, not the pothole's nominal
        # position: z comes from the lane waypoint (road height, never 0.0 as in
        # ground_truth.json) and yaw from the lane. Anything projecting these
        # tiles into camera space needs the real values.
        if manifest_out is not None:
            geom = TILE_GEOMETRY.get(blueprint_id, {})
            manifest_out.append({
                "pothole_id": getattr(pothole, "pothole_id", None),
                "blueprint": blueprint_id,
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll,
                "tile_size": TILE_SIZE,
                **geom,
            })

    # Advance one tick so the spawns register. MUST branch on the mode: the
    # recorder drives the world SYNCHRONOUSLY, and in that mode wait_for_tick()
    # waits for a tick nobody will ever produce -- it blocks until the client
    # times out. Only world.tick() advances a synchronous server.
    #
    # Anything that raises from here on must not leak the actors already
    # spawned: the caller has no reference to them yet, so they would be
    # orphaned in the world and silently contaminate the NEXT run. (That is not
    # hypothetical -- it happened on 2026-09-08, leaving 4 stray tiles behind.)
    try:
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick()
    except Exception:
        destroy_tiles(spawned)
        raise

    if failures:
        logger.warning("%d/%d tiles failed to spawn (collision with existing "
                       "geometry is the usual cause).", failures, len(potholes))
    logger.info("Spawned %d Level B tiles%s.", len(spawned),
                " (FLAT CONTROL)" if control else "")
    return spawned


def destroy_tiles(actors) -> int:
    destroyed = 0
    for actor in actors:
        try:
            actor.destroy()
            destroyed += 1
        except RuntimeError:
            pass
    return destroyed
