"""
carla_sim/assets/probe_tile_collision.py

Proves -- or disproves -- that an imported pothole tile presents a REAL hole to
CARLA's physics, using the same `world.cast_ray` grid probe that measured the 99
stock props on 2026-09-07.

    venv/Scripts/python.exe carla_sim/assets/probe_tile_collision.py

Requires a running CARLA server (port 2000) whose content includes the
PavePotholes package.

WHY THIS TEST AND NOT A VISUAL CHECK
------------------------------------
A tile can look perfect in the editor and still be a bump. CARLA gives props
convex collision by default, which seals a bowl into a filled dome. That is not
a hypothesis: on 2026-09-07 all 99 stock `static.prop.*` blueprints were probed
this way and NOT ONE had a usable cavity -- every open-topped container (bin,
container, plantpot, box) returned depth 0.00 at a hit fraction of 1.00,
meaning rays landed on a lid that does not visually exist.

So the pass condition here is not "the mesh has a bowl". It is: rays cast down
into the bowl footprint must hit MATERIALLY LOWER than rays cast at the rim,
and the interior must not read as one uniform surface.
"""

import math
import sys

import carla

HOST, PORT = "127.0.0.1", 2000
GRID = 9                 # rays per side across the tile footprint
MIN_DEPTH_M = 0.02       # below this a "bowl" is indistinguishable from noise


def _ground_z(world: "carla.World", loc: "carla.Location") -> float:
    hits = world.cast_ray(carla.Location(loc.x, loc.y, loc.z + 5.0),
                          carla.Location(loc.x, loc.y, loc.z - 20.0))
    return max((h.location.z for h in hits), default=loc.z)


def probe(world, blueprint, spawn_at) -> dict:
    """Spawn one prop and measure its top surface on a ray grid."""
    actor = world.try_spawn_actor(blueprint, carla.Transform(spawn_at, carla.Rotation()))
    if actor is None:
        return {"error": "spawn failed"}
    try:
        actor.set_simulate_physics(False)
    except RuntimeError:
        pass
    world.wait_for_tick()

    try:
        bb = actor.bounding_box
        ex, ey = bb.extent.x, bb.extent.y
        cx, cy = spawn_at.x + bb.location.x, spawn_at.y + bb.location.y
        top = spawn_at.z + bb.extent.z * 2.0 + 1.0
        ground = spawn_at.z

        cells = []
        for i in range(GRID):
            for j in range(GRID):
                gx = cx + (2 * i / (GRID - 1) - 1) * ex * 0.85
                gy = cy + (2 * j / (GRID - 1) - 1) * ey * 0.85
                hits = world.cast_ray(carla.Location(gx, gy, top),
                                      carla.Location(gx, gy, ground - 0.5))
                above = [h.location.z for h in hits if h.location.z > ground + 0.005]
                r = math.hypot(gx - cx, gy - cy)
                cells.append((r, max(above) if above else None))
    finally:
        actor.destroy()

    solid = [z for _, z in cells if z is not None]
    if not solid:
        return {"error": "no hits at all -- prop has no collision"}

    rim = max(solid)
    # Interior = the inner third of the footprint, where a bowl would be.
    inner = [z for r, z in cells if r < min(ex, ey) * 0.45]
    inner_solid = [z for z in inner if z is not None]
    floor = min(inner_solid) if inner_solid else rim
    return {
        "rim_m": rim - spawn_at.z,
        "floor_m": floor - spawn_at.z,
        "depth_m": rim - floor,
        "open_interior": sum(1 for z in inner if z is None),
        "hit_fraction": round(len(solid) / len(cells), 2),
    }


def main() -> int:
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    print(f"map: {world.get_map().name}")

    bl = world.get_blueprint_library()
    tiles = sorted(b.id for b in bl.filter("static.prop.potholetile*"))
    print(f"\n--- registered pothole blueprints: {len(tiles)} ---")
    for t in tiles:
        print("   ", t)
    if not tiles:
        print("\n[FAIL] No PotholeTile blueprints registered. The package JSON path fix "
              "or the import did not take. Nothing to probe.")
        return 1

    sp = world.get_map().get_spawn_points()[0]
    base = carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.5)
    ground = _ground_z(world, base)
    at = carla.Location(base.x, base.y, ground + 0.02)
    print(f"\nground_z = {ground:.3f}; spawning at z = {at.z:.3f}\n")

    print(f"{'blueprint':44s} {'rim':>7s} {'floor':>7s} {'depth':>7s} {'open':>5s} {'hit%':>5s}")
    results = {}
    for tid in tiles:
        r = probe(world, bl.find(tid), at)
        results[tid] = r
        if "error" in r:
            print(f"{tid:44s}  ERROR: {r['error']}")
            continue
        print(f"{tid:44s} {r['rim_m']:7.3f} {r['floor_m']:7.3f} {r['depth_m']:7.3f} "
              f"{r['open_interior']:5d} {r['hit_fraction']:5.2f}")

    # --- verdict ---
    print("")
    ok = True
    for tid, r in results.items():
        if "error" in r:
            ok = False
            continue
        is_control = "flatcontrol" in tid
        if is_control:
            if r["depth_m"] > MIN_DEPTH_M:
                print(f"[FAIL] {tid}: control tile shows a {r['depth_m']:.3f} m depression; "
                      f"it should be flat.")
                ok = False
        elif r["depth_m"] < MIN_DEPTH_M:
            print(f"[FAIL] {tid}: depth {r['depth_m']:.3f} m -- the bowl is SEALED. "
                  f"Collision was hulled, exactly as the 99 stock props were.")
            ok = False

    if ok:
        print("[PASS] Bowl tiles present a real cavity to CARLA's physics; the flat "
              "control reads flat. This is the thing all 99 stock props failed.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
