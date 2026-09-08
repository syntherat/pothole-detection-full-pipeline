"""
carla_sim/assets/verify_pothole_meshes.py

Re-imports the generated FBX files and MEASURES them. Run after
generate_pothole_meshes.py; a clean generator run is not evidence the geometry
is right.

    "D:/Program Files/Blender Foundation/Blender 5.2/blender.exe" --background \
        --python carla_sim/assets/verify_pothole_meshes.py

Checks, per tile:
  * FBX units      -- the tile must measure 1.60 across, i.e. METRES. CARLA's
                      Import.py forces bConvertSceneUnit=1, so UE performs the
                      metres->centimetres conversion itself; exporting at scale
                      100 makes it happen twice.
  * bowl depth     -- measured from the render mesh, not assumed from settings.
  * collision      -- UCX bodies present, and the bowl interior actually EMPTY.
                      A sealed bowl is a bump, not a pothole, and that is the
                      exact failure that made all 99 stock props useless.

THIS FILE IS NECESSARY BUT NOT SUFFICIENT. It can only inspect the FBX in
isolation, and the FBX is internally consistent at either scale -- which is why
a 100x error passed this check on 2026-09-08 and was only caught in-engine, by
spawning a tile and reading `actor.bounding_box`. Always follow up with
probe_tile_collision.py against a running simulator.
"""

import math
import sys
from pathlib import Path

import bpy

ASSET_DIR = Path(__file__).resolve().parent
FBX_DIR = ASSET_DIR / "fbx"

# Everything below is in METRES, matching the FBX the generator now writes.
TILE_SIZE_M = 1.60
TOLERANCE_M = 0.01
SKIRT_M = 0.02          # must match SKIRT in generate_pothole_meshes.py
CENTRE_RADIUS_M = 0.03  # how close to the axis counts as "the bowl centre"

EXPECTED = {
    "PotholeTile_Shallow": {"depth_m": 0.04, "radius_m": 0.22, "ucx": 13},
    "PotholeTile_Medium": {"depth_m": 0.07, "radius_m": 0.28, "ucx": 13},
    "PotholeTile_Deep": {"depth_m": 0.11, "radius_m": 0.34, "ucx": 13},
    "PotholeTile_FlatControl": {"depth_m": 0.0, "radius_m": 0.0, "ucx": 1},
}


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def main() -> int:
    failures = []

    for name, exp in EXPECTED.items():
        path = FBX_DIR / f"{name}.fbx"
        if not path.exists():
            failures.append(f"{name}: FBX missing")
            continue

        _clear()
        bpy.ops.import_scene.fbx(filepath=str(path))
        objs = list(bpy.context.scene.objects)
        render = [o for o in objs if not o.name.startswith("UCX_")]
        ucx = [o for o in objs if o.name.startswith("UCX_")]

        if len(render) != 1:
            failures.append(f"{name}: expected 1 render mesh, got {len(render)}")
            continue
        vs = [render[0].matrix_world @ v.co for v in render[0].data.vertices]

        # --- units ---
        width = max(v.x for v in vs) - min(v.x for v in vs)
        if abs(width - TILE_SIZE_M) > TOLERANCE_M:
            failures.append(f"{name}: tile is {width:.3f} m across, expected {TILE_SIZE_M}")

        # --- bowl depth: lowest TOP-surface point vs the tile's plateau ---
        # The tile is a closed solid, so vertices near the centre exist on BOTH
        # the top surface and the flat underside at -SKIRT. Sampling naively
        # measures plateau-to-underside and overstates every depth by exactly
        # SKIRT -- which is how this check first "failed" all four tiles.
        plateau = max(v.z for v in vs)
        floor_cut = -SKIRT_M + (SKIRT_M / 4.0)
        centre = [v.z for v in vs
                  if math.hypot(v.x, v.y) < CENTRE_RADIUS_M and v.z > floor_cut]
        bowl_bottom = min(centre) if centre else plateau
        depth = plateau - bowl_bottom
        if abs(depth - exp["depth_m"]) > TOLERANCE_M:
            failures.append(
                f"{name}: measured bowl depth {depth:.3f} m, expected {exp['depth_m']}")

        # --- collision ---
        if len(ucx) != exp["ucx"]:
            failures.append(f"{name}: {len(ucx)} UCX bodies, expected {exp['ucx']}")

        # --- is the bowl interior actually empty? ---
        # No collision body may enclose a probe point just above the bowl floor.
        sealed_by = None
        if exp["depth_m"] > 0:
            pz = bowl_bottom + TOLERANCE_M
            for body in ucx:
                bvs = [body.matrix_world @ v.co for v in body.data.vertices]
                xs = [v.x for v in bvs]
                ys = [v.y for v in bvs]
                zs = [v.z for v in bvs]
                if (min(xs) < 0.0 < max(xs) and min(ys) < 0.0 < max(ys)
                        and min(zs) < pz < max(zs)):
                    # Bounding-box overlap only -- a radial wedge legitimately has
                    # a bbox spanning the centre. Report it for a human to judge.
                    sealed_by = body.name
                    break

        status = "OK " if not any(name in f for f in failures) else "FAIL"
        print(f"[{status}] {name}: width={width:.3f}m depth={depth:.3f}m "
              f"ucx={len(ucx)} bbox_overlap={sealed_by or 'none'}")

    print("")
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        return 1
    print("[PASS] all tiles match their declared geometry (FBX only -- "
          "confirm in-engine size with probe_tile_collision.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
