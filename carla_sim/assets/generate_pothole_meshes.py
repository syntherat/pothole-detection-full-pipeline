"""
carla_sim/assets/generate_pothole_meshes.py

Generates the Level B pothole tiles as FBX, ready for CARLA's `make import`.

Runs INSIDE Blender, headless:

    "D:/Program Files/Blender Foundation/Blender 5.2/blender.exe" --background \
        --python carla_sim/assets/generate_pothole_meshes.py

Level B = a real hole in the collision geometry, so the suspension drops on its
own instead of being kicked by a scripted impulse (Level A). See
context/15-carla-testbed-plan.md.

--------------------------------------------------------------------------
THE TWO CONSTRAINTS THAT SHAPE EVERY NUMBER BELOW
--------------------------------------------------------------------------

1. A prop sits ON the road; it cannot cut into it. The road surface stays at
   z=0, so the deepest hole a tile can present is its OWN THICKNESS. Asking for
   a 12 cm pothole means a 12 cm tile, and therefore a 12 cm lip to climb first.
   That lip is the Level B "tile edge" artifact the plan flags, and it is
   exactly why Level C (holes cut into the road mesh) is rated higher fidelity.
   Mitigation here: ramped edges, plus a FLAT CONTROL TILE so the lip's own
   contribution can be measured and subtracted rather than guessed at.

2. CARLA/UE give an imported prop CONVEX collision by default, which would seal
   the bowl into a filled dome -- a bump, not a hole. This was measured on
   2026-09-07: all 99 stock `static.prop.*` blueprints were ray-probed and NOT
   ONE had a usable cavity, every open-topped container reading as solid. So
   these tiles ship explicit `UCX_` convex-decomposition collision: a ring of
   wedges around the rim plus a floor slab, leaving the bowl genuinely empty.
   Do not rely on "Use Complex Collision As Simple" being set after import.

Also deliberate: BOWL_RADIUS is kept >= 0.10 m (0.20 m across) because CARLA's
raycast wheels have zero width and would drop into holes a real tyre bridges.
The plan's instruction is to author wide and DOCUMENT the bias, not tune it out.
"""

import json
import math
import sys
from pathlib import Path

import bpy

# `Path(__file__)` is reliable here: Blender runs the script from its real path.
ASSET_DIR = Path(__file__).resolve().parent
FBX_DIR = ASSET_DIR / "fbx"

# --- package identity (CARLA import contract) -----------------------------
# Props land at /Game/<PACKAGE_NAME>/Static/<tag>/<name>, per Import.py.
PACKAGE_NAME = "PavePotholes"
PROP_TAG = "static"
# CARLA's EPropSize enum (PropParameters.h): Tiny/Small/Medium/Big/Huge.
# "Medium" is documented as "size of a human" -- right for a 1.6 m tile.
# NOT optional: Import.py's generate_package_file() does prop["size"] and dies
# with KeyError otherwise, AFTER the meshes have imported. The meshes look fine
# but no .Package.json is written, so the props never become spawnable
# static.prop.* blueprints -- which is the whole point of importing them.
PROP_SIZE = "Medium"

# --- tile geometry (metres; Blender works in metres, UE in centimetres) ---
TILE_SIZE = 1.60          # square tile edge. Lane is ~3.5 m, so this sits well inside one.
SKIRT = 0.02              # how far the tile body extends BELOW road level, to
                          # bury the knife edge and avoid z-fighting with the road.
RAMP_WIDTH = 0.40         # horizontal run over which the top ramps up from the
                          # road to full thickness. Bigger = gentler lip.
GRID = 64                 # heightfield resolution per side. 64 gives a smooth bowl.

# --- collision decomposition ---------------------------------------------
UCX_SECTORS = 12          # wedges in the ring around the bowl
UCX_FLOOR_SIDES = 12      # polygon sides for the bowl-floor slab

# --- FBX ------------------------------------------------------------------
# MUST be 1.0 for CARLA's import path, i.e. export in METRES and let UE convert.
#
# The plan says "set FBX scale 100", which is correct for a MANUAL UE import
# where you do not convert scene units. It is WRONG here: CARLA's Import.py
# hardcodes "bConvertSceneUnit": 1, so UE applies a metres->centimetres
# conversion of its own. Exporting at 100 makes that conversion happen TWICE and
# the tiles land in-engine at 160 m x 160 m x 14 m instead of 1.6 x 1.6 x 0.14.
#
# Measured 2026-09-08 by spawning a tile and reading actor.bounding_box. Note
# that verifying the FBX alone cannot catch this -- the file is internally
# consistent either way. The size that matters is the one after UE's conversion.
FBX_GLOBAL_SCALE = 1.0

# --- the variants to generate --------------------------------------------
# depth is capped by thickness (constraint 1 above); each entry keeps a
# ~0.01 m floor above road level so the bowl bottom is real geometry, not the
# road showing through.
VARIANTS = [
    # name,                thickness, bowl_depth, bowl_radius
    ("PotholeTile_Shallow", 0.05, 0.04, 0.22),
    ("PotholeTile_Medium",  0.08, 0.07, 0.28),
    ("PotholeTile_Deep",    0.12, 0.11, 0.34),
    # Control: identical footprint and ramp, NO bowl. Drive this to measure what
    # the lip alone contributes to the IMU signature. Without it, every Level B
    # detection is confounded by the edge.
    ("PotholeTile_FlatControl", 0.08, 0.0, 0.0),
]


def _smoothstep(t: float) -> float:
    """Hermite smoothstep, clamped. Used for both the ramp and the bowl wall."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _top_height(x: float, y: float, thickness: float, depth: float, radius: float) -> float:
    """Height of the tile's top surface at (x, y), relative to road level z=0."""
    half = TILE_SIZE / 2.0
    # Distance to the nearest tile edge -> ramp from 0 at the rim to full thickness.
    edge_dist = min(half - abs(x), half - abs(y))
    z = thickness * _smoothstep(edge_dist / RAMP_WIDTH)

    if depth > 0.0 and radius > 0.0:
        r = math.hypot(x, y)
        if r < radius:
            # Raised cosine: 1 at the centre, 0 at the rim, and flat-tangent at
            # BOTH ends. A parabola would meet the road surface at an angle and
            # put a crease at the rim, which reads as a second impact.
            profile = 0.5 * (1.0 + math.cos(math.pi * (r / radius)))
            z -= depth * profile
    return z


def _build_visual_mesh(name: str, thickness: float, depth: float, radius: float):
    """Closed heightfield solid: bowl-carved top, flat bottom at -SKIRT."""
    half = TILE_SIZE / 2.0
    step = TILE_SIZE / (GRID - 1)

    verts: list[tuple[float, float, float]] = []
    for j in range(GRID):
        for i in range(GRID):
            x = -half + i * step
            y = -half + j * step
            verts.append((x, y, _top_height(x, y, thickness, depth, radius)))

    faces: list[tuple[int, ...]] = []
    for j in range(GRID - 1):
        for i in range(GRID - 1):
            a = j * GRID + i
            faces.append((a, a + 1, a + GRID + 1, a + GRID))

    # Bottom: mirror the top grid at z=-SKIRT so the walls are quads, not a fan.
    base = len(verts)
    for j in range(GRID):
        for i in range(GRID):
            x = -half + i * step
            y = -half + j * step
            verts.append((x, y, -SKIRT))
    for j in range(GRID - 1):
        for i in range(GRID - 1):
            a = base + j * GRID + i
            faces.append((a, a + GRID, a + GRID + 1, a + 1))  # reversed winding

    # Side walls, stitching the top boundary ring to the bottom boundary ring.
    def idx(i: int, j: int) -> int:
        return j * GRID + i

    for i in range(GRID - 1):
        faces.append((idx(i, 0), idx(i + 1, 0), base + idx(i + 1, 0), base + idx(i, 0)))
        faces.append((idx(i + 1, GRID - 1), idx(i, GRID - 1),
                      base + idx(i, GRID - 1), base + idx(i + 1, GRID - 1)))
    for j in range(GRID - 1):
        faces.append((idx(0, j + 1), idx(0, j), base + idx(0, j), base + idx(0, j + 1)))
        faces.append((idx(GRID - 1, j), idx(GRID - 1, j + 1),
                      base + idx(GRID - 1, j + 1), base + idx(GRID - 1, j)))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _add_convex(name: str, points: list[tuple[float, float, float]]):
    """Create a UCX collision body from a point set.

    UE reads any mesh named UCX_<render mesh name>_NN in the same FBX as a
    convex collision primitive. Each body here is convex by construction.
    """
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Let Blender build the hull faces from the point cloud.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.convex_hull()
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def _build_collision(base_name: str, thickness: float, depth: float, radius: float):
    """Convex decomposition: a ring of wedges around the bowl + a floor slab.

    The bowl interior is deliberately left EMPTY so a wheel can fall into it.
    A single convex hull of the whole tile would seal it shut.
    """
    bodies = []
    outer = TILE_SIZE / 2.0

    if depth <= 0.0 or radius <= 0.0:
        # Flat control: one box is exact and convex.
        pts = [(sx * outer, sy * outer, z)
               for sx in (-1, 1) for sy in (-1, 1) for z in (-SKIRT, thickness)]
        bodies.append(_add_convex(f"UCX_{base_name}_01", pts))
        return bodies

    floor_z = thickness - depth
    for s in range(UCX_SECTORS):
        a0 = 2.0 * math.pi * s / UCX_SECTORS
        a1 = 2.0 * math.pi * (s + 1) / UCX_SECTORS
        pts = []
        for a in (a0, a1):
            ca, sa = math.cos(a), math.sin(a)
            # Inner face slants outward as it rises, approximating the bowl wall.
            pts.append((radius * 0.55 * ca, radius * 0.55 * sa, floor_z))
            pts.append((radius * ca, radius * sa, thickness))
            pts.append((radius * 0.55 * ca, radius * 0.55 * sa, -SKIRT))
            pts.append((outer * ca, outer * sa, thickness))
            pts.append((outer * ca, outer * sa, -SKIRT))
        bodies.append(_add_convex(f"UCX_{base_name}_{s + 1:02d}", pts))

    # Floor slab: what the wheel actually lands on at the bottom of the bowl.
    floor_pts = []
    for k in range(UCX_FLOOR_SIDES):
        a = 2.0 * math.pi * k / UCX_FLOOR_SIDES
        ca, sa = math.cos(a), math.sin(a)
        floor_pts.append((radius * 0.60 * ca, radius * 0.60 * sa, floor_z))
        floor_pts.append((radius * 0.60 * ca, radius * 0.60 * sa, -SKIRT))
    bodies.append(_add_convex(f"UCX_{base_name}_{UCX_SECTORS + 1:02d}", floor_pts))
    return bodies


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.objects):
        for item in list(block):
            if getattr(item, "users", 0) == 0:
                block.remove(item)


def _export(objects, path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        use_selection=True,
        global_scale=FBX_GLOBAL_SCALE,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        object_types={"MESH"},
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        bake_anim=False,
    )


def main() -> int:
    FBX_DIR.mkdir(parents=True, exist_ok=True)
    props = []

    for name, thickness, depth, radius in VARIANTS:
        if depth >= thickness:
            print(f"[FATAL] {name}: bowl_depth {depth} >= thickness {thickness}; "
                  f"the bowl would punch through the tile bottom.")
            return 1
        if depth > 0.0 and radius < 0.10:
            print(f"[FATAL] {name}: bowl_radius {radius} m is under the 0.10 m "
                  f"minimum -- CARLA's zero-width raycast wheels need a wide hole.")
            return 1

        _clear_scene()
        visual = _build_visual_mesh(name, thickness, depth, radius)
        collision = _build_collision(name, thickness, depth, radius)

        out = FBX_DIR / f"{name}.fbx"
        _export([visual] + collision, out)
        print(f"[OK] {name}: thickness={thickness} depth={depth} radius={radius} "
              f"collision_bodies={len(collision)} -> {out.name}")

        props.append({
            "name": name,
            "source": f"./{name}.fbx",
            "size": PROP_SIZE,
            "tag": PROP_TAG,
        })

    # CARLA's Import.py reads this next to the FBX files.
    manifest = FBX_DIR / f"{PACKAGE_NAME}.json"
    manifest.write_text(json.dumps({"maps": [], "props": props}, indent=2), encoding="utf-8")
    print(f"[OK] manifest -> {manifest.name} ({len(props)} props)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
