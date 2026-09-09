"""
carla_sim/label_from_tiles.py

Projects the Level B tiles into every recorded camera frame and writes YOLO
labels. This is the auto-labelling step: `best.pt` confirmed 0/45 on CARLA
frames (2026-09-08), so the frames have to be labelled from geometry we already
know rather than by hand.

    venv/Scripts/python.exe carla_sim/label_from_tiles.py carla_sim/out/<run>
    venv/Scripts/python.exe carla_sim/label_from_tiles.py --self-test

Reads, and why each is the file it is:
  frame_index.csv  the CAMERA's world pose at capture time (cam_* columns).
                   Vehicle pose read at drain time would lag the exposure, and
                   worst on the impact frames that matter most.
  tiles.json       the tiles AS SPAWNED -- real road z, lane yaw, real bowl
                   radius. NOT ground_truth.json, whose `z` is 0.0 and whose
                   `radius_m` is the 0.35 m route marker rather than the cavity.

Writes `labels/<frame>.txt` (YOLO: class cx cy w h, normalised) and
`labels_index.csv` carrying every projection including the ones it rejected and
why. An empty label file is indistinguishable from "no pothole in view", and
those two need different fixes, so the rejects are recorded rather than dropped.

NOT handled: occlusion. A tile behind a wall or a vehicle still projects. On
this route the road ahead is clear; do not assume that on a new one.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import carla  # noqa: E402  -- Transform math only; no server connection is made.

from carla_sim import config  # noqa: E402

CIRCLE_SAMPLES = 48      # points around the bowl rim; the bbox is their extent.
MIN_BOX_PX = 8.0         # smaller than this is not a trainable object.
MAX_RANGE_M = 40.0       # past this a 0.3 m bowl is a couple of pixels.


def intrinsics(width, height, fov_deg):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def project(points_world, world_2_cam, K):
    """
    CARLA is left-handed with X forward, Y right, Z up; the pinhole model wants
    X right, Y down, Z forward. That swap is the (y, -z, x) line below. Get it
    wrong and everything still projects, just mirrored or upside down -- which
    is why --self-test pins three known points rather than trusting the algebra.
    """
    n = points_world.shape[0]
    homo = np.hstack([points_world, np.ones((n, 1))])
    cam = (world_2_cam @ homo.T).T
    cam = np.stack([cam[:, 1], -cam[:, 2], cam[:, 0]], axis=1)
    in_front = cam[:, 2] > 0.01
    img = (K @ cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = img[:, :2] / img[:, 2:3]
    return uv, in_front, cam[:, 2]


def rim_points(tile):
    """The bowl opening, as a circle at the tile's top surface."""
    r = float(tile.get("bowl_radius") or 0.0)
    z_top = float(tile["z"]) + float(tile.get("thickness") or 0.0)
    ang = np.linspace(0.0, 2.0 * math.pi, CIRCLE_SAMPLES, endpoint=False)
    return np.stack([float(tile["x"]) + r * np.cos(ang),
                     float(tile["y"]) + r * np.sin(ang),
                     np.full(CIRCLE_SAMPLES, z_top)], axis=1)


def self_test():
    """
    Three points whose image position is known by inspection, with the camera at
    the origin facing +x. Runs without a recording, a server, or a GPU.
    """
    W, H, FOV = 1280, 720, 90.0
    K = intrinsics(W, H, FOV)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    assert abs(f - 640.0) < 1e-9, f"fov 90 on 1280 px must give f=640, got {f}"

    cam_tf = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))
    w2c = np.array(cam_tf.get_inverse_matrix())

    pts = np.array([[10.0, 0.0, 0.0],    # straight ahead -> image centre
                    [10.0, 1.0, 0.0],    # 1 m to the RIGHT  -> right of centre
                    [10.0, 0.0, 1.0]])   # 1 m UP            -> above centre
    uv, in_front, depth = project(pts, w2c, K)
    assert in_front.all(), "all three points are in front of the camera"
    assert np.allclose(depth, 10.0), f"depth should be 10 m, got {depth}"

    checks = [
        ("centre",     uv[0], (cx, cy)),
        ("1 m right",  uv[1], (cx + f * 0.1, cy)),
        ("1 m up",     uv[2], (cx, cy - f * 0.1)),
    ]
    for name, got, want in checks:
        assert np.allclose(got, want, atol=1e-6), f"{name}: got {got}, want {want}"
        print(f"  ok  {name:10s} -> ({got[0]:7.2f}, {got[1]:7.2f})")

    # A yawed camera: turn 90 deg left (yaw -90 in CARLA), a point on +y is now behind.
    yawed = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0.0, -90.0, 0.0))
    uv2, in_front2, _ = project(np.array([[0.0, 10.0, 0.0]]), np.array(yawed.get_inverse_matrix()), K)
    assert not in_front2[0], "with the camera yawed -90, a point at +y must be behind it"
    print("  ok  yaw handling  -> point correctly rejected as behind camera")
    print("\nself-test passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, nargs="?")
    ap.add_argument("--self-test", action="store_true",
                    help="check the projection math against known points and exit")
    ap.add_argument("--max-range", type=float, default=MAX_RANGE_M)
    ap.add_argument("--min-box-px", type=float, default=MIN_BOX_PX)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.run_dir is None:
        ap.error("run_dir is required unless --self-test is given")

    run_dir = args.run_dir.resolve()
    W, H = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    K = intrinsics(W, H, config.CAMERA_FOV)

    tiles = json.loads((run_dir / "tiles.json").read_text(encoding="utf-8"))
    tile_list = [t for t in tiles["tiles"] if float(t.get("bowl_radius") or 0.0) > 0.0]
    if not tile_list:
        print("No tiles with a bowl -- a flat control run has nothing to label.")
        return 1
    print(f"tiles with a cavity: {len(tile_list)} of {len(tiles['tiles'])}")

    index_path = run_dir / "frame_index.csv"
    rows = list(csv.DictReader(index_path.open(newline="", encoding="utf-8")))
    if not rows or "cam_yaw" not in rows[0]:
        print(f"ERROR: {index_path} has no cam_* pose columns.\n"
              "       It predates the 2026-09-08 pose logging. Re-record -- the pose\n"
              "       cannot be recovered from gnss.csv, which has position but no rotation.")
        return 1

    labels_dir = run_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    rims = [(t, rim_points(t)) for t in tile_list]

    detail, n_labels, n_frames_with = [], 0, 0
    for row in rows:
        cam_tf = carla.Transform(
            carla.Location(x=float(row["cam_x"]), y=float(row["cam_y"]), z=float(row["cam_z"])),
            carla.Rotation(pitch=float(row["cam_pitch"]), yaw=float(row["cam_yaw"]),
                           roll=float(row["cam_roll"])),
        )
        world_2_cam = np.array(cam_tf.get_inverse_matrix())

        lines = []
        for tile, pts in rims:
            uv, in_front, depth = project(pts, world_2_cam, K)
            rec = dict(frame=row["frame"], pothole_id=tile["pothole_id"],
                       dist_m="", x0="", y0="", x1="", y1="", rejected="")

            if not in_front.all():
                rec["rejected"] = "behind_or_straddling_camera"
                detail.append(rec)
                continue

            dist = float(np.median(depth))
            rec["dist_m"] = round(dist, 2)
            if dist > args.max_range:
                rec["rejected"] = f"beyond_{args.max_range:g}m"
                detail.append(rec)
                continue

            x0 = max(float(uv[:, 0].min()), 0.0)
            y0 = max(float(uv[:, 1].min()), 0.0)
            x1 = min(float(uv[:, 0].max()), W - 1.0)
            y1 = min(float(uv[:, 1].max()), H - 1.0)
            if x1 <= x0 or y1 <= y0:
                rec["rejected"] = "offscreen"
                detail.append(rec)
                continue
            if (x1 - x0) < args.min_box_px or (y1 - y0) < args.min_box_px:
                rec["rejected"] = f"under_{args.min_box_px:g}px"
                detail.append(rec)
                continue

            lines.append(f"0 {(x0 + x1) / 2.0 / W:.6f} {(y0 + y1) / 2.0 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            rec.update(x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1))
            detail.append(rec)

        (labels_dir / f"{int(row['frame']):08d}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_labels += len(lines)
        n_frames_with += 1 if lines else 0

    with (run_dir / "labels_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    print(f"frames                : {len(rows)}")
    print(f"frames with >=1 label : {n_frames_with}")
    print(f"labels written        : {n_labels}")
    rejects = {}
    for d in detail:
        if d["rejected"]:
            rejects[d["rejected"]] = rejects.get(d["rejected"], 0) + 1
    for k, v in sorted(rejects.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {k:28s} {v}")
    print(f"\nlabels -> {labels_dir}")
    print(f"detail -> {run_dir / 'labels_index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
