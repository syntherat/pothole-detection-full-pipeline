"""
Run the vision stage over the 45 picked Level B frames.

Reports THREE numbers per frame, not one, because "no detections" has three
different causes and they need different fixes:
  raw_n / raw_max : pothole model on the full frame at conf=0.05, BEFORE any
                    road filtering -- measures the domain gap alone.
  seg_ok          : did road_seg.pt actually produce a mask, or did we fall
                    through to the lower-60% crop? (CLAUDE.md issue #30)
  final           : what the orchestrator would actually see, conf=0.35.
"""
import sys, csv, json
from pathlib import Path
import numpy as np, cv2

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from integration.vision_adapter import VisionSession

RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "carla_sim/out/levelB_vision"
LOW_CONF = 0.05
FINAL_CONF = 0.35

picks = list(csv.DictReader((RUN / "vision_picks.csv").open()))
print(f"picks: {len(picks)} frames over {len(set(p['event'] for p in picks))} events\n")

sess = VisionSession()
det = sess.detector
print(f"use_road_seg = {det.use_road_seg}")
print(f"pothole model classes = {det.pothole_model.names}")
if det.road_model is not None:
    print(f"road model classes  = {det.road_model.names}")
print()

rows = []
for p in picks:
    fp = RUN / p["path"]
    frame = cv2.imread(str(fp))
    if frame is None:
        print(f"  !! unreadable: {fp}"); continue
    h, w = frame.shape[:2]

    # --- did segmentation find anything at all? ---
    seg_classes, seg_n = [], 0
    if det.road_model is not None:
        r = det.road_model(frame, verbose=False)[0]
        if r.masks is not None and len(r.masks) > 0:
            seg_n = len(r.masks)
            ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
            seg_classes = sorted({r.names[int(i)] for i in ids})

    # --- raw pothole model, no filtering, low threshold ---
    raw = det.pothole_model(frame, conf=LOW_CONF, verbose=False)[0]
    raw_n = 0 if raw.boxes is None else len(raw.boxes)
    raw_max = float(raw.boxes.conf.max()) if raw_n else 0.0

    # --- what the orchestrator sees ---
    out = sess.confirm(frame, conf=FINAL_CONF)

    rows.append(dict(event=int(p["event"]), lead=float(p["lead_s"]), frame=fp.name,
                     seg_n=seg_n, seg_classes=",".join(seg_classes),
                     raw_n=raw_n, raw_max=raw_max,
                     final=out["vision_score"], confirmed=out["vision_confirmed"]))
    print(f"ev{rows[-1]['event']:2d} lead{rows[-1]['lead']:4.1f}s {fp.name}  "
          f"seg={seg_n:2d}[{rows[-1]['seg_classes'][:28]:28s}]  "
          f"raw@{LOW_CONF}: n={raw_n:2d} max={raw_max:.3f}  final={out['vision_score']:.3f}")

print("\n" + "=" * 78)
n = len(rows)
any_raw = sum(1 for r in rows if r["raw_n"] > 0)
any_fin = sum(1 for r in rows if r["confirmed"])
# Count visible_road specifically, NOT "any mask". road_seg.pt emits plenty of
# masks (road_obstacle, roadside_object, ...) that are all excluded, so "any mask"
# reads as if segmentation worked when the road mask is in fact empty.
seg_hit = sum(1 for r in rows if "visible_road" in r["seg_classes"])
seg_any = sum(1 for r in rows if r["seg_n"] > 0)
print(f"frames scored            : {n}")
print(f"seg emitted any mask     : {seg_any}/{n}   (all non-road classes -> excluded)")
print(f"seg predicted visible_road: {seg_hit}/{n}   (0 => road mask empty, lower-60% fallback, issue #30)")
print(f"raw detections @conf=0.05: {any_raw}/{n}   max over all frames = {max((r['raw_max'] for r in rows), default=0):.3f}")
print(f"CONFIRMED  @conf=0.35    : {any_fin}/{n}")
ev = sorted({r["event"] for r in rows})
hit_ev = sorted({r["event"] for r in rows if r["confirmed"]})
print(f"events with >=1 confirm  : {len(hit_ev)}/{len(ev)}  {hit_ev}")

with open(RUN / "vision_results.csv", "w", newline="") as f:
    wri = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
print(f"\nwrote {RUN / 'vision_results.csv'}")
