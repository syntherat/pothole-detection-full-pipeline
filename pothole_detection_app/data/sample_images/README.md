# sample_images

**Put road photos here.** Anything in this folder (except this README) is gitignored.

## What this is for

The integration cascade runs the sensor stage over a CSV of IMU rows, and when a row looks like a
pothole it needs a camera frame to confirm against. There is no real synced camera data yet — see
issue #18 in the context notes — so `integration/frame_provider.py` substitutes a stand-in image from
this folder.

This is a **deliberate mock**, not a missing feature. It exists so the pipeline can be exercised
end to end before real vehicle data exists.

## What to put here

A handful is enough — 5 to 20 images.

- `.jpg`, `.jpeg` or `.png`, directly in this folder (not in subfolders)
- Road-facing photos, roughly what a dash camera would see
- Include some with visible potholes and some without, so the vision stage actually
  discriminates rather than confirming everything

Good sources: frames grabbed from a dash-cam video, your YOLO validation split, or any
public pothole dataset.

## You may not need to do anything

`frame_provider.py` searches several locations and uses the first that contains images:

```
pothole_detection_app/data/sample_images/          <- here, preferred
pothole_detection_app/input/
pothole_detection_app/data/dataset_v3/test/images/
pothole_detection_app/data/dataset_v3/val/images/
pothole_detection_app/data/dataset_v2/test/images/
pothole_detection_app/data/dataset_v2/val/images/
```

So if you have already built a training dataset, the cascade will find images without any setup.
Dropping files here overrides that.

## If none are found

The orchestrator no longer crashes. It logs one warning, skips every triggered event, and reports the
count at the end — so a run still completes and tells you exactly what was missing.

## What the images actually affect

Selection is deterministic: the same `event_id` always maps to the same image, via an MD5 of the id
modulo the pool size. Reruns are reproducible.

But be clear about what this does and does not prove. The chosen image has **no real relationship** to
the sensor event that triggered it — that link is exactly what is missing. So `vision_score`,
`final_confidence` and any end-to-end accuracy figure derived from a mock run are **not meaningful**.
What a mock run does prove is that the plumbing works: the cascade triggers, the vision stage runs,
fusion decides, and confirmed events reach the map.

For real numbers you need time-synchronised frames — either from a vehicle, or from the CARLA testbed
(`carla_sim/`), whose whole point is producing exactly that.
