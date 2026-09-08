"""
carla_sim/assets/fix_package_paths.py

Repairs the asset paths in CARLA's generated `<Package>.Package.json` so they
point at the assets that were actually imported.

    python carla_sim/assets/fix_package_paths.py [--carla-root D:/dev/carla-source]

WHY THIS IS NEEDED
------------------
`Util/BuildTools/Import.py` derives each prop's object path from the *FBX file
name*, assuming the imported asset is named the same:

    path = /Game/<pkg>/Static/<tag>/<name>/<fbx_basename>.<fbx_basename>

That holds for a single-mesh FBX. Our tiles are not single-mesh: each one ships
the render mesh plus 13 `UCX_` convex collision bodies, and UE prefixes the
asset with the FBX file name when an FBX has multiple root nodes. So the asset
lands as

    PotholeTile_Shallow_PotholeTile_Shallow.uasset

while Import.py registers `PotholeTile_Shallow.PotholeTile_Shallow`. The
package then references an asset that does not exist, and the prop silently
fails to spawn -- with no error at import time, because the import itself
succeeded.

Dropping the UCX bodies to get a clean name is NOT an option: CARLA gives props
convex collision by default, which seals the bowl into a dome. That was measured
on 2026-09-07 -- 0 of 99 stock props had a usable cavity. The UCX bodies are the
whole reason these tiles are holes rather than bumps.

Re-run this after every `make import`; Import.py rewrites the file each time.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CARLA_ROOT = Path("D:/dev/carla-source")
PACKAGE_NAME = "PavePotholes"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carla-root", type=Path, default=DEFAULT_CARLA_ROOT)
    ap.add_argument("--package", default=PACKAGE_NAME)
    args = ap.parse_args()

    content = args.carla_root / "Unreal" / "CarlaUE4" / "Content" / args.package
    pkg_file = content / "Config" / f"{args.package}.Package.json"
    if not pkg_file.is_file():
        print(f"[FATAL] {pkg_file} not found -- run the import first.")
        return 1

    data = json.loads(pkg_file.read_text(encoding="utf-8"))
    changed, missing = 0, []

    for prop in data.get("props", []):
        # The folder Import.py imported this prop into.
        folder = content / "Static" / "static" / prop["name"]
        assets = sorted(folder.glob("*.uasset")) if folder.is_dir() else []
        if len(assets) != 1:
            missing.append(f"{prop['name']}: expected 1 .uasset in {folder}, found {len(assets)}")
            continue

        stem = assets[0].stem                       # e.g. PotholeTile_Shallow_PotholeTile_Shallow
        real = f"/Game/{args.package}/Static/static/{prop['name']}/{stem}.{stem}"
        if prop.get("path") != real:
            print(f"  {prop['name']}")
            print(f"    was: {prop.get('path')}")
            print(f"    now: {real}")
            prop["path"] = real
            changed += 1

    if missing:
        for m in missing:
            print(f"[FATAL] {m}")
        return 1

    if changed:
        pkg_file.write_text(json.dumps(data, indent=4), encoding="utf-8")
        print(f"[OK] rewrote {changed} path(s) in {pkg_file.name}")
    else:
        print(f"[OK] all paths already correct in {pkg_file.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
