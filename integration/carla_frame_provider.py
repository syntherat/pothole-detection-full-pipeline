"""
integration/carla_frame_provider.py

The REAL frame and GPS provider -- the thing frame_provider.py has been faking.

This is what closes issue #18. Instead of picking an arbitrary stand-in image
and inventing a GPS track, it looks up:

  * the camera frame actually captured closest in time to a sensor sample
  * the GNSS fix actually recorded at that moment

...from a run recorded by carla_sim/scenario/drive_and_record.py.

Reads two files from a run directory:

    frame_index.csv   frame,timestamp,path
    gnss.csv          timestamp,frame,latitude,longitude,altitude

Both use the same clock as sensors.csv (seconds from the start of recording),
which is the entire point -- CARLA's synchronous mode gives every sensor the
same timebase.

Drop-in for the orchestrator: exposes get_frame(timestamp, event_id) and
get_gps(timestamp), matching MockFrameProvider in frame_provider.py.

Self-check:
    python integration/carla_frame_provider.py <run_dir>
"""

from __future__ import annotations

import bisect
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Camera runs at 20 Hz in the recorder, so consecutive frames are 50 ms apart and
# the worst-case nearest-frame error is 25 ms. This tolerance is deliberately a
# little wider than that, and deliberately NOT generous: pairing a sensor event
# with a frame from half a second later would silently reintroduce exactly the
# desynchronisation this class exists to remove.
DEFAULT_FRAME_TOLERANCE_S = 0.06

# GNSS is written every tick, so an exact match is normal. The tolerance only
# matters at the very start and end of a run.
DEFAULT_GPS_TOLERANCE_S = 0.05


class CarlaFrameProvider:
    """
    Timestamp-keyed lookup over one recorded CARLA run.

    Unlike the mock, which keys frames on event_id (arbitrary, and reproducible
    only within a process because str hashing is salted), this keys on TIME --
    which is the only thing that makes a frame and a sensor row belong together.
    """

    def __init__(self, run_dir: Path | str,
                 frame_tolerance_s: float = DEFAULT_FRAME_TOLERANCE_S,
                 gps_tolerance_s: float = DEFAULT_GPS_TOLERANCE_S):
        self.run_dir = Path(run_dir)
        self.frame_tolerance = frame_tolerance_s
        self.gps_tolerance = gps_tolerance_s

        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self._frame_times: list[float] = []
        self._frame_paths: list[Path] = []
        self._gps_times: list[float] = []
        self._gps_points: list[tuple[float, float]] = []

        self._load_frames()
        self._load_gnss()

        # Counters, so a run can be audited afterwards rather than silently
        # dropping events.
        self.misses_no_frame = 0
        self.misses_no_gps = 0

    # ---------------------------------------------------------------- loading

    def _load_frames(self) -> None:
        path = self.run_dir / "frame_index.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Record a run with carla_sim/scenario/drive_and_record.py "
                "(without --no-camera, or there will be no frames to look up)."
            )

        missing = 0
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frame_path = self.run_dir / row["path"]
                if not frame_path.exists():
                    missing += 1
                    continue
                self._frame_times.append(float(row["timestamp"]))
                self._frame_paths.append(frame_path)

        if missing:
            logger.warning("%d frames listed in frame_index.csv are missing on disk.", missing)

        if not self._frame_times:
            raise ValueError(
                f"No usable frames in {path}. The run was probably recorded with --no-camera."
            )

        self._sort_frames()
        logger.info("Loaded %d frames spanning %.2fs - %.2fs",
                    len(self._frame_times), self._frame_times[0], self._frame_times[-1])

    def _sort_frames(self) -> None:
        """bisect requires sorted input; do not assume the CSV is ordered."""
        pairs = sorted(zip(self._frame_times, self._frame_paths), key=lambda p: p[0])
        self._frame_times = [t for t, _ in pairs]
        self._frame_paths = [p for _, p in pairs]

    def _load_gnss(self) -> None:
        path = self.run_dir / "gnss.csv"
        if not path.exists():
            logger.warning("%s not found -- get_gps() will always return None.", path)
            return

        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._gps_times.append(float(row["timestamp"]))
                self._gps_points.append((float(row["latitude"]), float(row["longitude"])))

        pairs = sorted(zip(self._gps_times, self._gps_points), key=lambda p: p[0])
        self._gps_times = [t for t, _ in pairs]
        self._gps_points = [p for _, p in pairs]

        if self._gps_times:
            logger.info("Loaded %d GNSS fixes spanning %.2fs - %.2fs",
                        len(self._gps_times), self._gps_times[0], self._gps_times[-1])

    # ----------------------------------------------------------------- lookup

    @staticmethod
    def _nearest_index(sorted_times: list[float], target: float) -> int:
        """Index of the closest value in a sorted list. Assumes non-empty."""
        i = bisect.bisect_left(sorted_times, target)
        if i == 0:
            return 0
        if i >= len(sorted_times):
            return len(sorted_times) - 1
        before, after = sorted_times[i - 1], sorted_times[i]
        return i if (after - target) < (target - before) else i - 1

    def get_frame(self, timestamp: float, event_id: str | None = None) -> str | None:
        """
        Path to the frame captured nearest `timestamp`, or None if the nearest
        one is further away than the tolerance.

        Returning None is meaningful: it says "the camera was not looking at
        this moment", which is a real condition and must not be papered over
        with a frame from somewhere else in the run. `event_id` is accepted and
        ignored, to keep the signature interchangeable with the mock.
        """
        if not self._frame_times:
            self.misses_no_frame += 1
            return None

        i = self._nearest_index(self._frame_times, timestamp)
        delta = abs(self._frame_times[i] - timestamp)

        if delta > self.frame_tolerance:
            self.misses_no_frame += 1
            logger.debug("No frame within %.3fs of t=%.4f (nearest was %.3fs away).",
                         self.frame_tolerance, timestamp, delta)
            return None

        return str(self._frame_paths[i])

    def get_gps(self, timestamp: float) -> tuple[float, float] | None:
        """
        Linearly interpolated (lat, lng) at `timestamp`, or None if outside the
        recorded window by more than the tolerance.

        Interpolation is safe here: a CARLA town spans a few hundred metres, so
        there is no antimeridian or pole to worry about.
        """
        if not self._gps_times:
            self.misses_no_gps += 1
            return None

        if timestamp <= self._gps_times[0]:
            within = self._gps_times[0] - timestamp <= self.gps_tolerance
            if not within:
                self.misses_no_gps += 1
                return None
            return self._gps_points[0]

        if timestamp >= self._gps_times[-1]:
            within = timestamp - self._gps_times[-1] <= self.gps_tolerance
            if not within:
                self.misses_no_gps += 1
                return None
            return self._gps_points[-1]

        i = bisect.bisect_left(self._gps_times, timestamp)
        t0, t1 = self._gps_times[i - 1], self._gps_times[i]
        (lat0, lng0), (lat1, lng1) = self._gps_points[i - 1], self._gps_points[i]

        span = t1 - t0
        if span <= 0:
            return (lat0, lng0)

        f = (timestamp - t0) / span
        return (lat0 + (lat1 - lat0) * f, lng0 + (lng1 - lng0) * f)

    # ------------------------------------------------------------ diagnostics

    def coverage(self) -> dict:
        """Summary of what this run can and cannot answer. Used by the self-check."""
        gaps = [b - a for a, b in zip(self._frame_times, self._frame_times[1:])]
        return {
            "run_dir": str(self.run_dir),
            "frames": len(self._frame_times),
            "frame_span_s": (self._frame_times[-1] - self._frame_times[0]) if self._frame_times else 0.0,
            "frame_gap_mean_s": (sum(gaps) / len(gaps)) if gaps else 0.0,
            "frame_gap_max_s": max(gaps) if gaps else 0.0,
            "frame_tolerance_s": self.frame_tolerance,
            "gnss_fixes": len(self._gps_times),
            "misses_no_frame": self.misses_no_frame,
            "misses_no_gps": self.misses_no_gps,
        }


def _self_check(run_dir: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    provider = CarlaFrameProvider(run_dir)

    print("\nCoverage")
    print("-" * 52)
    for key, value in provider.coverage().items():
        print(f"  {key:22} {value}")

    max_gap = provider.coverage()["frame_gap_max_s"]
    if max_gap > provider.frame_tolerance * 2:
        print(f"\n  WARNING: largest frame gap ({max_gap:.3f}s) is well over the "
              f"tolerance ({provider.frame_tolerance:.3f}s).")
        print("  Sensor rows landing in that gap will get no frame and be skipped.")

    print("\nSpot checks")
    print("-" * 52)
    span = provider._frame_times[-1] - provider._frame_times[0]
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = provider._frame_times[0] + span * f
        frame = provider.get_frame(t)
        gps = provider.get_gps(t)
        name = Path(frame).name if frame else "NONE"
        coords = f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "NONE"
        print(f"  t={t:8.3f}s  frame={name:<16} gps={coords}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print("Usage: python integration/carla_frame_provider.py <run_dir>")
        sys.exit(2)
    sys.exit(_self_check(sys.argv[1]))
