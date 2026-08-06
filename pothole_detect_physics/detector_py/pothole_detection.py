import math
from dataclasses import dataclass
from typing import Optional

G = 9.81  # gravitational acceleration (m/s^2)


@dataclass
class PotholeEvent:
    pothole_detected: bool
    depth_estimate: Optional[float]
    length_estimate: Optional[float]
    air_time: Optional[float]
    impact_acceleration: Optional[float]


class PotholeDetector:
    """
    IMU-based pothole detector using a 3-phase acceleration pattern.

    Phases:
    1. DROP: vertical acceleration drops below threshold
    2. FREEFALL: acceleration approaches 0
    3. IMPACT: strong positive spike

    From these phases we estimate:
    - air_time
    - pothole depth
    - pothole length
    """

    def __init__(
        self,
        sampling_rate_hz: float = 400.0,
        drop_margin: float = 3.0,
        impact_margin: float = 10.0,
        freefall_threshold: float = 2.0,
        min_air_time: float = 0.01,
        max_air_time: float = 0.25,
    ):

        self.sampling_rate_hz = sampling_rate_hz

        self.drop_threshold = G - drop_margin
        self.impact_threshold = G + impact_margin
        self.freefall_threshold = freefall_threshold

        self.min_air_time = min_air_time
        self.max_air_time = max_air_time

        self.state = "IDLE"

        self.t_drop_start = None
        self.t_freefall_start = None
        self.t_impact = None

        self.speed_at_impact = None
        self.impact_accel = None

    def reset_state(self):
        """Reset detector state."""
        self.state = "IDLE"
        self.t_drop_start = None
        self.t_freefall_start = None
        self.t_impact = None
        self.speed_at_impact = None
        self.impact_accel = None

    def process_sample(
        self,
        timestamp: float,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        speed: float,
    ):

        event = PotholeEvent(
            pothole_detected=False,
            depth_estimate=None,
            length_estimate=None,
            air_time=None,
            impact_acceleration=None,
        )

        # -------------------------
        # Finite State Machine
        # -------------------------

        if self.state == "IDLE":

            if az < self.drop_threshold:
                self.state = "DROP"
                self.t_drop_start = timestamp

        elif self.state == "DROP":

            if abs(az) < self.freefall_threshold:
                self.state = "FREEFALL"
                self.t_freefall_start = timestamp

            elif az > (G - 0.5):
                self.reset_state()

        elif self.state == "FREEFALL":

            if az > self.impact_threshold:

                self.t_impact = timestamp
                self.speed_at_impact = speed
                self.impact_accel = az

                event = self._finalize_event()

                self.reset_state()

            elif az > (G - 0.5):
                self.reset_state()

        return {
            "pothole_detected": event.pothole_detected,
            "depth_estimate": event.depth_estimate,
            "length_estimate": event.length_estimate,
            "air_time": event.air_time,
            "impact_acceleration": event.impact_acceleration,
        }

    def _finalize_event(self):

        if (
            self.t_drop_start is None
            or self.t_freefall_start is None
            or self.t_impact is None
            or self.speed_at_impact is None
            or self.impact_accel is None
        ):
            return PotholeEvent(False, None, None, None, None)

        t_air = self.t_impact - self.t_freefall_start
        delta_t = self.t_impact - self.t_drop_start

        if t_air < self.min_air_time or t_air > self.max_air_time:
            return PotholeEvent(False, None, None, None, None)

        # Depth formula
        depth_estimate = (G * (t_air ** 2)) / 8.0

        # Length formula
        length_estimate = self.speed_at_impact * delta_t

        return PotholeEvent(
            pothole_detected=True,
            depth_estimate=depth_estimate,
            length_estimate=length_estimate,
            air_time=t_air,
            impact_acceleration=self.impact_accel,
        )