"""
carla_sim/scenario/route.py

Deterministic route generation plus a minimal waypoint-following controller.

Why not autopilot: the Traffic Manager will not reliably drive over specific
coordinates, and Level A needs the vehicle to actually hit every pothole. So we
build the route FIRST, place potholes along that exact route, and drive it with
a simple controller. Same seed => same route => comparable runs.
"""

from __future__ import annotations

import logging
import math

import carla

logger = logging.getLogger(__name__)


def build_route(world_map: carla.Map, start: carla.Transform,
                length_m: float, step_m: float = 2.0) -> list[carla.Waypoint]:
    """
    Walk the lane graph forward from `start` for `length_m`.

    At junctions `next()` returns several options; we always take the first,
    which keeps the route deterministic without needing a seeded choice.
    """
    waypoint = world_map.get_waypoint(start.location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    route = [waypoint]
    travelled = 0.0

    while travelled < length_m:
        options = route[-1].next(step_m)
        if not options:
            logger.warning("Lane graph ended after %.0f m -- route is short.", travelled)
            break
        route.append(options[0])
        travelled += step_m

    logger.info("Route: %d waypoints, %.0f m", len(route), travelled)
    return route


class WaypointFollower:
    """
    Pure-pursuit steering plus proportional speed control.

    Deliberately simple. It only has to hold a lane at 30 km/h so the vehicle
    passes over known coordinates -- it is not meant to be a good driver.
    """

    def __init__(self, vehicle: carla.Vehicle, route: list[carla.Waypoint],
                 target_speed_kmh: float, lookahead_m: float = 5.0):
        self.vehicle = vehicle
        self.route = route
        self.target_speed = target_speed_kmh / 3.6
        self.lookahead = lookahead_m
        self.index = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.route) - 1

    def _advance_index(self, location: carla.Location) -> None:
        """Consume waypoints until the target is at least `lookahead` ahead."""
        while self.index < len(self.route) - 1:
            wp = self.route[self.index].transform.location
            if location.distance(wp) > self.lookahead:
                break
            self.index += 1

    def step(self) -> carla.VehicleControl:
        transform = self.vehicle.get_transform()
        location = transform.location
        self._advance_index(location)

        target = self.route[self.index].transform.location

        # Steering: signed angle between the vehicle's forward vector and the
        # bearing to the target, normalised into [-1, 1].
        forward = transform.get_forward_vector()
        to_target = carla.Vector3D(target.x - location.x, target.y - location.y, 0.0)

        norm = math.hypot(to_target.x, to_target.y)
        if norm < 1e-3:
            steer = 0.0
        else:
            to_target.x /= norm
            to_target.y /= norm
            cross = forward.x * to_target.y - forward.y * to_target.x
            dot = max(-1.0, min(1.0, forward.x * to_target.x + forward.y * to_target.y))
            steer = max(-1.0, min(1.0, math.atan2(cross, dot) / (math.pi / 4)))

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        error = self.target_speed - speed

        throttle = max(0.0, min(0.6, 0.35 * error))
        brake = max(0.0, min(0.4, -0.25 * error))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def speed_of(vehicle: carla.Vehicle) -> float:
    """Scalar speed in m/s -- the `speed` column of contract #1."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
