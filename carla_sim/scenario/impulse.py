"""
carla_sim/scenario/impulse.py

The Level A jerk model.

DESIGN NOTE -- read this before changing anything here.

We seed ONLY the drop. We do not script freefall and we do not script impact.

    1. A downward impulse at the wheel makes the body accelerate down.
    2. The suspension unloads. With the wheel unloaded the contact force the
       accelerometer measures falls toward zero -- that is a GENUINE freefall
       reading, produced by the physics engine, not by us.
    3. The spring re-compresses as the body falls back onto it -- that is a
       GENUINE impact spike, also from the engine.

So two of PotholeDetector's three phases come out of vehicle dynamics even at
Level A. Scripting all three would just replay generate_dataset.py inside a
prettier window and would prove nothing.

WHY OFF-CENTRE MATTERS
An impulse at the centre of mass produces NO rotation. The IMU also records
gx, gy, gz, so a centre-of-mass impulse leaves all three gyro channels flat --
and the RandomForest will happily learn 'flat gyro' as the tell, giving a model
that looks excellent in simulation and collapses on the first real one-sided
wheel strike. Applying at the wheel produces the roll and pitch a real
asymmetric hit produces, so all six channels carry signal.
"""

from __future__ import annotations

import logging

import carla

G = 9.81

logger = logging.getLogger(__name__)


class ImpulseApplier:
    """
    Applies a pothole impulse, adapting to whichever impulse API this CARLA
    build exposes.

    Two paths, physically equivalent:

      A. `add_impulse_at_location(J, p)` -- the engine derives the torque.
      B. `add_impulse(J)` + `add_angular_impulse(r x J)` -- we derive it, where
         r is the offset from the centre of mass to the strike point.

    Which one is in use is logged at construction. verify_setup.py reports the
    available methods.
    """

    def __init__(self, vehicle: carla.Vehicle, delta_v: float, ticks: int,
                 unload_ticks: int = 8, unload_scale: float = 1.0):
        self.vehicle = vehicle
        self.delta_v = delta_v
        self.ticks = max(1, ticks)
        self.unload_ticks = max(1, unload_ticks)
        self.unload_scale = unload_scale

        physics = vehicle.get_physics_control()
        self.mass = physics.mass
        self.com = physics.center_of_mass  # vehicle-local metres

        self.direct = hasattr(vehicle, "add_impulse_at_location")
        self._pending: list[tuple[carla.Vector3D, carla.Location]] = []
        self._pending_force: list[tuple[carla.Vector3D, carla.Location]] = []

        logger.info(
            "Impulse model: mass=%.0f kg, path=%s",
            self.mass,
            "add_impulse_at_location" if self.direct else "add_impulse + add_angular_impulse",
        )

    def schedule(self, severity: float, strike_point: carla.Location) -> None:
        """
        Queue a pothole impulse, spread over `ticks` so it reads as a push
        rather than a teleport.

        Magnitude scales with vehicle mass so the same severity means the same
        thing across vehicle blueprints:
            J = mass * severity * delta_v
        """
        total = self.mass * severity * self.delta_v
        per_tick = total / self.ticks

        # Straight down in world coordinates. CARLA is left-handed with Z up,
        # so a pothole is -Z.
        impulse = carla.Vector3D(0.0, 0.0, -per_tick)
        for _ in range(self.ticks):
            self._pending.append((impulse, strike_point))

    def schedule_unload(self, severity: float, strike_point: carla.Location) -> None:
        """
        Queue a sustained wheel-unloading phase: the wheel falling INTO a hole,
        rather than being kicked downward.

        Why this exists at all -- an impulse produces a 1-2 sample transient, and
        `PotholeDetector` discards any event whose near-zero phase is shorter
        than `min_air_time` (0.01 s). Every impulse-only setting therefore
        produced DROP, FREEFALL and IMPACT samples that were individually correct
        and still never formed a valid event. Issue #35.

        Magnitude is the vehicle's weight, so the downward force cancels the
        suspension's upward reaction and the accelerometer settles near zero for
        as long as it is held. Severity scales how completely it unloads.
        """
        magnitude = self.mass * G * severity * self.unload_scale
        force = carla.Vector3D(0.0, 0.0, -magnitude)
        for _ in range(self.unload_ticks):
            self._pending_force.append((force, strike_point))

    def _apply_at_point(self, vec: carla.Vector3D, point: carla.Location,
                        linear, angular) -> None:
        """
        Apply `vec` at `point` using the linear+angular pair given.

        Same decomposition the impulse fallback uses: a force (or impulse) at an
        off-centre point is the same force at the centre of mass plus a torque of
        r x F. Applying it only at the centre would leave the gyro channels flat,
        and the classifier would happily learn "flat gyro" as the tell.
        """
        linear(vec)

        transform = self.vehicle.get_transform()
        com_world = transform.transform(carla.Location(
            x=self.com.x, y=self.com.y, z=self.com.z
        ))
        rx = point.x - com_world.x
        ry = point.y - com_world.y
        rz = point.z - com_world.z

        angular(carla.Vector3D(
            x=ry * vec.z - rz * vec.y,
            y=rz * vec.x - rx * vec.z,
            z=rx * vec.y - ry * vec.x,
        ))

    def tick(self) -> None:
        """Apply one tick's worth of any queued impulse or force. Once per world tick."""
        if self._pending_force:
            force, point = self._pending_force.pop(0)
            self._apply_at_point(force, point,
                                 self.vehicle.add_force,
                                 self.vehicle.add_torque)

        if not self._pending:
            return

        impulse, point = self._pending.pop(0)

        if self.direct:
            self.vehicle.add_impulse_at_location(impulse, point)
            return

        # Fallback: linear impulse plus the angular impulse it would have caused.
        self.vehicle.add_impulse(impulse)

        transform = self.vehicle.get_transform()
        com_world = transform.transform(carla.Location(
            x=self.com.x, y=self.com.y, z=self.com.z
        ))

        # r = strike point relative to the centre of mass, in world axes.
        rx = point.x - com_world.x
        ry = point.y - com_world.y
        rz = point.z - com_world.z

        # L = r x J
        jx, jy, jz = impulse.x, impulse.y, impulse.z
        self.vehicle.add_angular_impulse(carla.Vector3D(
            x=ry * jz - rz * jy,
            y=rz * jx - rx * jz,
            z=rx * jy - ry * jx,
        ))

    @property
    def busy(self) -> bool:
        return bool(self._pending) or bool(self._pending_force)


def wheel_positions(vehicle: carla.Vehicle, scale: float) -> list[carla.Location]:
    """
    World positions of the four wheels.

    `scale` comes from config.WHEEL_POSITION_SCALE, measured by verify_setup.py:
    CARLA has documented WheelPhysicsControl.position inconsistently as metres
    and as centimetres, so it is measured rather than assumed.
    """
    if scale is None:
        raise RuntimeError(
            "config.WHEEL_POSITION_SCALE is None. "
            "Run `python carla_sim/verify_setup.py` and paste the measured value."
        )

    return [
        carla.Location(x=w.position.x * scale,
                       y=w.position.y * scale,
                       z=w.position.z * scale)
        for w in vehicle.get_physics_control().wheels
    ]
