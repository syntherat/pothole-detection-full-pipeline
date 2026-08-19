"""
carla_sim/scenario/sensors.py

Sensor rig and synchronous-mode collection.

Sensors run at different rates on purpose:
  IMU    every tick (400 Hz) -- this is the signal the FSM consumes
  GNSS   every tick          -- cheap, and gives per-sample position
  camera 20 Hz via sensor_tick -- 400 Hz imagery is neither needed nor affordable

Because of that, the IMU/GNSS queues can be drained with a blocking get() each
tick, while the camera queue must be polled without blocking.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass

import carla

logger = logging.getLogger(__name__)


@dataclass
class ImuSample:
    frame: int
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class GnssSample:
    frame: int
    timestamp: float
    latitude: float
    longitude: float
    altitude: float


class SensorRig:
    """Owns every sensor actor and its queue. Always call `destroy()`."""

    def __init__(self, world: carla.World, vehicle: carla.Vehicle, cfg):
        self.world = world
        self.vehicle = vehicle
        self.cfg = cfg
        self.actors: list[carla.Actor] = []

        self.imu_queue: queue.Queue = queue.Queue()
        self.gnss_queue: queue.Queue = queue.Queue()
        self.image_queue: queue.Queue = queue.Queue()

        library = world.get_blueprint_library()
        self._spawn_imu(library)
        self._spawn_gnss(library)
        if cfg.CAMERA_ENABLED:
            self._spawn_camera(library)

    def _spawn_imu(self, library) -> None:
        bp = library.find("sensor.other.imu")
        bp.set_attribute("sensor_tick", "0.0")  # every tick

        # CARLA's IMU is noiseless by default, which is unrealistically clean --
        # a classifier trained on it solves a problem that does not exist.
        accel = str(self.cfg.IMU_NOISE_ACCEL_STDDEV)
        gyro = str(self.cfg.IMU_NOISE_GYRO_STDDEV)
        for axis in ("x", "y", "z"):
            if bp.has_attribute(f"noise_accel_stddev_{axis}"):
                bp.set_attribute(f"noise_accel_stddev_{axis}", accel)
            if bp.has_attribute(f"noise_gyro_stddev_{axis}"):
                bp.set_attribute(f"noise_gyro_stddev_{axis}", gyro)

        sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(*self.cfg.IMU_LOCATION)),
            attach_to=self.vehicle,
        )
        sensor.listen(self.imu_queue.put)
        self.actors.append(sensor)

    def _spawn_gnss(self, library) -> None:
        bp = library.find("sensor.other.gnss")
        bp.set_attribute("sensor_tick", "0.0")
        sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(*self.cfg.IMU_LOCATION)),
            attach_to=self.vehicle,
        )
        sensor.listen(self.gnss_queue.put)
        self.actors.append(sensor)

    def _spawn_camera(self, library) -> None:
        bp = library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.cfg.CAMERA_WIDTH))
        bp.set_attribute("image_size_y", str(self.cfg.CAMERA_HEIGHT))
        bp.set_attribute("fov", str(self.cfg.CAMERA_FOV))
        bp.set_attribute("sensor_tick", str(1.0 / self.cfg.CAMERA_HZ))

        sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(*self.cfg.CAMERA_LOCATION)),
            attach_to=self.vehicle,
        )
        sensor.listen(self.image_queue.put)
        self.actors.append(sensor)

    def next_imu(self, timeout: float = 2.0) -> ImuSample | None:
        """One IMU sample per tick. Blocking, because one is guaranteed."""
        try:
            d = self.imu_queue.get(timeout=timeout)
        except queue.Empty:
            logger.error("No IMU sample within %.1fs -- is the world still ticking?", timeout)
            return None
        return ImuSample(
            frame=d.frame, timestamp=d.timestamp,
            ax=d.accelerometer.x, ay=d.accelerometer.y, az=d.accelerometer.z,
            gx=d.gyroscope.x, gy=d.gyroscope.y, gz=d.gyroscope.z,
        )

    def next_gnss(self, timeout: float = 2.0) -> GnssSample | None:
        try:
            d = self.gnss_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return GnssSample(
            frame=d.frame, timestamp=d.timestamp,
            latitude=d.latitude, longitude=d.longitude, altitude=d.altitude,
        )

    def drain_images(self) -> list:
        """
        Non-blocking. The camera ticks far slower than the world, so most calls
        return nothing -- that is expected, not an error.
        """
        images = []
        while True:
            try:
                images.append(self.image_queue.get_nowait())
            except queue.Empty:
                return images

    def destroy(self) -> None:
        for actor in self.actors:
            try:
                actor.stop()
            except RuntimeError:
                pass  # already stopped
            actor.destroy()
        self.actors.clear()
