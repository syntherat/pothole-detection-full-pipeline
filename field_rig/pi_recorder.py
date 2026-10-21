"""
field_rig/pi_recorder.py

Records a real drive on a Raspberry Pi: an MPU-6050 for acceleration and rotation,
plus a GPS for position, speed and the true time. The camera is NOT recorded here;
a phone running Sensor Logger records video separately, and the two are lined up
afterwards (see README.md).

    python3 field_rig/pi_recorder.py --check                # wiring + mounting test, then exit
    python3 field_rig/pi_recorder.py                        # record until Ctrl+C
    python3 field_rig/pi_recorder.py --finalize <run_dir>   # rebuild sensors.csv from raw

Writes field_rig/out/run_<ts>/:
  imu_raw.csv      segment,index,ax,ay,az,gx,gy,gz   raw MPU counts, never modified
  imu_clock.csv    segment,host_t,samples            when each FIFO read happened
  gnss.csv         timestamp,latitude,longitude,altitude,speed_mps,course_deg,
                   fix_quality,satellites,gps_unix
  sensors.csv      timestamp,ax,ay,az,gx,gy,gz,speed,label,gps_fix,unix_time
                   <- contract #1 columns first, so the existing pipeline reads it
  run_meta.json    settings, calibration, measured rate, sync source

sensors.csv is DERIVED. The recorder writes only the raw files while driving and
builds sensors.csv when it stops. If the Pi loses power mid-drive the raw files
survive (they are flushed every second), and --finalize rebuilds sensors.csv from
them -- on the Pi or on a laptop, it needs only numpy.

`label` is left EMPTY. On a real road nobody knows which samples are a pothole
until the video is reviewed; writing 0 would claim "no pothole here" (Rule 6).
"""

import argparse
import csv
import json
import math
import signal
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"

# --- MPU-6050 registers -------------------------------------------------------
MPU_ADDR = 0x68
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
FIFO_EN = 0x23
INT_STATUS = 0x3A
ACCEL_XOUT_H = 0x3B
USER_CTRL = 0x6A
PWR_MGMT_1 = 0x6B
FIFO_COUNTH = 0x72
FIFO_R_W = 0x74
WHO_AM_I = 0x75

# 500 Hz, not the synthetic set's 400 Hz. The MPU-6050 cannot make exactly 400 Hz
# with its low-pass filter on (1 kHz / (1 + divider) gives 500 or 333), and turning
# the filter off to get 400 lets vibration above 200 Hz alias into the signal.
# PotholeDetector works from timestamps, never sample counts (sampling_rate_hz is
# stored and unused, issue #17), so 500 Hz feeds it unchanged.
RATE_HZ = 500
DLPF_CFG = 1            # ~184 Hz accel / 188 Hz gyro bandwidth, 1 kHz internal rate
SMPLRT_DIVIDER = 1      # 1 kHz / (1 + 1) = 500 Hz

# +-8 g. CARLA impacts peaked near 25 m/s^2 (2.5 g); a real wheel hitting a hole
# edge can go well past 4 g, and a clipped peak still clears the 19.81 gate but
# loses impact_acceleration. Resolution at 8 g is still 0.0024 m/s^2.
ACCEL_CONFIG_VAL = 0x10
ACCEL_LSB_PER_G = 4096.0
GYRO_CONFIG_VAL = 0x08  # +-500 deg/s
GYRO_LSB_PER_DPS = 65.5

STANDARD_G = 9.80665
SAMPLE_BYTES = 12       # accel xyz + gyro xyz, int16 big-endian, in FIFO order
FIFO_OVERFLOW_BIT = 0x10
POLL_S = 0.02           # FIFO holds 85 samples = 170 ms at 500 Hz; drain well before

KNOTS_TO_MPS = 0.514444
GPS_STALE_S = 3.0       # beyond this from the nearest fix, speed is unknown
GPS_PORTS = ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/serial0"]

RAW_COLUMNS = ["segment", "index", "ax", "ay", "az", "gx", "gy", "gz"]
GNSS_COLUMNS = ["timestamp", "latitude", "longitude", "altitude", "speed_mps",
                "course_deg", "fix_quality", "satellites", "gps_unix"]
SENSOR_COLUMNS = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "speed", "label",
                  "gps_fix", "unix_time"]

_stop = threading.Event()


# =============================================================================
# MPU-6050
# =============================================================================

class Mpu6050:
    """FIFO-driven reader. The chip's own clock spaces the samples, so Python's
    scheduling jitter on the Pi never reaches the data -- only the read times."""

    def __init__(self, bus_num: int = 1, addr: int = MPU_ADDR):
        from smbus2 import SMBus, i2c_msg
        self._i2c_msg = i2c_msg
        self.bus = SMBus(bus_num)
        self.addr = addr
        self.who_am_i = self.bus.read_byte_data(addr, WHO_AM_I)

    def configure(self) -> None:
        w = self.bus.write_byte_data
        w(self.addr, PWR_MGMT_1, 0x80)          # reset
        time.sleep(0.1)
        w(self.addr, PWR_MGMT_1, 0x01)          # wake, clock from gyro X PLL (steadier than the RC oscillator)
        time.sleep(0.05)
        w(self.addr, CONFIG, DLPF_CFG)
        w(self.addr, SMPLRT_DIV, SMPLRT_DIVIDER)
        w(self.addr, GYRO_CONFIG, GYRO_CONFIG_VAL)
        w(self.addr, ACCEL_CONFIG, ACCEL_CONFIG_VAL)
        w(self.addr, FIFO_EN, 0x78)             # accel + gyro xyz
        self.reset_fifo()

    def reset_fifo(self) -> None:
        self.bus.write_byte_data(self.addr, USER_CTRL, 0x04)   # FIFO reset
        self.bus.write_byte_data(self.addr, USER_CTRL, 0x40)   # FIFO enable
        self.bus.read_byte_data(self.addr, INT_STATUS)         # clear a stale overflow flag

    def overflowed(self) -> bool:
        return bool(self.bus.read_byte_data(self.addr, INT_STATUS) & FIFO_OVERFLOW_BIT)

    def read_fifo(self) -> list[tuple[int, ...]]:
        hi, lo = self.bus.read_i2c_block_data(self.addr, FIFO_COUNTH, 2)
        count = (hi << 8) | lo
        n = (count // SAMPLE_BYTES) * SAMPLE_BYTES
        if n == 0:
            return []
        wr = self._i2c_msg.write(self.addr, [FIFO_R_W])
        rd = self._i2c_msg.read(self.addr, n)
        self.bus.i2c_rdwr(wr, rd)
        data = bytes(list(rd))
        return [struct.unpack_from(">6h", data, i) for i in range(0, n, SAMPLE_BYTES)]

    def read_once(self) -> tuple[int, ...]:
        """One direct register read, for --check before the FIFO is running."""
        data = bytes(self.bus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14))
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", data)
        return ax, ay, az, gx, gy, gz

    def close(self) -> None:
        self.bus.close()


def counts_to_si(raw: np.ndarray) -> np.ndarray:
    """(N, 6) int counts -> (N, 6) m/s^2 and rad/s."""
    out = raw.astype(np.float64)
    out[:, :3] *= STANDARD_G / ACCEL_LSB_PER_G
    out[:, 3:] *= math.pi / 180.0 / GYRO_LSB_PER_DPS
    return out


# =============================================================================
# GPS (NMEA over serial)
# =============================================================================

def _nmea_valid(line: str) -> bool:
    if not line.startswith("$") or "*" not in line:
        return False
    body, _, checksum = line[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(checksum[:2], 16)
    except ValueError:
        return False


def _nmea_deg(value: str, hemi: str) -> float | None:
    if not value:
        return None
    v = float(value)
    deg = int(v // 100) + (v % 100) / 60.0
    return -deg if hemi in ("S", "W") else deg


def _rmc_unix(time_s: str, date_s: str) -> float | None:
    if len(time_s) < 6 or len(date_s) != 6:
        return None
    frac = float(time_s[6:] or 0)
    dt = datetime(2000 + int(date_s[4:6]), int(date_s[2:4]), int(date_s[0:2]),
                  int(time_s[0:2]), int(time_s[2:4]), int(time_s[4:6]), tzinfo=timezone.utc)
    return dt.timestamp() + frac


class GpsReader(threading.Thread):
    """Reads NMEA, writes one gnss.csv row per valid RMC fix.

    gps_unix is the satellite time of the fix. Paired with this Pi's own clock at
    the moment the sentence arrived, it gives the true time of every IMU sample --
    the Pi has no battery clock, so without GPS its idea of "now" can be days off.
    NMEA arrives a little after the second it describes (typically <0.5 s on 9600
    baud); that lag is inside the 0.5-2 s frame lead the vision stage uses.
    """

    def __init__(self, port: str, baud: int, out_path: Path, t0: float):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.port = port
        self.t0 = t0
        self.out_path = out_path
        self.fixes = 0
        self.last = {"fix_quality": 0, "satellites": 0, "speed_mps": None}
        self._gga = {"altitude": "", "fix_quality": "", "satellites": ""}

    def run(self) -> None:
        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(GNSS_COLUMNS)
            last_flush = time.monotonic()
            while not _stop.is_set():
                try:
                    line = self.ser.readline().decode("ascii", errors="ignore").strip()
                except Exception as exc:  # unplugged mid-drive: keep the IMU running
                    print(f"[gps] read failed: {exc}")
                    time.sleep(1.0)
                    continue
                host_t = time.monotonic() - self.t0
                if not _nmea_valid(line):
                    continue
                fields = line.split("*")[0].split(",")
                kind = fields[0][3:]
                if kind == "GGA" and len(fields) > 9:
                    self._gga = {"fix_quality": fields[6], "satellites": fields[7],
                                 "altitude": fields[9]}
                    self.last["fix_quality"] = int(fields[6] or 0)
                    self.last["satellites"] = int(fields[7] or 0)
                elif kind == "RMC" and len(fields) > 9 and fields[2] == "A":
                    lat = _nmea_deg(fields[3], fields[4])
                    lon = _nmea_deg(fields[5], fields[6])
                    if lat is None or lon is None:
                        continue
                    speed = float(fields[7] or 0.0) * KNOTS_TO_MPS
                    gps_unix = _rmc_unix(fields[1], fields[9])
                    writer.writerow([
                        f"{host_t:.6f}", f"{lat:.8f}", f"{lon:.8f}", self._gga["altitude"],
                        f"{speed:.3f}", fields[8], self._gga["fix_quality"],
                        self._gga["satellites"],
                        f"{gps_unix:.3f}" if gps_unix is not None else "",
                    ])
                    self.fixes += 1
                    self.last["speed_mps"] = speed
                if time.monotonic() - last_flush > 1.0:
                    f.flush()
                    last_flush = time.monotonic()
        self.ser.close()


def find_gps_port(requested: str) -> str | None:
    if requested != "auto":
        return requested
    for p in GPS_PORTS:
        if Path(p).exists():
            return p
    return None


# =============================================================================
# Calibration + levelling
# =============================================================================

def calibrate(si: np.ndarray) -> dict:
    """Stats over the still window at the start of the run."""
    acc, gyr = si[:, :3], si[:, 3:]
    g_vec = acc.mean(axis=0)
    g_mag = float(np.linalg.norm(g_vec))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, g_vec[2] / g_mag)))) if g_mag else 180.0
    return {
        "samples": int(len(si)),
        "gravity_vector": [round(float(x), 4) for x in g_vec],
        "gravity_magnitude": round(g_mag, 4),
        "tilt_from_z_deg": round(tilt, 2),
        "gyro_bias_rad_s": [round(float(x), 6) for x in gyr.mean(axis=0)],
        "accel_std": round(float(acc.std(axis=0).max()), 4),
        "gyro_std": round(float(gyr.std(axis=0).max()), 5),
    }


def level_rotation(g_vec) -> np.ndarray:
    """Rotation taking the measured gravity direction onto +z.

    PotholeDetector reads `az` as vertical with +9.81 at rest. A sensor taped down
    a few degrees off level leaks part of every bump into ax/ay and shortens az;
    this undoes that. Heading (rotation about z) is left as mounted -- only the
    RandomForest sees ax/ay, and it never knew which way was forward anyway.
    """
    v = np.asarray(g_vec, dtype=np.float64)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    c = float(np.dot(v, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def mounting_verdict(cal: dict) -> list[str]:
    """Plain-language problems with the mount. Empty list means fine."""
    problems = []
    if cal["accel_std"] > 0.3 or cal["gyro_std"] > 0.02:
        problems.append("the sensor was MOVING during calibration -- keep the car still and engine idle")
    if cal["tilt_from_z_deg"] > 30:
        # Levelling corrects any orientation, upside down included. Flagged anyway:
        # with --no-level this mount makes az useless, and a steep mount usually
        # means the sensor is on something that is not the car's floor.
        problems.append(f"the sensor is {cal['tilt_from_z_deg']:.0f} deg from flat"
                        + (" (UPSIDE DOWN)" if cal["tilt_from_z_deg"] > 150 else "")
                        + " -- corrected in software unless --no-level, but mount it flat, chip up")
    if abs(cal["gravity_magnitude"] - STANDARD_G) > 0.5:
        problems.append(f"gravity reads {cal['gravity_magnitude']:.2f} m/s^2, not ~9.8 -- "
                        "check the wiring, or the board may be faulty")
    return problems


# =============================================================================
# Recording
# =============================================================================

def new_run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    n = 1
    while run_dir.exists():           # the Pi's clock can repeat across boots without network
        run_dir = base / f"run_{stamp}_{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def check(args) -> int:
    """Wiring and mounting test. Reads 3 s directly, prints a verdict, records nothing."""
    try:
        mpu = Mpu6050(args.i2c_bus)
    except OSError as exc:
        print(f"[FAIL] no sensor on I2C bus {args.i2c_bus}: {exc}")
        print("       Is I2C enabled (raspi-config)? Run `i2cdetect -y 1` -- you want to see 68.")
        return 1
    print(f"[ok] sensor found, WHO_AM_I = 0x{mpu.who_am_i:02x}"
          + ("" if mpu.who_am_i == 0x68 else "  (not 0x68: probably a clone chip, usually fine)"))
    mpu.configure()
    rows = []
    end = time.monotonic() + args.calib_seconds
    while time.monotonic() < end:
        rows.append(mpu.read_once())
        time.sleep(0.005)
    mpu.close()
    cal = calibrate(counts_to_si(np.array(rows)))
    print(f"[..] gravity {cal['gravity_vector']}  |g|={cal['gravity_magnitude']}  "
          f"tilt={cal['tilt_from_z_deg']} deg")
    problems = mounting_verdict(cal)
    for p in problems:
        print(f"[!!] {p}")
    if not problems:
        print("[ok] mounting looks right")

    if args.no_gps:
        return 0 if not problems else 1
    port = find_gps_port(args.gps_port)
    if port is None:
        print(f"[!!] no GPS found at {', '.join(GPS_PORTS)} -- is the USB GPS plugged in?")
        return 1
    import serial
    with serial.Serial(port, args.gps_baud, timeout=1.0) as ser:
        seen = [ser.readline().decode("ascii", errors="ignore").strip() for _ in range(10)]
    nmea = [s for s in seen if _nmea_valid(s)]
    if not nmea:
        print(f"[!!] {port} opened but sent no NMEA -- wrong port or baud rate?")
        return 1
    print(f"[ok] GPS talking on {port} ({len(nmea)}/10 valid sentences). "
          "A fix can take a few minutes outdoors the first time.")
    return 0 if not problems else 1


def record(args) -> int:
    run_dir = new_run_dir(Path(args.out) if args.out else OUT_DIR)
    print(f"recording to {run_dir}")

    mpu = Mpu6050(args.i2c_bus)
    mpu.configure()
    t0 = time.monotonic()
    meta = {
        "rig": "field_rig/pi_recorder.py",
        "started_system_time": datetime.now(timezone.utc).isoformat(),
        "started_system_unix": time.time(),
        "mpu_who_am_i": f"0x{mpu.who_am_i:02x}",
        "rate_hz_nominal": RATE_HZ,
        "dlpf_cfg": DLPF_CFG,
        "accel_range_g": 8,
        "gyro_range_dps": 500,
        "calib_seconds": args.calib_seconds,
        "level": not args.no_level,
        "gps_port": None,
    }

    gps = None
    if not args.no_gps:
        port = find_gps_port(args.gps_port)
        if port is None:
            print("[!!] no GPS found -- recording IMU only; speed will be 0 and there is no true time")
        else:
            gps = GpsReader(port, args.gps_baud, run_dir / "gnss.csv", t0)
            gps.start()
            meta["gps_port"] = port

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: _stop.set())

    segment, index, overflows = 0, 0, 0
    calib_rows: list[tuple[int, ...]] = []
    calib_done = False
    last_flush = last_status = time.monotonic()

    with (run_dir / "imu_raw.csv").open("w", newline="", encoding="utf-8") as rf, \
         (run_dir / "imu_clock.csv").open("w", newline="", encoding="utf-8") as cf:
        raw_w, clk_w = csv.writer(rf), csv.writer(cf)
        raw_w.writerow(RAW_COLUMNS)
        clk_w.writerow(["segment", "host_t", "samples"])
        print(f"KEEP THE CAR STILL for {args.calib_seconds:.0f} s ...")

        while not _stop.is_set():
            if mpu.overflowed():
                # Samples were lost and how many is unknowable, so the index-based
                # clock cannot bridge it. Start a new segment; finalize fits each
                # segment's timing separately.
                overflows += 1
                mpu.reset_fifo()
                segment, index = segment + 1, 0
                print(f"[!!] FIFO overflow #{overflows} -- a few ms lost. "
                      "If this repeats, set the I2C bus to 400 kHz (README).")
                continue
            samples = mpu.read_fifo()
            host_t = time.monotonic() - t0
            if samples:
                for s in samples:
                    raw_w.writerow((segment, index, *s))
                    index += 1
                clk_w.writerow((segment, f"{host_t:.6f}", index))
                if not calib_done and segment == 0:
                    calib_rows.extend(samples)
                    if len(calib_rows) >= args.calib_seconds * RATE_HZ:
                        calib_done = True
                        cal = calibrate(counts_to_si(np.array(calib_rows)))
                        problems = mounting_verdict(cal)
                        for p in problems:
                            print(f"[!!] {p}")
                        print("calibration done -- you can drive." if not problems
                              else "calibration done, WITH PROBLEMS above. Ctrl+C, fix, restart.")

            now = time.monotonic()
            if now - last_flush > 1.0:
                rf.flush()
                cf.flush()
                last_flush = now
            if now - last_status > 5.0:
                gps_txt = "no GPS"
                if gps is not None:
                    sp = gps.last["speed_mps"]
                    gps_txt = (f"GPS fixes={gps.fixes} sats={gps.last['satellites']} "
                               f"speed={'-' if sp is None else f'{sp * 3.6:.0f} km/h'}")
                print(f"[{host_t:7.0f} s] {index} samples in seg {segment}, "
                      f"overflows={overflows} | {gps_txt}")
                last_status = now
            time.sleep(POLL_S)

    mpu.close()
    if gps is not None:
        gps.join(timeout=3.0)
        meta["gps_fixes"] = gps.fixes
    meta["stopped_after_s"] = round(time.monotonic() - t0, 3)
    meta["fifo_overflows"] = overflows
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("stopped -- building sensors.csv ...")
    return finalize(run_dir)


# =============================================================================
# Finalize: raw files -> sensors.csv
# =============================================================================

def _segment_times(seg_idx: np.ndarray, clock: np.ndarray) -> tuple[np.ndarray, float]:
    """Timestamps for one segment from its read log.

    Each read records (host time, samples so far). The newest sample in that read
    was made just before host time, so host_t ~= a + period * (samples - 1). A
    straight-line fit over the whole segment averages out read jitter AND measures
    the chip's true rate, which differs from 500 Hz by up to a percent or two --
    over a 30-minute drive that is tens of seconds, far too much to ignore when
    lining up with video.
    """
    if len(clock) >= 2:
        period, offset = np.polyfit(clock[:, 1] - 1, clock[:, 0], 1)
    else:
        period = 1.0 / RATE_HZ
        offset = (clock[0, 0] - period * (clock[0, 1] - 1)) if len(clock) else 0.0
    return offset + period * seg_idx, float(period)


def finalize(run_dir: Path) -> int:
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "note": "run_meta.json was missing (power cut?); rebuilt by --finalize",
        "calib_seconds": 3.0, "level": True}

    raw = _load_complete_rows(run_dir / "imu_raw.csv", len(RAW_COLUMNS), np.int64)
    clock = _load_complete_rows(run_dir / "imu_clock.csv", 3, np.float64)
    if len(raw) == 0:
        print("[FAIL] imu_raw.csv is empty")
        return 1

    ts = np.empty(len(raw))
    periods = []
    for seg in np.unique(raw[:, 0]):
        rows = raw[:, 0] == seg
        seg_clock = clock[clock[:, 0] == seg][:, 1:]
        ts[rows], period = _segment_times(raw[rows, 1].astype(np.float64), seg_clock)
        periods.append(period)
    order = np.argsort(ts, kind="stable")
    ts, raw = ts[order], raw[order]

    si = counts_to_si(raw[:, 2:8])
    calib_n = int(meta.get("calib_seconds", 3.0) * RATE_HZ)
    cal = calibrate(si[:calib_n])
    si[:, 3:] -= np.array(cal["gyro_bias_rad_s"])
    if meta.get("level", True):
        rot = level_rotation(cal["gravity_vector"])
        si[:, :3] = si[:, :3] @ rot.T
        si[:, 3:] = si[:, 3:] @ rot.T
        cal["level_rotation"] = [[round(float(x), 6) for x in r] for r in rot]

    # Speed and true time from GPS.
    speed = np.zeros(len(ts))
    gps_fix = np.zeros(len(ts), dtype=np.int64)
    unix_offset, sync_source = None, "system_clock"
    gnss_path = run_dir / "gnss.csv"
    if gnss_path.exists():
        with gnss_path.open(newline="", encoding="utf-8") as f:
            fixes = [r for r in csv.DictReader(f) if r["speed_mps"]]
        if fixes:
            g_t = np.array([float(r["timestamp"]) for r in fixes])
            g_v = np.array([float(r["speed_mps"]) for r in fixes])
            ok = _nearest_gap(ts, g_t) <= GPS_STALE_S
            speed[ok] = np.interp(ts[ok], g_t, g_v)
            gps_fix[ok] = 1
            offs = [float(r["gps_unix"]) - float(r["timestamp"]) for r in fixes if r["gps_unix"]]
            if offs:
                unix_offset = float(np.median(offs))
                sync_source = "gps"
    if unix_offset is None and "started_system_unix" in meta:
        unix_offset = float(meta["started_system_unix"])
    unix_time = ts + (unix_offset if unix_offset is not None else 0.0)

    out = np.column_stack([ts, si, speed, gps_fix, unix_time])
    fmt = "%.6f,%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.3f,,%d,%.3f"
    np.savetxt(run_dir / "sensors.csv", out, fmt=fmt, header=",".join(SENSOR_COLUMNS), comments="")

    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    meta.update({
        "calibration": cal,
        "mounting_problems": mounting_verdict(cal),
        "rows": int(len(ts)),
        "segments": int(len(periods)),
        "measured_rate_hz": [round(1.0 / p, 3) for p in periods if p > 0],
        "duration_s": round(duration, 3),
        "rows_with_gps_speed": int(gps_fix.sum()),
        "time_sync_source": sync_source,
        "label_note": "label is empty: ground truth comes from reviewing the phone video",
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"sensors.csv: {len(ts)} rows over {duration:.1f} s, "
          f"rate {meta['measured_rate_hz']} Hz, {int(gps_fix.sum())} rows with GPS speed, "
          f"time from {sync_source}")
    for p in meta["mounting_problems"]:
        print(f"[!!] {p}")
    return 0


def _load_complete_rows(path: Path, n_cols: int, dtype) -> np.ndarray:
    """Load a raw CSV, dropping a half-written last line -- what a power cut leaves."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()[1:]
    if lines and not text.endswith("\n"):
        lines[-1] = ""      # cut mid-line: even a well-formed tail may be a truncated number
    rows = [ln for ln in lines if ln.count(",") == n_cols - 1 and not ln.endswith(",")]
    if len(rows) < len(lines):
        print(f"[..] {path.name}: skipped {len(lines) - len(rows)} incomplete line(s)")
    if not rows:
        return np.empty((0, n_cols), dtype=dtype)
    return np.loadtxt(rows, delimiter=",", dtype=dtype, ndmin=2)


def _nearest_gap(ts: np.ndarray, g_t: np.ndarray) -> np.ndarray:
    """Distance from each IMU time to the closest GPS fix, without an N x M matrix."""
    if len(g_t) == 1:
        return np.abs(ts - g_t[0])
    pos = np.clip(np.searchsorted(g_t, ts), 1, len(g_t) - 1)
    return np.minimum(np.abs(ts - g_t[pos - 1]), np.abs(ts - g_t[pos]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="test wiring, mounting and GPS, then exit")
    ap.add_argument("--finalize", type=Path, metavar="RUN_DIR", help="rebuild sensors.csv from raw files")
    ap.add_argument("--out", help=f"parent folder for runs (default {OUT_DIR})")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--gps-port", default="auto", help=f"serial port, or 'auto' to try {GPS_PORTS}")
    ap.add_argument("--gps-baud", type=int, default=9600)
    ap.add_argument("--no-gps", action="store_true")
    ap.add_argument("--calib-seconds", type=float, default=3.0,
                    help="still time at the start used to measure gravity and gyro bias")
    ap.add_argument("--no-level", action="store_true", help="do not rotate gravity onto +z")
    args = ap.parse_args()

    if args.finalize:
        return finalize(args.finalize)
    if args.check:
        return check(args)
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
