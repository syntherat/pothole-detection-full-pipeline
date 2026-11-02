# 16 — Field Rig: Real-Drive Validation (Phone + Raspberry Pi)

**Scope:** the plan for validating PAVE on a real car instead of CARLA: why, the options weighed, the
hardware chosen, what was bought, how it is set up, and what is still open. Code lives in `field_rig/`.
**Last verified against:** working tree on 2026-10-21 (`field_rig/` uncommitted at that date).
**Status:** recorder written and tested on **simulated hardware only**. Parts chosen; **not yet received**.
No real recording exists.

---

## Why this exists

The CARLA validation (see [`15-carla-testbed-plan.md`](15-carla-testbed-plan.md)) could not be finished on
the Windows machine. The user asked whether "the phone spec discussed earlier" was still an option.

**That earlier phone discussion is not recorded anywhere.** On 2026-10-07 it was searched for in the repo,
`context/`, `docs/`, agent memory and past session transcripts on the Mac, with no match. Everything below
was rebuilt from scratch that day. If the original discussion turns up and disagrees, reconcile it here.

A real drive also helps vision on its own terms: `best.pt` confirmed **0/45** CARLA pothole frames but
**10/16** real dash-camera frames ([`04-current-state.md`](04-current-state.md)). Real footage is a fairer
test of that model than CARLA renders.

---

## What a field rig must provide

The cascade needs three inputs. The rig must deliver each one in the shape the pipeline already reads:

| Input | Pipeline requirement | Where it is defined |
|---|---|---|
| IMU | `timestamp, ax, ay, az, gx, gy, gz, speed`; `az` vertical with ≈ +9.81 at rest; m/s² and rad/s | contract #1, [`20-data-contracts.md`](20-data-contracts.md) |
| IMU rate | FSM tuned on 400 Hz; shortest valid air time 0.01 s = 4 samples at 400 Hz | [`10-physics-sensor-pipeline.md`](10-physics-sensor-pipeline.md) |
| Camera | Forward-facing road video, frames picked 0.5–2.0 s before an event | [`11-vision-pipeline.md`](11-vision-pipeline.md) |
| GPS | Position plus speed (speed feeds the FSM's length estimate) | contract #1 |
| Ground truth | Which moments were real potholes | none on a real road; see "Ground truth" |

---

## Options considered

### Option 1 — phone only (accelerometer + camera + GPS from one Android phone)
Simplest: one app records everything. **Rejected as the primary sensor source** because of:

1. **Rate.** iPhones give about 100 Hz. Many Androids reach 200–500 Hz, but the actual figure varies
   by device. At 100 Hz the 0.01 s minimum air time is about 1 sample, so small potholes are likely missed.
2. **Orientation.** A phone standing on the windscreen has gravity on its y-axis, not z. It needs rotating.
3. **Units.** Some apps log in g, not m/s².
4. **Mounting.** A phone in a windscreen cradle moves relative to the car body.

The phone is **kept as the camera**, and it may still turn out to be usable as the IMU. The pending
10-second Sensor Logger test (below) measures this phone's real rate.

### Option 2 — Raspberry Pi 4 + MPU-6050 + GPS for sensors, phone for video ✅ CHOSEN
The user already owns a Pi 4. The MPU-6050 easily exceeds 400 Hz (the rig runs it at 500 Hz) and is taped
rigidly to the car body. The phone records video only. **The two devices do not talk during the drive.**
Each records on its own and they are aligned afterwards (see "Synchronisation").

### Rejected sub-options
| Rejected | Why |
|---|---|
| Streaming phone video to the Pi live | Not needed for offline validation; adds a failure point |
| Breadboard | Friction-fit parts rattle in a car, and the rattle reads as fake bumps. The sensor still needs soldered pins to plug in. Direct female-female jumpers instead |
| NEO-6M GPS board (GY-NEO6MV2) | Needs soldering and UART wiring. A USB dongle plugs in |
| Loose header pins pushed into holes without solder | Holds on a desk, wobbles loose on a road |
| Pi Camera module | The phone is the camera |
| Adafruit MPU-6050 with STEMMA QT (no-solder) | Works, but costs ~₹1,000–1,500+ with cable and is hard to find in India. Fallback only if soldering is impossible |
| Pi running YOLO live | Pi 4 manages only a few FPS. Record now, process on a laptop |

---

## Phone: app and settings

**App: Sensor Logger** by Kelvin Choi. Google Play package `com.kelvin.sensorapp`. Chosen because one
recording captures accelerometer, gyroscope, GPS and back-camera video, and exports as Zipped CSV.

| Setting | Value |
|---|---|
| Sensors on | Accelerometer, Gyroscope, Location, Camera (video, back) |
| IMU sampling | Highest / fastest |
| Export | Zipped CSV |

⚠ **Video records only while the app is in the foreground.** The phone stays unlocked with Sensor Logger on
screen and on a charger for the whole drive. This is why the Pi recorder is started **before** the phone,
over SSH, and the SSH app is then left.

The phone also serves as the **Wi-Fi hotspot** the Pi joins, so it can be controlled over SSH in the car.

**Test export received 2026-11-02** (Sensor Logger 1.68, Android, device `CPH2491`, ~12 s, phone flat on a
table). Kept outside the repo; it contains the user's location. What it showed:

| File | Columns | Measured rate |
|---|---|---|
| `TotalAcceleration.csv` | `time,seconds_elapsed,z,y,x` (m/s², **with** gravity) | 400 Hz |
| `Accelerometer.csv` | same (gravity removed) | 200 Hz |
| `AccelerometerUncalibrated.csv` | same | 400 Hz |
| `Gyroscope.csv` / `GyroscopeUncalibrated.csv` | same (rad/s) | 400 Hz |
| `Location.csv` | `time,seconds_elapsed,…,speed,…,longitude,latitude` | ~1 Hz |
| `Camera/<epoch_ms>.jpg` | one still per file, **1280×720** | **~1.8 fps, not video** |
| `Annotation.csv` | empty (no annotations taken) | — |
| `Metadata.csv` | device, `recording epoch time` (ms), timezone, `sampleRateMs` | — |

- `time` is **Unix epoch in nanoseconds**; JPG filenames are **Unix epoch in milliseconds**, same clock. So
  phone IMU and frames need no alignment with each other — only phone ↔ Pi (see Synchronisation).
- Column order is `z,y,x`, not `x,y,z`. Read by name.
- Use `TotalAcceleration` (400 Hz, gravity kept) for cross-correlation with the MPU-6050, not
  `Accelerometer` (200 Hz, gravity removed).
- The phone itself reaches 400 Hz, the synthetic set's rate, so it could serve as a backup IMU.
- **Camera captured stills, not video**, at ~0.57 s spacing. Video mode is a **paid** feature in Sensor
  Logger (user checked 2026-11-02). Decision: first drive uses the free stills; upgrade only if they prove
  too sparse. Stills are usable (frames are picked 0.5–2.0 s before an event, so 2–3 fall in the window),
  and their epoch-ms filenames make the merge simpler than decoding video.
- **Annotations are the ground-truth plan** (decided 2026-11-02, replaces passenger-says-pothole-aloud).
  The passenger, never the driver, taps a saved preset as the car crosses each one. Presets, spelled exactly:
  `pothole`, `speed_breaker`, `rough_road`. `pothole` → `label` 1.0; `speed_breaker` stays 0.0 but is kept
  because it is the main false-positive to measure (contract §1 already treats breakers as 0.0). Taps lag
  the real hit by up to ~1–2 s, so the merge script must match each tap to the strongest MPU spike in a
  window *before* it, not at the tap time.
- **`Annotation.csv` format** (seen 2026-11-02, 8 test taps): `time,seconds_elapsed,text,millisecond_press_duration`.
  `text` is the preset name verbatim. ⚠ **`time` is NOT Unix epoch** like every other file: it is the phone's
  boot clock (uptime) in ms ×10⁶ (e.g. `797028797000000` ≈ 9.2 days), and `seconds_elapsed` is garbage
  (that value minus the epoch start, ≈ −1.79e9). Spacing between taps is correct.
- **Recovering the offset:** each tap shows as a jolt in the phone's own `TotalAcceleration`. In the test, the
  8 tap spacings matched 8 accel spikes to within ~20 ms, giving the boot→epoch offset exactly. The offset is
  constant until the phone reboots, but do not rely on that. **Drive procedure:** with the car parked and the
  phone in its holder, tap any preset 3 times ~2 s apart (a `sync` preset is fine) before driving. The merge
  script fits the offset from those taps, then applies it to every later tap.

---

## Bill of materials

Prices are rough Indian retail as seen on Amazon.in on 2026-10-07, not quotes. ✅ marks a specific listing
the user screenshotted and that was checked against the requirement.

### Must buy
| Item | Listing / search term | Price | Notes |
|---|---|---|---|
| MPU-6050 | ✅ **Robocraze GY-521 MPU-6050 Module**, Amazon.in | ₹268 for **3 boards** (₹89.33 each) | **Pins come loose — needs soldering** (below). Two boards are spares |
| USB GPS | ✅ **VK172 G-Mouse USB GPS Dongle (GPS/GLONASS)**, brand IDUINO, Amazon.in | ₹699 | u-blox 7 class. Appears as `/dev/ttyACM0` at 9600 baud, which are the recorder's defaults (**verified on the real Pi 2026-11-02**: NMEA arrives on `/dev/ttyACM0` at 9600; it also emits harmless `$GPTXT ... NMEA unknown msg` lines, and ModemManager is not installed) |
| USB extension cable | `USB extension cable 1.5m male to female` | ~₹100–200 | The Pi sits low by the sensor, but the GPS needs sky view on the dashboard |
| Jumper wires | `female to female jumper wires 20cm` | ~₹60–100 | 4 needed; packs are 40 |
| Mounting tape | `3M VHB double sided foam tape` | ~₹100–200 | Holds the sensor rigidly to the car body |

**Must-buy total ≈ ₹1,350.**

### Only if not already owned (user has not answered yet)
| Item | Search term | Price |
|---|---|---|
| microSD card | `SanDisk 32GB microSD A1` | ₹300–450 |
| microSD reader | `USB microSD card reader` (USB-C for a MacBook Air) | ₹150–300 |
| Power bank | `10000mAh power bank USB-C PD 18W` | ₹800–1,500 |
| USB-C to USB-C cable, 3A | `USB C to C cable 3A` | ₹150–300 |
| Car phone holder | `car dashboard phone holder` | ₹300–600 |
| Phone car charger | `car charger USB-C fast` | ₹300–500 |

### Nice to have
Pi 4 case with an opening over the GPIO header (₹200–400). Electrical tape to hold the jumpers on (₹30).

### Do not buy
Breadboard · soldering kit · NEO-6M board · Pi Camera · male-ended jumper wires.

**Open question to the user (asked 2026-10-07, unanswered):** is there already an SD card in the Pi, a
USB-C power bank, and a car phone holder?

---

## Soldering (the one hardware task)

The Robocraze boards ship with header strips loose. **One board needs its 8-pin header soldered**, about 5
minutes of work. The user is a hardware beginner and does not solder. The plan is to have it done at the
college EPICS/electronics lab, a mobile repair shop (~₹20–50), or an electronics shop.

Instruction to give the solderer: *"8-pin header, pins pointing **down**, on the side **without** the
chip."* Then the chip faces up when the board is mounted flat.

Checking the result: every pin has a small shiny cone of solder, no bridge joins neighbouring pins, and no
pin wiggles.

---

## Wiring, mounting, setup, drive procedure

Pi login (username/password) is in [`local-secrets.md`](local-secrets.md), which is **gitignored** and exists
only on the machine where it was written. Never copy the values into a tracked file.

Kept in one place to avoid drift: **[`field_rig/README.md`](../field_rig/README.md)**. In short:
- **Wiring:** VCC→pin 1 (3.3 V, **never** pin 2 = 5 V), GND→6, SDA→3, SCL→5, with female-female jumpers.
  **On the user's Pi, GND goes to pin 9**: pins 4 and 6 power the case fan. Any GND pin works.
- **The user's board is a clone:** `WHO_AM_I = 0x70` (likely an MPU-6500 sold as MPU-6050). Accel, gyro and
  FIFO registers used by the recorder behave the same; gravity read 9.75 m/s² still, noise well under limits.
  `i2cdetect -y 1` should show `68`.
- **Mounting:** sensor flat, chip up, VHB-taped to the car floor or centre console, never a seat. GPS on
  the dashboard via the extension cable.
- **Pi setup:** enable I2C, raise the I2C bus to 400 kHz, venv with `field_rig/requirements.txt`.
- **Drive:** car still → start the recorder over SSH → wait 3 s for calibration → start Sensor Logger →
  cross a speed breaker early → passenger says "pothole" aloud at each one → stop the phone → Ctrl+C on the Pi.

Recorder design decisions (500 Hz, per-segment clock fit, levelling, empty `label`): see the
2026-10-07 `field_rig/` entry in [`50-changelog.md`](50-changelog.md). Output columns are in
[`20-data-contracts.md`](20-data-contracts.md) §1.

---

## Synchronisation

Pi and phone clocks are independent. Planned alignment, two layers:
1. **Coarse:** the Pi's `unix_time` comes from GPS satellite time (the Pi has no RTC). The phone's
   clock is network-synced. Expected agreement is within about a second.
2. **Fine:** Sensor Logger records the phone's own accelerometer on the same clock as its video.
   Cross-correlating that against the MPU-6050 over the drive's bumps gives the offset. The early speed
   breaker guarantees one strong common spike.

Required precision is loose: frames are chosen 0.5–2.0 s before an event, so ~0.1 s is enough.
**Not built yet.**

---

## Ground truth

There are no tile coordinates on a real road. The plan: a passenger says "pothole" aloud at each one,
captured on the video's audio track, then reviewed. This is weaker than CARLA's exact labels, and:
- any accuracy figure from a field run must state this ground-truth method (Rule 6)
- fine-tuning vision on field frames would need hand-drawn boxes; CARLA labels are computed

---

## Status checklist

| Step | State |
|---|---|
| Recorder `field_rig/pi_recorder.py` | ✅ written, tested on fake hardware only |
| Parts chosen | ✅ MPU-6050 and GPS listings checked |
| Parts ordered / received | ✅ user said on 2026-11-02 they have all components (soldering state not confirmed) |
| Sensor board soldered | ✅ 2026-11-02, **right-angle** header (board lies flat, wires leave sideways) |
| SD / power / holder inventory answered | ⏳ |
| 10 s Sensor Logger test export | ✅ 2026-11-02, format recorded above; camera came out as stills |
| Pi OS flashed, I2C on at 400 kHz, SSH over phone hotspot, `field_rig/` copied, `~/pave-venv` installed | ✅ 2026-11-02 |
| USB GPS talking to the Pi | ✅ 2026-11-02 (no fix yet, indoors) |
| First real `--check` on the Pi | ✅ 2026-11-02: sensor at 0x68 on I2C, GPS 10/10 NMEA; MOVING false alarm traced to a startup-read bug, fixed in `configure()`; **re-run passes** (`mounting looks right`, GPS ok) |
| Merge script (Pi + phone → frames → cascade) | ❌ not started, waiting on the real export |
| First real drive | ❌ |
