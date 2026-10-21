# field_rig — record a real drive

A Raspberry Pi 4 with an **MPU-6050** (shaking) and a **USB GPS** (position, speed, true time)
records the sensor half of a drive. A phone running **Sensor Logger** records the video. The two
are lined up afterwards.

> **Status:** written 2026-10-07 and tested only against **simulated** hardware (a fake MPU-6050
> and fake GPS). It has never run on a real Pi. Treat the first real `--check` as the real test.

## What you need

- Raspberry Pi 4, SD card with **Raspberry Pi OS (64-bit)**, 5V 3A power
- MPU-6050 (GY-521) **with its 8-pin header soldered on** (most boards ship with the pins loose — get
  them soldered pins-down on the side without the chip), plus 4 female-to-female jumper wires
- USB GPS dongle (u-blox 7 / VK-172 type) and a 1.5 m USB extension cable, so the GPS can sit on the
  dashboard while the Pi stays low
- Android phone with Sensor Logger

## Wiring

| MPU-6050 | Pi pin |
|---|---|
| VCC | **1** (3.3V — **not** pin 2, which is 5V) |
| GND | 6 |
| SDA | 3 |
| SCL | 5 |

Leave XDA, XCL, AD0, INT empty. Plug the GPS into any USB port, through the extension cable.

**Mount the MPU-6050 flat, chip side up, taped hard to the car floor or centre console.** Not
on a seat. The GPS goes on the dashboard where it can see the sky.

## One-time Pi setup

When flashing the SD card with Raspberry Pi Imager, set a username and password, turn on SSH, and
add your **phone's hotspot** as the Wi-Fi network. That lets you control the Pi from the phone in
the car.

Then, on the Pi:

```bash
sudo raspi-config nonint do_i2c 0
```
```bash
echo "dtparam=i2c_arm_baudrate=400000" | sudo tee -a /boot/firmware/config.txt
```
```bash
sudo apt install -y i2c-tools python3-venv
```
```bash
sudo reboot
```

The second line raises the I2C speed from 100 to 400 kHz. At 100 kHz, 500 samples a second uses
most of the bus, and the recorder will print `FIFO overflow` warnings.

Copy this `field_rig/` folder to the Pi (USB stick, or `scp -r field_rig <user>@<pi>:~/`), then:

```bash
python3 -m venv ~/pave-venv
```
```bash
~/pave-venv/bin/pip install -r ~/field_rig/requirements.txt
```

## Test before every first drive

```bash
i2cdetect -y 1
```
A `68` in the grid means the sensor is wired right.

```bash
~/pave-venv/bin/python ~/field_rig/pi_recorder.py --check
```
It reads for 3 s and tells you, in words, whether the sensor is wired, flat, the right way up and
still, and whether the GPS is talking. Fix anything marked `[!!]`.

## Recording a drive

1. **Park. Engine can idle, car must not move.**
2. On the phone, SSH into the Pi (e.g. with Termux or JuiceSSH) and start:
   ```bash
   ~/pave-venv/bin/python ~/field_rig/pi_recorder.py
   ```
3. Wait for `calibration done -- you can drive.` (3 seconds).
4. **Switch to Sensor Logger and start recording.** It must stay on screen: Android stops its
   video if the app goes to the background. That is why the Pi is started *first*.
5. Try to cross one speed breaker in the first minute. Lining up the two recordings matches the
   bumps that both the phone and the Pi felt, and one big early bump makes that easy.
6. Drive. Have a passenger say "pothole" out loud at each one; the video records the audio.
7. Stop Sensor Logger. SSH back in and press **Ctrl+C**. The recorder then builds `sensors.csv`.

If the Pi loses power instead, nothing much is lost. The raw files are saved every second, and
this rebuilds `sensors.csv` from them on the Pi or on a laptop:

```bash
python3 field_rig/pi_recorder.py --finalize field_rig/out/run_<timestamp>
```

## What comes out — `out/run_<timestamp>/`

| File | What it is |
|---|---|
| `sensors.csv` | **What the pipeline reads.** `timestamp,ax,ay,az,gx,gy,gz,speed,label` (the same columns as the synthetic and CARLA data) plus `gps_fix,unix_time` |
| `gnss.csv` | One row per GPS fix: position, speed, satellite time |
| `imu_raw.csv`, `imu_clock.csv` | Untouched sensor counts and read times; `sensors.csv` is rebuilt from these |
| `run_meta.json` | Settings, calibration, measured sample rate, warnings |

Things that are different from the synthetic data on purpose:

- **500 samples a second, not 400.** The MPU-6050 cannot make exactly 400 with its noise filter
  on. The detector works from timestamps, so this does not matter to it.
- **`label` is empty.** Which moments were potholes is only known after watching the video.
- **`speed` is 0 where `gps_fix` is 0** (no fix yet, or none for over 3 s). That is "unknown",
  not "stopped".
- **`az` is levelled.** The still period at the start measures which way is down, and every
  sample is rotated so that down is `+az`. Turn this off with `--no-level`.
