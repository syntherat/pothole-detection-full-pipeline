# 13 — PAVE Map UI

**Scope:** `pothole_map_ui/` — the driver-facing dashboard. Google Maps, GPS tracking, proximity alerts, event ingestion.
**Last verified against:** commit `3c586b8` (2026-08-19).

---

## What it is

A single-page, dependency-free dashboard: dark map, live vehicle marker, red pothole markers, and an alert when the driver comes within 100 m of a known pothole. Three files, no build step, no npm, no framework. Open it over HTTP and it runs.

It is also the only part of PAVE that uses **real** GPS — the browser Geolocation API. Everything upstream mocks location.

---

## Files

| File | Role |
|---|---|
| `index.html` | Structure only. No inline logic beyond `onclick` handlers |
| `styles.css` | Complete dark theme, all colours via `:root` custom properties |
| `app.js` | All behaviour |
| `config.example.js` | Template: sets `window.GOOGLE_API_KEY` |
| `config.js` | **Gitignored, you must create it.** Holds the real key |

### Script load order matters
`index.html` loads `config.js` **before** `app.js`, because `app.js` reads `window.GOOGLE_API_KEY` at the bottom of the file and injects the Maps script itself. Reversing the order breaks the map.

> Minor stale comment: the comment block in `index.html` says `app.js` reads `CONFIG.GOOGLE_API_KEY`. It actually reads `window.GOOGLE_API_KEY`, which is what `config.example.js` sets. The code is right; the comment is not.

---

## Configuration constants (top of `app.js`)

| Constant | Value | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | `window.GOOGLE_API_KEY \|\| "API_KEY"` | Maps key |
| `MAP_CENTER` | `{lat: 23.2599, lng: 77.4126}` | Bhopal — fallback centre when GPS is unavailable |
| `PROXIMITY_RADIUS_M` | `100` | Alert radius in metres |
| `PAVE_EVENTS_URL` | `'../integration/pave_events.json'` | Relative to `index.html` |
| `PAVE_POLL_INTERVAL_MS` | `3000` | Poll cadence |
| `MAP_MODE` | `'real'`, or `'carla'` when the URL carries `?mode=carla` | Which basemap to draw -- see **CARLA mode** below |
| `CARLA_TOWN` | `?town=` param, default `'Town03'` | Which exported town CARLA mode loads |
| `CARLA_ROADS_URL` | `../carla_sim/out/<CARLA_TOWN>_roads.geojson` | Road network written by `scenario/export_map.py` |

---

## Runtime state

```js
potholes      // [] — the store. Every detection, in arrival order
mainMap, panelMap
mainMarkers, panelMarkers   // [{marker, ph, iw}]
carMarker     // the vehicle marker
userHeading   // last known heading, retained when a fix omits it
lastAlertedId // suppresses repeat toasts for the same pothole
simCount      // rotates through NEARBY_NAMES
seenEventIds  // Set — dedupes polled events by event_id
```

The store is **in-memory only**. A page refresh loses every pothole; the poll then re-adds them from `pave_events.json` because `seenEventIds` is also reset.

---

## Map initialisation

`initMaps()` is the Maps API callback. It creates two maps — the main map at zoom 14 and the panel map at zoom 13 — with `mapTypeControl`, `streetViewControl` and `fullscreenControl` disabled.

Which style array they get, and what happens next, depends on `MAP_MODE`:

- **`'real'`** (default, and what every existing link does) -- `REAL_STYLE`, the dark Google tile theme with POI and transit hidden, then `startLocationTracking()`.
- **`'carla'`** -- `CARLA_STYLE`, which switches *every* Google feature off and leaves a flat `#0d1117` ground, then `loadCarlaRoads()`. Browser geolocation is deliberately **not** started.

The Maps script is injected by an IIFE at the bottom of `app.js`. If `window.GOOGLE_API_KEY` is missing, it does **not** inject anything; instead it replaces both map containers with a red "config.js not loaded / GOOGLE_API_KEY missing" message. That message is the expected appearance on a fresh clone.

---

## CARLA mode

Added 2026-09-06. Opt-in via URL: `index.html?mode=carla&town=Town03`. Without that parameter nothing about the dashboard changes.

**The problem it solves.** CARLA towns are geo-referenced near (0, 0) -- open ocean in the Gulf of Guinea. Feeding CARLA GNSS into the dashboard puts every pothole marker in the Atlantic on a blank sea tile. The alternative considered and rejected was anchoring a town over a real city, which lands markers inside buildings because CARLA's street grid is not that city's street grid.

**How it works.** `loadCarlaRoads()` fetches the GeoJSON written by `carla_sim/scenario/export_map.py`, adds it to both maps' `map.data` layers, styles junction segments (`#3a4255`) differently from plain road (`#2a3040`), and calls `fitBounds()` using the `properties.bounds` block the exporter ships inside the file. Zoom is fitted rather than guessed because towns differ in size and the default zoom 14 would frame empty ocean beside the network.

**What is deliberately unchanged.** Markers, the Haversine proximity maths, `pollPaveEvents()` and the whole `PotholeGuard` API. Only the basemap layer differs -- that is what keeps this inside Rule 3.

**Known limitation -- no vehicle feed.** In CARLA mode there is no car marker and **proximity alerts never fire**. Browser geolocation would place the vehicle in Bhopal while the roads sit near (0, 0), so it is switched off rather than drawn wrong; the position would have to come from the run's own `gnss.csv`, which is not plumbed to the UI. `checkProximityFromLastKnown()` guards on `if (carMarker)`, so this degrades quietly rather than throwing. Tracked as issue #28.

If the GeoJSON is missing, the catch branch logs loudly and toasts *"CARLA roads missing -- run export_map.py"*, because a silent failure is visually identical to a town with no roads: a blank dark rectangle.

---

## GPS tracking

`startLocationTracking()` uses `navigator.geolocation.watchPosition` with `{enableHighAccuracy: true, maximumAge: 1000, timeout: 10000}`.

On each fix: update the car marker, update the GPS card (5-decimal coords plus `±Nm` accuracy), and run a proximity check.

**The car icon** is generated per-frame as an inline SVG data URI by `carIconSVG(heading)` — a 36×36 top-down car inside a translucent blue circle, rotated by the heading via an SVG `transform`. Because the icon is a data URI, rotation costs a re-encode on every position update; cheap enough at 1 Hz. `userHeading` is retained when a fix reports `heading: null` (common when stationary), so the car does not snap to north.

First fix also pans the map, sets zoom 15, and switches the GPS pill to its active state.

**Geolocation requires a secure context** — `https://` or `localhost`. On a plain `http://` LAN address browsers refuse it.

---

## Proximity alerting

`checkProximity(userLat, userLng)` linearly scans `potholes`, computes Haversine distance to each, and keeps the nearest within `PROXIMITY_RADIUS_M`.

```js
getDistanceMeters(lat1, lng1, lat2, lng2)   // R = 6,371,000 m
```

When something is in range: the proximity card gets `.active` (CSS pulses it red), the distance and location name are shown, and — **only if `nearest.id !== lastAlertedId`** — a toast fires. That guard is what stops a toast storm every second while parked next to a pothole. Leaving the radius resets `lastAlertedId` to `null`, so re-approaching the same pothole alerts again.

Linear scan is fine at prototype scale. At thousands of potholes it would need spatial bucketing.

---

## Adding a pothole

```js
addPothole(lat, lng, locationName, detectedBy)
```

Assigns a sequential id `PH-001`, `PH-002`, … (`potholes.length + 1`, zero-padded to 3), stamps `new Date()`, pushes to the store, then: place a red marker, set the boolean status card, prepend to the live feed, update the badge, show a toast.

**Note the two id schemes:** the UI mints its own `PH-NNN` and ignores the `event_id` UUID from the integration layer. Dedup uses the incoming `event_id`; display uses the local one. They are not the same identifier.

`placeRedDot(map, arr, ph, animate)` creates a `SymbolPath.CIRCLE` marker (red fill, white stroke, scale 10), optionally with a DROP animation, plus an InfoWindow showing id, location, model, timestamp and coordinates.

---

## UI regions

| Region | Elements |
|---|---|
| **Topbar** | `PAVE` logo · MONITORING pill · GPS pill (`GPS ON/OFF`, turns blue when active) · pothole count button with badge · Simulate Detection button |
| **Map** | Full-height dark Google Map with the car marker and red dots |
| **Sidebar** | Detection status card (`TRUE`/`FALSE`, auto-reverting after 4 s) · GPS coordinates card · Proximity alert card · Live event feed (**capped at 10 items**, newest first, click to pan and open the InfoWindow) |
| **History panel** | Full-screen overlay, split list + map. On open it re-renders all markers on the panel map and `fitBounds` to them (inside a 150 ms `setTimeout` so the hidden map has laid out first) |
| **Toast** | Slide-up notification, auto-hides after 3.5 s |

`updateStatus()` sets the boolean card to `TRUE`/`.danger`, then a 4-second timer (stored on `window._rst`, cleared each call) reverts it to `FALSE`/`.safe`.

---

## Integration ingestion — `pollPaveEvents()`

Runs immediately on load and every 3 s thereafter via `setInterval`.

```js
fetch('../integration/pave_events.json', { cache: 'no-store' })
  → if !res.ok: return quietly (file may not exist yet — normal)
  → for each event not in seenEventIds:
      seenEventIds.add(evt.event_id)
      PotholeGuard.reportDetection(
        evt.lat, evt.lng,
        `Hybrid Detection (conf ${evt.confidence.toFixed(2)})`,
        'Both')
```

Design notes:

- **`cache: 'no-store'`** is required — without it the browser serves a stale copy and new events never appear.
- **A 404 is not an error.** The file only exists after the first confirmed event.
- **Must be served over HTTP.** On `file://` the fetch is a CORS failure and the catch logs a warning every 3 s. This is the single most common setup mistake.
- The whole file is re-fetched and re-scanned every tick. Fine for a demo, O(n) forever in principle.
- `evt.confidence.toFixed(2)` assumes `confidence` is a number — a null would throw inside the loop and abort the remaining events in that batch.

---

## Public API — `window.PotholeGuard`

The sanctioned way for any external source to inject detections (Rule 3).

```js
window.PotholeGuard.reportDetection(lat, lng, locationName = 'Road', model = 'Image Model');
window.PotholeGuard.getPotholes();   // the store array
window.PotholeGuard.isDetected();    // potholes.length > 0
```

Callable from the browser console — the fastest way to test the UI without running any Python.

`MODELS = ['Image Model', 'Math Model', 'Both']` is the vocabulary shown in the UI for detection provenance: vision-only, sensor-only, and fused. The integration poller always reports `'Both'`.

---

## Simulation

`simulateDetection()` drops a pothole at a random offset of up to ±0.003 deg (roughly ±330 m) from the current car position — or `MAP_CENTER` if there is no fix — names it from the rotating `NEARBY_NAMES` list, picks a random model label, pans there, and re-runs the proximity check. It is the demo button, and it works with no backend at all.

---

## Theme

Every colour and font flows from `:root` in `styles.css`:

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0a0c10` | Page background |
| `--surface` | `#111318` | Cards, bars |
| `--surface2` | `#181c24` | Inner cards |
| `--border` | `#1e2430` | Borders |
| `--accent` | `#f0a500` | Amber highlights, simulate button |
| `--text` | `#e8eaf0` | Primary text |
| `--muted` | `#6b7280` | Secondary text |
| `--danger` | `#ff3b3b` | Alerts, pothole markers |
| `--safe` | `#00e676` | Safe state |
| `--font-head` | Rajdhani | Display |
| `--font-mono` | JetBrains Mono | Data and coordinates |

Fonts come from Google Fonts via a `<link>` in `index.html`. Retheming means editing these nine values — **do not hardcode hex values elsewhere** (Rule 9).

---

## Setup and gotchas

1. `cp config.example.js config.js` and paste a real key.
2. Enable **Maps JavaScript API** in Google Cloud Console, with billing active.
3. Restrict the key by HTTP referrer (`localhost/*`, `127.0.0.1/*`) — the key ships to the browser and is inherently public.
4. Serve over HTTP: VS Code Live Server, or `python -m http.server` from the **repo root** (not from `pothole_map_ui/`, or the `../integration/` path escapes the served tree).

| Symptom | Cause |
|---|---|
| Red "GOOGLE_API_KEY missing" box | `config.js` not created |
| Dark background, no tiles | Check console: `ApiNotActivatedMapError`, `RefererNotAllowedMapError`, or `InvalidKeyMapError` |
| `Could not poll pave_events.json` every 3 s | Opened via `file://`, or the relative path does not resolve from the served root |
| GPS never locks | Not on `https`/`localhost`, or permission denied |
| Markers appear but no proximity alert | Nothing within 100 m; raise `PROXIMITY_RADIUS_M` to test |
| Everything disappears on refresh | Expected — the store is in-memory |
