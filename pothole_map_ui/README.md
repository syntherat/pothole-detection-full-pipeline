# PAVE Map UI

Real-time pothole detection dashboard for in-car use. Plots detected potholes as red dots on a live Google Maps interface, tracks your GPS location, and alerts you when a pothole is within 100m of your vehicle.

---

## Folder Structure

```
pothole-map-ui/
├── index.html    — HTML structure and layout
├── styles.css    — All styling, theme, and animations
└── app.js        — All logic: maps, GPS, proximity, detection
```

---

## File Breakdown

### `index.html`
The entry point of the app. Contains only the HTML skeleton — no inline CSS or JavaScript. Links to `styles.css` and `app.js`.

**What's inside:**
- Top bar with logo, monitoring status pill, GPS status pill, "Potholes Detected" button, and Simulate button
- Main layout with the Google Maps container on the left and the sidebar on the right
- Sidebar with four sections: boolean detection card, GPS coordinates card, proximity alert card, and live event feed
- Pothole history panel overlay (hidden by default, opens when "Potholes Detected" is clicked) with a split list + map view
- Toast notification element at the bottom

**To open the app:** open `index.html` in a browser or serve it through a local server.

---

### `styles.css`
All visual styling for the app. Uses CSS custom properties (variables) defined in `:root` so the entire theme can be changed from one place.

**Theme variables:**
```css
--bg          /* main background       #0a0c10 */
--surface     /* card/bar background   #111318 */
--surface2    /* inner card background #181c24 */
--border      /* border color          #1e2430 */
--accent      /* amber highlight       #f0a500 */
--danger      /* red alert color       #ff3b3b */
--safe        /* green safe color      #00e676 */
--font-head   /* Rajdhani              display font  */
--font-mono   /* JetBrains Mono        data/code font */
```

**Key sections in the file:**
| Section | What it styles |
|---|---|
| `#topbar` | Top navigation bar and all its pills and buttons |
| `#sidebar` | Right panel container |
| `#detection-card` | Boolean TRUE/FALSE status card |
| `#gps-card` | Live GPS coordinates display |
| `#proximity-card` | Pothole nearby alert — pulses red when active |
| `.pothole-item` | Individual entries in the live event feed |
| `#pothole-panel` | Full-screen history overlay |
| `.panel-ph-item` | Individual entries in the history panel list |
| `#toast` | Slide-up notification at the bottom of the screen |

**Animations defined:**
- `pulse` — green monitoring dot
- `gpsPulse` — blue GPS dot when active
- `borderPulse` — red border on "Potholes Detected" button
- `proximityPulse` — red glow on proximity card when a pothole is nearby

---

### `app.js`
All application logic. This is the only file you need to edit to configure the app.

**Configuration (top of file):**
```js
const GOOGLE_API_KEY    = "YOUR_GOOGLE_MAPS_API_KEY"; // ← replace this
const MAP_CENTER        = { lat: 23.2599, lng: 77.4126 }; // ← default map center (Bhopal)
const PROXIMITY_RADIUS_M = 100; // ← alert radius in meters
```

**Functions overview:**

| Function | Description |
|---|---|
| `initMaps()` | Initializes both the main map and the panel map with dark styling. Called automatically by Google Maps API after load |
| `startLocationTracking()` | Starts `watchPosition` to continuously track the device GPS |
| `updateCarMarker(lat, lng, heading)` | Places or moves the blue car icon on the main map. Rotates the icon based on heading direction |
| `updateGPSDisplay(lat, lng, accuracy)` | Updates the GPS card in the sidebar and triggers proximity check |
| `checkProximity(userLat, userLng)` | Runs the Haversine formula against all stored potholes. Activates the proximity card and fires a toast if a pothole is within `PROXIMITY_RADIUS_M` |
| `getDistanceMeters(lat1, lng1, lat2, lng2)` | Haversine formula — returns accurate real-world distance in meters between two coordinates |
| `addPothole(lat, lng, locationName, detectedBy)` | Core function — adds a pothole to the data store, places a red dot on the map, updates all UI elements |
| `placeRedDot(map, arr, ph, animate)` | Places a red circle marker on a given map with a clickable info window |
| `updateStatus(ph)` | Sets the boolean card to TRUE (red) and resets to FALSE after 4 seconds |
| `addToList(ph)` | Prepends a new entry to the live event feed in the sidebar, keeps max 10 items |
| `updateBadge()` | Updates the count badge on the "Potholes Detected" button |
| `showToast(msg)` | Shows the slide-up toast notification for 3.5 seconds |
| `openPotholePanel()` | Opens the full-screen history panel and re-renders all markers on the panel map |
| `closePotholePanel()` | Closes the history panel |
| `simulateDetection()` | Drops a random pothole within ~300m of your current GPS position for testing |
| `getUserPosition()` | Returns current car marker position, or falls back to `MAP_CENTER` if GPS hasn't locked yet |

---

## Setup

**1. Enable Maps JavaScript API**

Go to [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Library → enable **Maps JavaScript API**. Make sure billing is active on your project.

**2. Add your API key**

Open `app.js` and replace line 3:
```js
const GOOGLE_API_KEY = "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
```

**3. Restrict your key (for public repos)**

Go to Credentials → your key → HTTP Referrers → add:
```
localhost/*
localhost:5500/*
127.0.0.1/*
```
This prevents anyone who sees the key in your public repo from using it on their own project.

**4. Run the app**

Open `index.html` directly in a browser, or serve through VS Code Live Server, or any local HTTP server.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Map shows dark background, no tiles | Check Console (F12) for the exact Google Maps error |
| `ApiNotActivatedMapError` | Enable Maps JavaScript API in Google Cloud Console |
| `RefererNotAllowedMapError` | Add `localhost/*` to your API key's allowed HTTP referrers |
| `InvalidKeyMapError` | Re-copy the key — check for extra spaces or characters |
| `styles.css 404` | Make sure all 3 files are in the same folder |
| Map loads but no red dots | Click "+ Simulate Detection" to test, or call `window.PotholeGuard.reportDetection()` from the console |
| GPS not locking | Allow location permission in the browser when prompted. Use HTTPS or localhost — GPS is blocked on plain HTTP |
| Proximity card not activating | GPS must be active and a pothole must be within 100m. Increase `PROXIMITY_RADIUS_M` in `app.js` for testing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 |
| Styling | CSS3 with custom properties |
| Logic | Vanilla JavaScript (ES6+) |
| Maps | Google Maps JavaScript API |
| Location | Browser Geolocation API (`watchPosition`) |
| Fonts | Rajdhani + JetBrains Mono via Google Fonts |
| Build | None — plain files, no bundler required |

