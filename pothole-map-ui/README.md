# PAVE Map UI

A real-time pothole detection dashboard built for in-car use. This module handles the Google Maps interface layer of the PotholeGuard system — displaying live pothole detections as red markers on the map, maintaining a detection log, and providing a full pothole history panel.

This UI connects to two detection models running in the backend:
- **Image Model** — detects potholes using computer vision on road-facing camera feed
- **Math Model** — detects potholes using accelerometer data and vehicle motion analysis

---

## Folder Structure

```
potholE-map-ui/
└── index.html        # Single-file app — all HTML, CSS, and JS included
```

---

## Features

- **Live Boolean Detection Card** — displays TRUE/FALSE in real time when a pothole is detected, resets automatically after 4 seconds
- **Red Dot Markers on Google Maps** — every detected pothole is plotted as a red circle on the map at its exact GPS coordinates
- **Clickable Markers** — click any dot to see pothole ID, location name, detection model, timestamp, and coordinates
- **Potholes Detected Button** — top bar button that turns red and pulses when potholes exist; shows a live count badge
- **Full Pothole History Panel** — opens a split-screen view with a scrollable list of all potholes on the left and a full map with all dots on the right
- **Live Feed Sidebar** — shows the last 10 detection events with location and timestamp
- **Simulate Detection** — test button that drops random potholes on real Bhopal road coordinates for demo/testing purposes

---

## Setup

**1. Get a Google Maps API Key**

- Go to [console.cloud.google.com](https://console.cloud.google.com)
- Create a project and enable the **Maps JavaScript API**
- Go to APIs & Services → Credentials → Create API Key
- Make sure billing is enabled on your project (required by Google even on free tier)

**2. Add your API key**

Open `index.html` and find this line near the bottom:

```js
const GOOGLE_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY";
```

Replace it with your actual key:

```js
const GOOGLE_API_KEY = "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
```

**3. Open the file**

Open `index.html` directly in a browser, or serve it through your existing project server.

---

## Connecting Your Detection Models

The UI exposes a global API so your backend models can push detections directly into the map:

```js
// Report a pothole from your model
window.PotholeGuard.reportDetection(lat, lng, locationName, model);

// Examples:
window.PotholeGuard.reportDetection(23.2599, 77.4126, "DB City Mall Road", "Image Model");
window.PotholeGuard.reportDetection(23.2555, 77.4022, "Arera Colony Road", "Math Model");
window.PotholeGuard.reportDetection(23.2710, 77.4300, "Habibganj Railway Rd", "Both");

// Check current detection state
window.PotholeGuard.isDetected();   // → true / false
window.PotholeGuard.getPotholes();  // → array of all logged potholes
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `lat` | number | Latitude of detected pothole |
| `lng` | number | Longitude of detected pothole |
| `locationName` | string | Human-readable road/area name |
| `model` | string | `"Image Model"`, `"Math Model"`, or `"Both"` |

---

## Map Location

Default map center is set to **Bhopal, Madhya Pradesh, India**.
To change it, find this line in `index.html`:

```js
const MAP_CENTER = { lat: 23.2599, lng: 77.4126 };
```

Replace with any coordinates you need.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | HTML5, CSS3, Vanilla JavaScript |
| Maps | Google Maps JavaScript API |
| Fonts | Rajdhani, JetBrains Mono (Google Fonts) |
| Deployment | Single HTML file — no build step required |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Map not showing, just dark background | Check browser Console (F12) for the exact Maps error |
| `ApiNotActivatedMapError` | Enable Maps JavaScript API in Google Cloud Console |
| `RefererNotAllowedMapError` | Add your domain or `localhost/*` to the API key's allowed referrers |
| `InvalidKeyMapError` | Re-copy the key — no extra spaces or characters |
| `styles.css` or `index.js` 404 errors | Remove those `<link>` / `<script>` tags from your project's main `index.html` |
| Map loads but no dots appear | Call `simulateDetection()` in the browser console to test |

---
