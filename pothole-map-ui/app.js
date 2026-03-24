// ── POTHOLEGUARD — app.js ──

const GOOGLE_API_KEY = "AIzaSyAm6wJnu_CrdP7cPn6dQu26SYglZ71gxTU"; // ← replace with your key
const MAP_CENTER = { lat: 23.2599, lng: 77.4126 };  // ← Bhopal default
const PROXIMITY_RADIUS_M = 100;                      // ← alert radius in meters

// ── STATE ────────────────────────────────────────────────────────────────
let potholes     = [];
let mainMap      = null;
let panelMap     = null;
let mainMarkers  = [];
let panelMarkers = [];
let carMarker    = null;
let userHeading  = 0;
let toastTimer   = null;
let lastAlertedId = null;
let simCount     = 0;

const MODELS = ['Image Model', 'Math Model', 'Both'];
const NEARBY_NAMES = [
  "Main Road", "Junction Ahead", "Near Crossroad", "Side Street",
  "Highway Stretch", "Colony Road", "Market Road", "Bypass Road",
  "Inner Road", "Outer Ring", "Service Lane", "Bridge Approach",
  "School Zone Road", "Industrial Road", "Old City Road"
];

// ── MAPS INIT ─────────────────────────────────────────────────────────────
function initMaps() {
  const style = [
    { elementType: 'geometry', stylers: [{ color: '#1a1f2e' }] },
    { elementType: 'labels.text.stroke', stylers: [{ color: '#0a0c10' }] },
    { elementType: 'labels.text.fill', stylers: [{ color: '#6b7280' }] },
    { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#2a3040' }] },
    { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#3a4255' }] },
    { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#9ca3af' }] },
    { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0d1117' }] },
    { featureType: 'poi', stylers: [{ visibility: 'off' }] },
    { featureType: 'transit', stylers: [{ visibility: 'off' }] },
    { featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{ color: '#9ca3af' }] },
  ];
  const opts = {
    zoom: 14, center: MAP_CENTER, styles: style,
    mapTypeControl: false, streetViewControl: false, fullscreenControl: false
  };
  mainMap  = new google.maps.Map(document.getElementById('google-map'), opts);
  panelMap = new google.maps.Map(document.getElementById('panel-google-map'), { ...opts, zoom: 13 });
  startLocationTracking();
}

// ── CAR MARKER ────────────────────────────────────────────────────────────
function carIconSVG(heading) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
      <circle cx="18" cy="18" r="16" fill="rgba(100,180,255,0.12)" stroke="rgba(100,180,255,0.4)" stroke-width="1.5"/>
      <g transform="rotate(${heading}, 18, 18)">
        <rect x="13" y="8" width="10" height="20" rx="4" fill="#64b4ff"/>
        <rect x="14.5" y="10" width="7" height="5" rx="1.5" fill="#0a1828" opacity="0.8"/>
        <rect x="14.5" y="21" width="7" height="4" rx="1" fill="#0a1828" opacity="0.6"/>
        <rect x="10" y="11" width="3.5" height="5" rx="1.5" fill="#1e2430"/>
        <rect x="10" y="20" width="3.5" height="5" rx="1.5" fill="#1e2430"/>
        <rect x="22.5" y="11" width="3.5" height="5" rx="1.5" fill="#1e2430"/>
        <rect x="22.5" y="20" width="3.5" height="5" rx="1.5" fill="#1e2430"/>
        <rect x="14.5" y="8.5" width="3" height="2" rx="0.5" fill="#fffde0" opacity="0.9"/>
        <rect x="18.5" y="8.5" width="3" height="2" rx="0.5" fill="#fffde0" opacity="0.9"/>
        <polygon points="18,9 16,13 18,12 20,13" fill="#ffffff" opacity="0.9"/>
      </g>
    </svg>`;
  return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
}

// ── GPS / LOCATION TRACKING ───────────────────────────────────────────────
function startLocationTracking() {
  if (!navigator.geolocation) {
    document.getElementById('gps-coords').textContent = 'GPS not supported';
    return;
  }
  navigator.geolocation.watchPosition(
    (pos) => {
      const lat = pos.coords.latitude;
      const lng = pos.coords.longitude;
      const heading = pos.coords.heading || userHeading;
      if (pos.coords.heading !== null) userHeading = pos.coords.heading;
      updateCarMarker(lat, lng, heading);
      updateGPSDisplay(lat, lng, pos.coords.accuracy);
    },
    (err) => {
      document.getElementById('gps-coords').textContent = 'GPS unavailable';
      document.getElementById('gps-label').textContent = 'GPS OFF';
      console.warn('Geolocation error:', err.message);
    },
    { enableHighAccuracy: true, maximumAge: 1000, timeout: 10000 }
  );
}

function updateCarMarker(lat, lng, heading) {
  if (!mainMap) return;
  const pos = { lat, lng };
  if (!carMarker) {
    carMarker = new google.maps.Marker({
      position: pos, map: mainMap, title: 'Your Vehicle',
      icon: { url: carIconSVG(heading), scaledSize: new google.maps.Size(36, 36), anchor: new google.maps.Point(18, 18) },
      zIndex: 999,
    });
    mainMap.panTo(pos);
    mainMap.setZoom(15);
    document.getElementById('gps-pill').classList.add('active');
    document.getElementById('gps-label').textContent = 'GPS ON';
  } else {
    carMarker.setPosition(pos);
    carMarker.setIcon({ url: carIconSVG(heading), scaledSize: new google.maps.Size(36, 36), anchor: new google.maps.Point(18, 18) });
  }
}

function updateGPSDisplay(lat, lng, accuracy) {
  document.getElementById('gps-coords').textContent =
    `${lat.toFixed(5)}, ${lng.toFixed(5)}\n±${Math.round(accuracy)}m`;
  checkProximity(lat, lng);
}

// ── PROXIMITY ─────────────────────────────────────────────────────────────
function checkProximity(userLat, userLng) {
  let nearest = null;
  let nearestDist = Infinity;

  potholes.forEach(ph => {
    const dist = getDistanceMeters(userLat, userLng, ph.lat, ph.lng);
    if (dist < PROXIMITY_RADIUS_M && dist < nearestDist) {
      nearestDist = dist;
      nearest = ph;
    }
  });

  const card   = document.getElementById('proximity-card');
  const distEl = document.getElementById('proximity-dist');
  const nameEl = document.getElementById('proximity-name');

  if (nearest) {
    card.classList.add('active');
    distEl.textContent = `${Math.round(nearestDist)}m away`;
    nameEl.textContent = nearest.locationName;
    if (lastAlertedId !== nearest.id) {
      lastAlertedId = nearest.id;
      showToast(`⚠ Pothole ahead — ${Math.round(nearestDist)}m · ${nearest.locationName}`);
    }
  } else {
    card.classList.remove('active');
    distEl.textContent = 'No potholes nearby';
    nameEl.textContent = `Radius: ${PROXIMITY_RADIUS_M}m`;
    lastAlertedId = null;
  }
}

function checkProximityFromLastKnown() {
  if (carMarker) {
    const pos = carMarker.getPosition();
    checkProximity(pos.lat(), pos.lng());
  }
}

// Haversine formula — real-world distance in meters
function getDistanceMeters(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
            Math.sin(dLng/2) * Math.sin(dLng/2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ── POTHOLE CORE ──────────────────────────────────────────────────────────
function addPothole(lat, lng, locationName, detectedBy) {
  const id = `PH-${String(potholes.length + 1).padStart(3,'0')}`;
  const ts = new Date();
  const ph = { id, lat, lng, locationName, detectedBy, timestamp: ts };
  potholes.push(ph);
  if (mainMap) placeRedDot(mainMap, mainMarkers, ph, true);
  updateStatus(ph);
  addToList(ph);
  updateBadge();
  showToast(`${id} — ${locationName}`);
}

function placeRedDot(map, arr, ph, animate) {
  const marker = new google.maps.Marker({
    position: { lat: ph.lat, lng: ph.lng },
    map,
    title: ph.id,
    animation: animate ? google.maps.Animation.DROP : null,
    icon: {
      path: google.maps.SymbolPath.CIRCLE,
      fillColor: '#ff3b3b',
      fillOpacity: 1,
      strokeColor: '#ffffff',
      strokeWeight: 2,
      scale: 10,
    },
  });
  const iw = new google.maps.InfoWindow({
    content: `<div style="font-family:'JetBrains Mono',monospace;background:#111318;color:#e8eaf0;padding:10px;border-radius:6px;font-size:12px;">
      <b style="color:#ff3b3b;">${ph.id}</b><br>${ph.locationName}<br>
      <span style="color:#6b7280;">Model: ${ph.detectedBy}<br>${formatTime(ph.timestamp)}<br>${ph.lat.toFixed(5)}, ${ph.lng.toFixed(5)}</span>
    </div>`,
  });
  marker.addListener('click', () => iw.open(map, marker));
  arr.push({ marker, ph, iw });
}

// ── UI UPDATES ────────────────────────────────────────────────────────────
function updateStatus(ph) {
  document.getElementById('detection-status').textContent = 'TRUE';
  document.getElementById('detection-status').className = 'danger';
  document.getElementById('detection-sub').textContent = `${ph.id} · ${ph.locationName}`;
  clearTimeout(window._rst);
  window._rst = setTimeout(() => {
    document.getElementById('detection-status').textContent = 'FALSE';
    document.getElementById('detection-status').className = 'safe';
    document.getElementById('detection-sub').textContent = 'No pothole in current zone';
  }, 4000);
}

function addToList(ph) {
  const list  = document.getElementById('live-list');
  const empty = list.querySelector('div[style]');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = 'pothole-item';
  item.innerHTML = `
    <div class="ph-id">${ph.id} · ${ph.detectedBy}</div>
    <div class="ph-loc">${ph.locationName}</div>
    <div class="ph-time">${formatTime(ph.timestamp)}</div>`;
  item.onclick = () => {
    if (!mainMap) return;
    mainMap.panTo({ lat: ph.lat, lng: ph.lng });
    mainMap.setZoom(16);
    const e = mainMarkers.find(m => m.ph.id === ph.id);
    if (e) e.iw.open(mainMap, e.marker);
  };
  list.insertBefore(item, list.firstChild);
  while (list.children.length > 10) list.removeChild(list.lastChild);
}

function updateBadge() {
  document.getElementById('badge-count').textContent = potholes.length;
  document.getElementById('btn-potholes').classList.toggle('has-potholes', potholes.length > 0);
}

function showToast(msg) {
  document.getElementById('toast-msg').textContent = `Pothole: ${msg}`;
  document.getElementById('toast').classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => document.getElementById('toast').classList.remove('show'), 3500);
}

// ── POTHOLE PANEL ─────────────────────────────────────────────────────────
function openPotholePanel() {
  document.getElementById('pothole-panel').classList.add('open');
  document.getElementById('panel-count').textContent =
    `${potholes.length} pothole${potholes.length !== 1 ? 's' : ''}`;

  const list = document.getElementById('panel-ph-list');
  list.innerHTML = potholes.length === 0
    ? '<div style="font-size:12px;font-family:var(--font-mono);color:var(--muted);padding:10px 0;">No potholes detected yet.</div>'
    : '';

  [...potholes].reverse().forEach(ph => {
    const item = document.createElement('div');
    item.className = 'panel-ph-item';
    item.innerHTML = `
      <div class="panel-ph-name">${ph.locationName}</div>
      <div class="panel-ph-meta">${ph.id} · ${ph.detectedBy}</div>
      <div class="panel-ph-meta">${ph.lat.toFixed(5)}, ${ph.lng.toFixed(5)}</div>`;
    item.onclick = () => {
      document.querySelectorAll('.panel-ph-item').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
      if (!panelMap) return;
      panelMap.panTo({ lat: ph.lat, lng: ph.lng });
      panelMap.setZoom(16);
      const e = panelMarkers.find(m => m.ph.id === ph.id);
      if (e) e.iw.open(panelMap, e.marker);
    };
    list.appendChild(item);
  });

  setTimeout(() => {
    if (!panelMap) return;
    google.maps.event.trigger(panelMap, 'resize');
    panelMarkers.forEach(m => m.marker.setMap(null));
    panelMarkers = [];
    potholes.forEach(ph => placeRedDot(panelMap, panelMarkers, ph, false));
    if (potholes.length > 0) {
      const b = new google.maps.LatLngBounds();
      potholes.forEach(ph => b.extend({ lat: ph.lat, lng: ph.lng }));
      panelMap.fitBounds(b, { top: 50, right: 50, bottom: 50, left: 50 });
    }
  }, 150);
}

function closePotholePanel() {
  document.getElementById('pothole-panel').classList.remove('open');
}

// ── SIMULATE ──────────────────────────────────────────────────────────────
function getUserPosition() {
  if (carMarker) {
    const pos = carMarker.getPosition();
    return { lat: pos.lat(), lng: pos.lng() };
  }
  return MAP_CENTER;
}

function simulateDetection() {
  const base = getUserPosition();
  const lat  = base.lat + (Math.random() - 0.5) * 0.006;
  const lng  = base.lng + (Math.random() - 0.5) * 0.006;
  const name = NEARBY_NAMES[simCount % NEARBY_NAMES.length];
  simCount++;
  addPothole(lat, lng, name, MODELS[Math.floor(Math.random() * MODELS.length)]);
  if (mainMap) { mainMap.panTo({ lat, lng }); mainMap.setZoom(15); }
  checkProximityFromLastKnown();
}

// ── EXTERNAL API ──────────────────────────────────────────────────────────
// Connect your detection models via this global object
window.PotholeGuard = {
  reportDetection: (lat, lng, locationName = 'Road', model = 'Image Model') =>
    addPothole(lat, lng, locationName, model),
  getPotholes:  () => potholes,
  isDetected:   () => potholes.length > 0,
};

// ── HELPERS ───────────────────────────────────────────────────────────────
function formatTime(d) {
  return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── LOAD GOOGLE MAPS ──────────────────────────────────────────────────────
(function() {
  if (GOOGLE_API_KEY === "YOUR_GOOGLE_MAPS_API_KEY") {
    ['google-map', 'panel-google-map'].forEach(id => {
      const el = document.getElementById(id);
      el.style.cssText = 'display:flex;align-items:center;justify-content:center;background:#13181f;';
      el.innerHTML = `<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#2a3040;text-align:center;letter-spacing:1px;">SET YOUR GOOGLE MAPS API KEY<br>in app.js</div>`;
    });
    return;
  }
  const s = document.createElement('script');
  s.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_API_KEY}&callback=initMaps`;
  s.async = true;
  s.defer = true;
  document.head.appendChild(s);
})();