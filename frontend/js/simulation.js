// Same-origin in production (the API serves this folder). The local launcher runs
// the UI on :8080 and the API on :8000, so point at :8000 when the ports differ.
const API = (location.port && location.port !== '8000') ? `${location.protocol}//${location.hostname}:8000` : '';
const state = {
  vessels: [],
  entities: new Map(),
  motion: new Map(),
  ports: [],
  portEntities: [],
  land: null,
  running: false,
  timer: null,
  stepInFlight: false,
  lastStepDurationMs: 1000,
  motionDurationMs: 1000,
  clockMotion: null,
  tickCount: 0,
  selected: null,
  cinema: false,
  speed: 1,
  filter: 'all',
  searchQuery: '',
  trackHistory: new Map(),
  traceEntity: null,
  vectorEntity: null,
  lastSnapshot: null,
  traceScrubbing: false,
  weatherVisible: false,
  weatherField: [],
  weatherGrid: null,
  weatherFieldCanvas: null,
  weatherFieldCtx: null,
  weatherLayer: 'waves',
  weatherParticles: [],
  weatherFrame: null,
  weatherProjectionDirty: true,
  weatherBaseDirty: true,
  weatherLastBaseRender: 0,
  weatherTime: 0,
  weatherSeed: 20260922,
  weatherFieldSeed: null,
  weatherFieldHours: 0,
  weatherFieldBucket: null,
  weatherRequestTimer: null,
  weatherRequestId: 0,
  weatherOpacity: 0.75,
  weatherParticlesEnabled: true,
  simHours: 0,
  scenarioConfig: null,
  controller: 'bootstrap heading heuristic'
};

const $ = id => document.getElementById(id);

const viewer = new Cesium.Viewer('globe', {
  animation: false,
  timeline: false,
  baseLayerPicker: false,
  fullscreenButton: false,
  geocoder: false,
  homeButton: false,
  infoBox: false,
  navigationHelpButton: false,
  sceneModePicker: false,
  selectionIndicator: false,
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  imageryProvider: new Cesium.OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' })
});

const baseImagery = viewer.imageryLayers.get(0);
if (baseImagery) {
  baseImagery.brightness = 0.78;
  baseImagery.saturation = 0.72;
  baseImagery.contrast = 1.08;
}

viewer.scene.screenSpaceCameraController.maximumZoomDistance = 19000000;
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#030712');
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.showGroundAtmosphere = true;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#061026');
viewer.scene.globe.depthTestAgainstTerrain = false;
viewer.scene.skyAtmosphere.show = true;
viewer.scene.skyAtmosphere.hueShift = -0.05;
viewer.scene.skyAtmosphere.saturationShift = 0.12;
viewer.scene.skyAtmosphere.brightnessShift = 0.08;
viewer.scene.globe.undergroundColor = Cesium.Color.fromCssColorString('#030712');

const defaultView = {
  destination: Cesium.Cartesian3.fromDegrees(68, 12, 19000000),
  orientation: { heading: 0, pitch: Cesium.Math.toRadians(-90), roll: 0 }
};
viewer.camera.setView(defaultView);

/* ── Smooth Per-Second Dead-Reckoning Render Loop ── */
viewer.scene.preRender.addEventListener(() => {
  const now = performance.now();
  if (state.clockMotion) {
    const t = clamp((now - state.clockMotion.started) / state.clockMotion.duration);
    $('simTime').textContent = `${(state.clockMotion.from + (state.clockMotion.to - state.clockMotion.from) * t).toFixed(1)} h`;
    if (t >= 1) state.clockMotion = null;
  }
  for (const [id, m] of state.motion) {
    const e = state.entities.get(id);
    if (!e) {
      state.motion.delete(id);
      continue;
    }
    const elapsed = now - m.started;
    const t = elapsed / m.duration;
    if (t <= 1.0) {
      const scaled = t * m.segments.length;
      const index = Math.min(m.segments.length - 1, Math.floor(scaled));
      const localT = t >= 1 ? 1 : scaled - index;
      const segment = m.segments[index];
      const point = segment.geodesic?.interpolateUsingFraction(localT);
      e.position = point
        ? Cesium.Cartesian3.fromRadians(point.longitude, point.latitude, 22000)
        : Cesium.Cartesian3.lerp(segment.from, segment.to, localT, new Cesium.Cartesian3());
      const rotation = segment.fromHeading + segment.headingDelta * localT;
      if (e.billboard.rotation?.setValue) e.billboard.rotation.setValue(rotation);
      else e.billboard.rotation = rotation;
    } else {
      // Continuous smooth extrapolation while awaiting the next authoritative packet
      const extraSeconds = (elapsed - m.duration) / 1000;
      if (extraSeconds < 10.0 && m.speedKn > 0) {
        const cur = Cesium.Cartographic.fromCartesian(m.to);
        const distRad = ((m.speedKn * (extraSeconds / 3600) * Math.max(1, state.speed)) * 1.852) / 6371.0;
        const hdg = -m.toHeading;
        const lat2 = Math.asin(Math.sin(cur.latitude) * Math.cos(distRad) + Math.cos(cur.latitude) * Math.sin(distRad) * Math.cos(hdg));
        const lon2 = cur.longitude + Math.atan2(Math.sin(hdg) * Math.sin(distRad) * Math.cos(cur.latitude), Math.cos(distRad) - Math.sin(cur.latitude) * Math.sin(lat2));
        e.position = Cesium.Cartesian3.fromRadians(lon2, lat2, 22000);
      } else {
        e.position = m.to;
      }
      if (e.billboard.rotation?.setValue) e.billboard.rotation.setValue(m.toHeading);
      else e.billboard.rotation = m.toHeading;
      if (elapsed > m.duration + 15000) state.motion.delete(id);
    }
  }
});

const weatherCanvas = $('weatherCanvas');
const weatherCtx = weatherCanvas.getContext('2d');
weatherCanvas.style.setProperty('--weather-opacity', String(state.weatherOpacity));
state.weatherFieldCanvas = document.createElement('canvas');
state.weatherFieldCtx = state.weatherFieldCanvas.getContext('2d');

const WEATHER_LAYERS = {
  waves: { title: 'WAVES', min: '1.3 m', max: '5.9 m', value: p => p.wave_m, minValue: 1.3, maxValue: 5.9, unit: 'm', direction: p => p.wave_dir_deg, speed: p => p.wave_m },
  wind: { title: 'WIND', min: '17 kt', max: '33 kt', value: p => p.wind_kn, minValue: 17.4, maxValue: 33, unit: 'kt', direction: p => p.wind_dir_deg, speed: p => p.wind_kn },
  current: { title: 'CURRENT', min: '1.1 kt', max: '2.5 kt', value: p => p.current_kn, minValue: 1.07, maxValue: 2.5, unit: 'kt', direction: p => p.current_dir_deg, speed: p => p.current_kn },
  storm: { title: 'STORM INTENSITY', min: '0%', max: '100%', value: p => p.storm * 100, minValue: 0, maxValue: 100, unit: '%', direction: null, speed: null }
};

function resizeWeather() {
  const d = Math.min(devicePixelRatio || 1, 1.5, Math.sqrt(2500000 / Math.max(1, innerWidth * innerHeight)));
  for (const canvas of [weatherCanvas, state.weatherFieldCanvas]) {
    canvas.width = Math.max(1, Math.floor(innerWidth * d));
    canvas.height = Math.max(1, Math.floor(innerHeight * d));
    canvas.style.width = innerWidth + 'px';
    canvas.style.height = innerHeight + 'px';
  }
  weatherCtx.setTransform(d, 0, 0, d, 0, 0);
  state.weatherFieldCtx.setTransform(d, 0, 0, d, 0, 0);
  state.weatherProjectionDirty = true;
  state.weatherBaseDirty = true;
  if (state.weatherVisible && !state.weatherFrame) state.weatherFrame = requestAnimationFrame(drawWeather);
}
addEventListener('resize', resizeWeather);
resizeWeather();

function setStatus(t, ok = true) {
  $('statusText').textContent = t;
  $('statusDot').style.background = ok ? '#05ffa1' : '#ff2a5f';
}

function toast(t) {
  $('toast').textContent = t;
  $('toast').classList.add('show');
  clearTimeout(toast.t);
  toast.t = setTimeout(() => $('toast').classList.remove('show'), 2600);
}

async function get(path) {
  const r = await fetch(API + path, { cache: 'no-store' });
  if (!r.ok) throw Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function post(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  if (!r.ok) throw Error(await r.text());
  return r.json();
}

async function loadMap() {
  try {
    const land = await Cesium.GeoJsonDataSource.load(`${API}/api/simulation/coastline`, {
      clampToGround: true,
      stroke: Cesium.Color.fromCssColorString('#00f2fe').withAlpha(0.45),
      fill: Cesium.Color.fromCssColorString('#071328').withAlpha(0.88),
      strokeWidth: 1.5
    });
    viewer.dataSources.add(land);
    state.land = land;
  } catch (e) {
    console.error(e);
    toast('Local coastline data is unavailable from the API');
  }
  try {
    const data = await get('/api/simulation/ports');
    const ports = data.ports || [];
    state.ports = ports;
    $('portCount').textContent = ports.length;
    for (const p of ports) {
      const ent = viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 3000),
        point: {
          pixelSize: 4,
          color: Cesium.Color.fromCssColorString('#00f2fe'),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 1,
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 13000000)
        },
        label: {
          text: p.name,
          font: '11px "Plus Jakarta Sans", sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.fromCssColorString('#030712'),
          outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -8),
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 7500000)
        }
      });
      ent.properties = { kind: 'port', portId: p.id };
      state.portEntities.push(ent);
    }
  } catch (e) {
    console.error(e);
    toast('Port catalog is unavailable from the API');
  }
}

function shipColor(v) {
  if (v.status === 'completed') return Cesium.Color.fromCssColorString('#05ffa1');
  if (v.status === 'destroyed' || v.status === 'failed') return Cesium.Color.fromCssColorString('#ff2a5f');
  if (v.land_clearance_km < 15.0) return Cesium.Color.fromCssColorString('#ffb703');
  return Cesium.Color.fromCssColorString('#00f2fe');
}

function addOrUpdateVessel(v, motionPath) {
  let e = state.entities.get(v.id);
  const pos = Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 22000);
  const heading = Cesium.Math.toRadians(Number(v.heading) || 0);
  const dotSize = v.status === 'destroyed' || v.status === 'failed' ? 6 : 3.5;
  if (!e) {
    e = viewer.entities.add({
      position: pos,
      billboard: {
        image: 'assets/vessel.png',
        width: 16,
        height: 48,
        rotation: -heading,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        scale: 1,
        color: Cesium.Color.WHITE,
        disableDepthTestDistance: 0,
        scaleByDistance: new Cesium.NearFarScalar(2.0e6, 0.9, 2.0e7, 0.42)
      },
      point: {
        pixelSize: dotSize,
        color: shipColor(v),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 1
      },
      label: {
        text: v.id,
        font: '10px monospace',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -30),
        show: false
      },
      properties: { kind: 'vessel', vesselId: v.id }
    });
    state.entities.set(v.id, e);
  } else {
    const current = e.position?.getValue?.(Cesium.JulianDate.now()) || pos;
    const fromHeading = e.billboard.rotation?.getValue?.() ?? -heading;
    const points = motionPath?.length ? motionPath : [v];
    let previous = current, previousHeading = fromHeading;
    const segments = points.map(point => {
      const to = Cesium.Cartesian3.fromDegrees(point.lon, point.lat, 22000);
      const toHeading = -Cesium.Math.toRadians(Number(point.heading) || 0);
      const start = Cesium.Cartographic.fromCartesian(previous);
      const end = Cesium.Cartographic.fromCartesian(to);
      const distance = Cesium.Cartesian3.distance(previous, to);
      const geodesic = distance > 1 ? new Cesium.EllipsoidGeodesic(start, end, viewer.scene.globe.ellipsoid) : null;
      const headingDelta = ((toHeading - previousHeading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      const segment = { from: previous, to, geodesic, fromHeading: previousHeading, headingDelta };
      previous = to;
      previousHeading = toHeading;
      return segment;
    });
    state.motion.set(v.id, {
      segments,
      to: pos,
      toHeading: -heading,
      speedKn: Number(v.speed_kn) || 16.0,
      started: performance.now(),
      duration: state.motionDurationMs
    });
    e.billboard.color = Cesium.Color.WHITE;
    e.point.color = shipColor(v);
    e.point.pixelSize = dotSize;
  }
  if (state.selected === v.id) selectVessel(v, false);
}

function applyVesselFilters() {
  const query = (state.searchQuery || '').trim().toLowerCase();
  for (const v of state.vessels) {
    const e = state.entities.get(v.id);
    if (!e) continue;
    let match = true;
    if (query && !v.id.toLowerCase().includes(query) && !v.origin.toLowerCase().includes(query) && !v.destination.toLowerCase().includes(query)) {
      match = false;
    }
    if (match && state.filter !== 'all') {
      if (state.filter === 'active' && v.status !== 'active') match = false;
      else if (state.filter === 'risk' && v.land_clearance_km >= 15.0) match = false;
      else if (state.filter === 'completed' && v.status !== 'completed' && (v.voyages_completed || 0) === 0) match = false;
    }
    e.show = match;
  }
}

function recordTrackPoint(v) {
  let history = state.trackHistory.get(v.id);
  if (!history) {
    history = [];
    state.trackHistory.set(v.id, history);
  }
  const point = {
    sim_hours: state.simHours,
    voyage_number: (v.voyages_completed || 0) + 1,
    lat: v.lat,
    lon: v.lon,
    heading: v.heading,
    progress: v.progress,
    status: v.status,
    action: v.ai_action_name,
    reward: v.reward_last,
    fuel_remaining_t: v.fuel_remaining_t,
    wind_kn: v.wind_kn,
    wave_m: v.wave_m,
    current_kn: v.current_kn,
    storm: v.storm
  };
  const previous = history[history.length - 1];
  if (previous && previous.sim_hours === point.sim_hours && previous.voyage_number === point.voyage_number) {
    history[history.length - 1] = point;
  } else {
    history.push(point);
  }
  if (history.length > 180) history.splice(0, history.length - 180);
}

function showTracePoint(index) {
  const v = state.vessels.find(item => item.id === state.selected);
  const history = state.trackHistory.get(state.selected) || [];
  const point = history[index];
  if (!v || !point) return;
  if (!state.traceEntity) {
    state.traceEntity = viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, 30000),
      point: { pixelSize: 10, color: Cesium.Color.fromCssColorString('#00f2fe'), outlineColor: Cesium.Color.fromCssColorString('#030712'), outlineWidth: 2 },
      label: { text: 'TRACE', font: '11px "JetBrains Mono", sans-serif', fillColor: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, -15) }
    });
  } else {
    state.traceEntity.position = Cesium.Cartesian3.fromDegrees(point.lon, point.lat, 30000);
  }
  $('traceReadout').textContent = `Voyage ${point.voyage_number} · T+${point.sim_hours.toFixed(1)} h · ${point.lat.toFixed(2)}°, ${point.lon.toFixed(2)}° · ${Math.round(point.heading)}° · ${point.action || '—'} · ${point.fuel_remaining_t.toFixed(0)} t fuel · wind ${point.wind_kn.toFixed(1)} kt · waves ${point.wave_m.toFixed(1)} m`;
}

function updateTraceControls(v) {
  const history = state.trackHistory.get(v.id) || [];
  const cursor = $('traceCursor');
  cursor.max = String(Math.max(0, history.length - 1));
  cursor.disabled = history.length < 2;
  $('tracePointCount').textContent = `${history.length} samples`;
  const hasTrace = history.length > 0;
  $('traceLive').disabled = !hasTrace;
  $('exportTrack').disabled = !hasTrace;
  if (!state.traceScrubbing && hasTrace) {
    cursor.value = String(history.length - 1);
    showTracePoint(history.length - 1);
  }
}

function exportTrack() {
  const history = state.trackHistory.get(state.selected) || [];
  if (!history.length) return;
  downloadJson(`navruna-${state.selected}-trace.json`, {
    format: 'navruna.voyage-trace',
    version: 1,
    exported_at_utc: new Date().toISOString(),
    run: state.lastSnapshot?.run || null,
    scenario: state.lastSnapshot?.scenario || null,
    vessel_id: state.selected,
    sample_limit: 180,
    points: history
  });
}

function exportRun() {
  if (!state.lastSnapshot?.run) return;
  downloadJson(`navruna-run-${state.lastSnapshot.run.run_id}.json`, {
    format: 'navruna.run-record',
    version: 1,
    exported_at_utc: new Date().toISOString(),
    run: state.lastSnapshot.run,
    scenario: state.lastSnapshot.scenario,
    sim_hours: state.lastSnapshot.sim_hours,
    metrics: state.lastSnapshot.metrics,
    events: state.lastSnapshot.events,
    tracks: Object.fromEntries(state.trackHistory)
  });
}

function clearVessels() {
  for (const e of state.entities.values()) viewer.entities.remove(e);
  state.entities.clear();
  state.motion.clear();
  if (state.traceEntity) viewer.entities.remove(state.traceEntity);
  state.traceEntity = null;
  if (state.vectorEntity) viewer.entities.remove(state.vectorEntity);
  state.vectorEntity = null;
}

function clamp(v, a = 0, b = 1) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }

function weatherColor(t) {
  t = clamp(t);
  const stops = [[20, 40, 165], [0, 166, 255], [20, 226, 192], [245, 235, 65], [255, 145, 45], [244, 52, 108], [133, 51, 255]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(lerp(a[0], b[0], f))},${Math.round(lerp(a[1], b[1], f))},${Math.round(lerp(a[2], b[2], f))})`;
}

function projectWeather() {
  if (!state.weatherProjectionDirty) return;
  state.weatherProjectionDirty = false;
  for (const p of state.weatherField) {
    const c = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 15000);
    const q = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, c);
    p._x = q?.x ?? -9999;
    p._y = q?.y ?? -9999;
    p._visible = !!q && q.x > -120 && q.x < innerWidth + 120 && q.y > -120 && q.y < innerHeight + 120;
  }
}

function wrappedIndex(value, count) { return ((value % count) + count) % count; }

function interpolateWeather(lat, lon) {
  const g = state.weatherGrid;
  if (!g || !state.weatherField.length) return null;
  const y = clamp((lat - g.lat_min) / g.lat_step, 0, g.lat_count - 1), y0 = Math.floor(y), y1 = Math.min(g.lat_count - 1, y0 + 1), fy = y - y0;
  const normalizedLon = ((lon + 180) % 360 + 360) % 360 - 180;
  const x = wrappedIndex((normalizedLon - g.lon_min) / g.lon_step, g.lon_count), x0 = Math.floor(x), x1 = (x0 + 1) % g.lon_count, fx = x - x0;
  const points = [state.weatherField[y0 * g.lon_count + x0], state.weatherField[y0 * g.lon_count + x1], state.weatherField[y1 * g.lon_count + x0], state.weatherField[y1 * g.lon_count + x1]], weights = [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx];
  if (points.some(p => !p)) return null;
  const result = { lat, lon: normalizedLon };
  for (const key of ['wind_kn', 'wave_m', 'wave_period_s', 'current_kn', 'storm', 'precipitation', 'visibility_nm', 'pressure_hpa']) {
    result[key] = points.reduce((sum, p, i) => sum + p[key] * weights[i], 0);
  }
  for (const key of ['wind_dir_deg', 'wave_dir_deg', 'current_dir_deg']) {
    const xdir = points.reduce((sum, p, i) => sum + Math.cos(p[key] * Math.PI / 180) * weights[i], 0);
    const ydir = points.reduce((sum, p, i) => sum + Math.sin(p[key] * Math.PI / 180) * weights[i], 0);
    result[key] = (Math.atan2(ydir, xdir) * 180 / Math.PI + 360) % 360;
  }
  return result;
}

function rgba(rgb, alpha) {
  const values = rgb.match(/\d+/g)?.slice(0, 3) || [255, 255, 255];
  return `rgba(${values.join(',')},${alpha})`;
}

function renderWeatherField() {
  if (!state.weatherGrid) return;
  projectWeather();
  const ctx = state.weatherFieldCtx, layer = WEATHER_LAYERS[state.weatherLayer];
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const radius = clamp((innerWidth + innerHeight) / 45, 42, 88);
  for (const p of state.weatherField) {
    if (!p._visible) continue;
    const n = clamp((layer.value(p) - layer.minValue) / Math.max(0.001, layer.maxValue - layer.minValue));
    const color = weatherColor(n);
    const g = ctx.createRadialGradient(p._x, p._y, 0, p._x, p._y, radius);
    g.addColorStop(0, rgba(color, 0.24));
    g.addColorStop(0.52, rgba(color, 0.12));
    g.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = g;
    ctx.fillRect(p._x - radius, p._y - radius, radius * 2, radius * 2);
  }
  state.weatherBaseDirty = false;
  state.weatherLastBaseRender = performance.now();
}

function particleReset(pt) {
  const p = state.weatherField[(Math.random() * state.weatherField.length) | 0];
  if (!p) { pt.life = 0; return; }
  const g = state.weatherGrid, latJitter = (Math.random() - 0.5) * (g?.lat_step || 5), lonJitter = (Math.random() - 0.5) * (g?.lon_step || 5);
  pt.lat = clamp(p.lat + latJitter, g?.lat_min ?? -90, g?.lat_max ?? 90);
  pt.lon = p.lon + lonJitter;
  pt.life = 0.7 + Math.random() * 2;
  pt.speed = 0.6 + Math.random() * 0.8;
}

function ensureParticles() {
  const target = Math.min(220, Math.max(100, Math.floor(innerWidth * innerHeight / 14000)));
  while (state.weatherParticles.length < target) {
    const pt = { life: 0, lat: 0, lon: 0, speed: 1 };
    particleReset(pt);
    state.weatherParticles.push(pt);
  }
}

function moveParticle(pt, dt, layer) {
  const sample = interpolateWeather(pt.lat, pt.lon);
  if (!sample || !layer.direction) {
    particleReset(pt);
    return null;
  }
  const angle = layer.direction(sample) * Math.PI / 180, speed = layer.speed(sample), scale = 0.018 * pt.speed * (0.55 + speed / 35);
  pt.lat += Math.cos(angle) * scale * dt;
  pt.lon += Math.sin(angle) * scale * dt / Math.max(0.25, Math.cos(pt.lat * Math.PI / 180));
  pt.lon = ((pt.lon + 180) % 360 + 360) % 360 - 180;
  pt.life -= dt * 0.25;
  const g = state.weatherGrid;
  if (pt.lat > (g?.lat_max ?? 90) || pt.lat < (g?.lat_min ?? -90) || pt.life <= 0) particleReset(pt);
  return sample;
}

function drawWeather(now = performance.now()) {
  if (!state.weatherVisible || !state.weatherField.length) {
    state.weatherFrame = null;
    return;
  }
  if (state.weatherBaseDirty && state.weatherLastBaseRender && now - state.weatherLastBaseRender < 66) {
    weatherCtx.clearRect(0, 0, innerWidth, innerHeight);
    weatherCtx.drawImage(state.weatherFieldCanvas, 0, 0, innerWidth, innerHeight);
    state.weatherFrame = requestAnimationFrame(drawWeather);
    return;
  }
  projectWeather();
  if (state.weatherBaseDirty) renderWeatherField();
  ensureParticles();
  const dt = Math.min(0.05, (now - state.weatherTime || 16) / 1000);
  state.weatherTime = now;
  weatherCtx.clearRect(0, 0, innerWidth, innerHeight);
  weatherCtx.drawImage(state.weatherFieldCanvas, 0, 0, innerWidth, innerHeight);
  const layer = WEATHER_LAYERS[state.weatherLayer];
  const animate = state.weatherParticlesEnabled && layer.direction && !matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (animate) {
    for (const pt of state.weatherParticles) {
      const sample = moveParticle(pt, dt, layer);
      if (!sample) continue;
      const c = Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, 16000);
      const q = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, c);
      if (!q || q.x < 0 || q.x > innerWidth || q.y < 0 || q.y > innerHeight) continue;
      const angle = layer.direction(sample) * Math.PI / 180;
      weatherCtx.strokeStyle = 'rgba(255,255,255,.58)';
      weatherCtx.lineWidth = 1.3;
      weatherCtx.beginPath();
      weatherCtx.moveTo(q.x, q.y);
      weatherCtx.lineTo(q.x + Math.sin(angle) * 8, q.y - Math.cos(angle) * 8);
      weatherCtx.stroke();
    }
  }
  if (layer.direction) {
    for (let i = 0; i < state.weatherField.length; i += 8) {
      const p = state.weatherField[i];
      if (!p?._visible) continue;
      const angle = layer.direction(p) * Math.PI / 180;
      const len = 7 + Math.min(10, layer.speed(p) * 0.2);
      const x2 = p._x + Math.sin(angle) * len;
      const y2 = p._y - Math.cos(angle) * len;
      weatherCtx.strokeStyle = 'rgba(242,248,250,.40)';
      weatherCtx.lineWidth = 1;
      weatherCtx.beginPath();
      weatherCtx.moveTo(p._x, p._y);
      weatherCtx.lineTo(x2, y2);
      weatherCtx.stroke();
    }
  }
  state.weatherFrame = animate ? requestAnimationFrame(drawWeather) : null;
}

function updateWeatherClock() {
  $('weatherClock').textContent = `SIM T+${state.simHours.toFixed(1)} h · FIELD T+${state.weatherFieldHours.toFixed(1)} h`;
}

function setWeatherLayer(layer) {
  if (!WEATHER_LAYERS[layer]) return;
  state.weatherLayer = layer;
  const cfg = WEATHER_LAYERS[layer];
  $('weatherLayerTitle').textContent = cfg.title;
  $('weatherMin').textContent = cfg.min;
  $('weatherMax').textContent = cfg.max;
  document.querySelectorAll('.weather-tabs button').forEach(b => {
    const active = b.dataset.weatherLayer === layer;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  state.weatherParticles.forEach(p => p.life = 0);
  state.weatherBaseDirty = true;
  if (state.weatherVisible && !state.weatherFrame) state.weatherFrame = requestAnimationFrame(drawWeather);
}

async function loadWeatherField(hours = 0, seed = state.weatherSeed) {
  const requestId = ++state.weatherRequestId;
  try {
    const d = await get(`/api/simulation/weather?hours=${encodeURIComponent(hours)}&seed=${encodeURIComponent(seed)}&lat_step=5&lon_step=5`);
    if (requestId !== state.weatherRequestId) return false;
    if (!d.ok || d.seed !== seed || !Array.isArray(d.field) || !d.grid || d.field.length !== d.grid.lat_count * d.grid.lon_count) {
      throw Error('Weather API returned an incomplete or mismatched grid');
    }
    state.weatherField = d.field;
    state.weatherGrid = d.grid;
    state.weatherFieldSeed = d.seed;
    state.weatherFieldHours = d.hours;
    state.weatherFieldBucket = Math.floor(d.hours / 3);
    state.weatherProjectionDirty = true;
    state.weatherBaseDirty = true;
    ensureParticles();
    updateWeatherClock();
    if (state.weatherVisible && !state.weatherFrame) state.weatherFrame = requestAnimationFrame(drawWeather);
    return true;
  } catch (e) {
    if (requestId === state.weatherRequestId) {
      console.error('NAVRUNA weather API:', e);
      $('weatherReadout').textContent = 'Weather field unavailable · check the local API';
      if (state.weatherVisible) toast('Weather field unavailable — check the local API');
    }
    return false;
  }
}

function queueWeatherTime(hours) {
  if (!state.weatherVisible) return;
  const bucket = Math.floor(hours / 3), seed = state.weatherSeed;
  if (bucket === state.weatherFieldBucket && seed === state.weatherFieldSeed) return;
  clearTimeout(state.weatherRequestTimer);
  state.weatherRequestTimer = setTimeout(() => loadWeatherField(bucket * 3, seed), 350);
}

function render(d) {
  state.lastSnapshot = d;
  if (d.run?.controller_at_start) state.controller = d.run.controller_at_start;
  $('exportRun').disabled = !d.run;
  if (d.vessels) state.vessels = d.vessels;
  const previousSimHours = state.simHours;
  state.simHours = Number(d.sim_hours) || 0;
  state.clockMotion = d.tick ? { from: previousSimHours, to: state.simHours, started: performance.now(), duration: state.motionDurationMs } : null;
  updateWeatherClock();
  queueWeatherTime(state.simHours);
  const m = d.metrics || {};
  state.tickCount++;
  $('simTime').textContent = `${state.simHours.toFixed(1)} h`;
  $('activeCount').textContent = m.active ?? 0;
  $('completedCount').textContent = m.voyages_completed ?? 0;
  $('failedCount').textContent = m.failed ?? 0;
  if (d.tick) {
    $('simActivity').textContent = `Tick ${state.tickCount} · +${d.tick.hours} h · moved ${d.tick.moved_vessels}/${m.total_vessels || 0} · ${d.tick.active_vessels} active · ${m.voyages_completed || 0} voyages done`;
  }
  $('total').textContent = m.total_vessels ?? 0;
  $('success').textContent = `${m.success_rate ?? 0}%`;
  $('fuel').textContent = `${m.avg_fuel_t ?? 0} t`;
  $('co2').textContent = `${m.avg_co2_t ?? 0} t`;
  $('avgRisk').textContent = m.avg_risk ?? 0;
  $('collisions').textContent = m.collisions ?? 0;
  $('successes').textContent = m.voyages_completed ?? m.completed ?? 0;
  $('fuelFailures').textContent = m.failure_reasons?.FUEL_EXHAUSTED ?? 0;
  $('noSafeFailures').textContent = m.failure_reasons?.NO_SAFE_ACTION ?? 0;
  $('safeRecoveries').textContent = m.safe_recoveries ?? 0;
  $('fleetTitle').textContent = `${m.total_vessels || 0} vessels · ${state.controller}`;
  const activeShare = m.total_vessels ? Math.min(100, (m.active || 0) / m.total_vessels * 100) : 0;
  $('fleetProgress').style.width = `${activeShare}%`;
  $('fleetProgress').parentElement.setAttribute('aria-valuenow', String(Math.round(activeShare)));
  
  state.vessels.forEach(v => {
    recordTrackPoint(v);
    addOrUpdateVessel(v, d.motion_paths?.[v.id]);
  });
  applyVesselFilters();
  
  if (state.selected) {
    const v = state.vessels.find(x => x.id === state.selected);
    if (v) selectVessel(v, false);
  }
  const active = state.vessels.filter(v => v.status === 'active');
  $('windValue').textContent = active.length ? `${Math.round(active.reduce((sum, v) => sum + v.wind_kn, 0) / active.length)} kt` : '—';
  $('waveValue').textContent = active.length ? `${(active.reduce((sum, v) => sum + v.wave_m, 0) / active.length).toFixed(1)} m` : '—';
  $('currentValue').textContent = active.length ? `${(active.reduce((sum, v) => sum + v.current_kn, 0) / active.length).toFixed(1)} kt` : '—';
  $('stormValue').textContent = active.length ? `${Math.round(active.reduce((sum, v) => sum + v.storm, 0) / active.length * 100)}%` : '—';
}

function selectVessel(v, focus = true) {
  if (focus) state.traceScrubbing = false;
  state.selected = v.id;
  $('selectedEmpty').hidden = true;
  $('selectedShip').hidden = false;
  $('selectedTitle').textContent = v.id;
  $('shipId').textContent = v.id;
  $('shipStatus').textContent = v.status.toUpperCase();
  $('shipOrigin').textContent = v.origin;
  $('shipDestination').textContent = v.destination;
  $('shipFuel').textContent = `${Math.round(v.fuel_remaining_t)} t`;
  $('shipClearance').textContent = `${v.land_clearance_km.toFixed(1)} km`;
  $('shipHeading').textContent = `${Math.round(v.heading)}°`;
  $('shipProgress').textContent = `${Math.round(v.progress * 100)}%`;
  $('shipAction').textContent = v.ai_action_name;
  $('shipReward').textContent = v.reward_last.toFixed(3);
  
  const isRisk = v.land_clearance_km < 15.0;
  $('clearanceText').textContent = isRisk ? 'COLLISION RISK' : 'SAFE';
  $('clearanceBar').style.width = Math.min(100, v.land_clearance_km / 50 * 100) + '%';
  $('clearanceBar').style.background = isRisk ? '#ff2a5f' : '#05ffa1';
  
  $('windValue').textContent = `${v.wind_kn.toFixed(1)} kt @ ${Math.round(v.wind_dir_deg)}°`;
  $('waveValue').textContent = `${v.wave_m.toFixed(1)} m`;
  $('currentValue').textContent = `${v.current_kn.toFixed(1)} kt @ ${Math.round(v.current_dir_deg)}°`;
  $('stormValue').textContent = `${Math.round(v.storm * 100)}%`;
  
  updateTraceControls(v);
  
  // Highlight heading vector
  const rad = Cesium.Math.toRadians(Number(v.heading) || 0);
  const targetLat = v.lat + 0.35 * Math.cos(rad);
  const targetLon = v.lon + (0.35 / Math.max(0.18, Math.cos(Cesium.Math.toRadians(v.lat)))) * Math.sin(rad);
  if (!state.vectorEntity) {
    state.vectorEntity = viewer.entities.add({
      polyline: {
        positions: [Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 25000), Cesium.Cartesian3.fromDegrees(targetLon, targetLat, 25000)],
        width: 3,
        material: new Cesium.PolylineGlowMaterialProperty({ glowPower: 0.3, color: Cesium.Color.fromCssColorString('#00f2fe') })
      }
    });
  } else {
    state.vectorEntity.polyline.positions = [Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 25000), Cesium.Cartesian3.fromDegrees(targetLon, targetLat, 25000)];
  }

  const entity = state.entities.get(v.id);
  if (focus && entity) viewer.flyTo(entity, { duration: 0.8, maximumHeight: 4000000 });
}

viewer.selectedEntityChanged.addEventListener(e => {
  const id = e?.properties?.vesselId?.getValue?.(Cesium.JulianDate.now());
  if (id) {
    const v = state.vessels.find(x => x.id === id);
    if (v) selectVessel(v, true);
  }
});

viewer.camera.changed.addEventListener(() => {
  state.weatherProjectionDirty = true;
  state.weatherBaseDirty = true;
  if (state.weatherVisible && !state.weatherFrame) state.weatherFrame = requestAnimationFrame(drawWeather);
});

async function start() {
  stop();
  setStatus('GENERATING FLEET');
  $('startBtn').disabled = true;
  try {
    const seed = Number($('scenarioSeed').value);
    const vesselCount = Number($('vesselCount').value);
    let scenario = state.scenarioConfig;
    if (!scenario || scenario.seed !== seed || scenario.vessel_count !== vesselCount) {
      scenario = await get(`/api/simulation/scenario/template?seed=${encodeURIComponent(seed)}&vessel_count=${encodeURIComponent(vesselCount)}`);
    }
    state.scenarioConfig = scenario;
    state.weatherSeed = seed;
    queueWeatherTime(state.simHours);
    const fleet = await post('/api/simulation/start', { scenario });
    clearVessels();
    state.trackHistory.clear();
    state.traceScrubbing = false;
    state.selected = null;
    $('selectedEmpty').hidden = false;
    $('selectedShip').hidden = true;
    $('traceCursor').disabled = true;
    $('tracePointCount').textContent = '0 samples';
    $('traceReadout').textContent = 'Select a vessel to inspect its recent history.';
    $('exportTrack').disabled = true;
    $('traceLive').disabled = true;
    render(fleet);
    setStatus('STARTING SIMULATION');
    run();
    toast(`${fleet.metrics.total_vessels} vessels deployed · controller: ${state.controller}`);
  } catch (e) {
    console.error(e);
    stop();
    setStatus('BACKEND ERROR', false);
    toast(`Could not start simulator: ${e.message}`);
  } finally {
    $('startBtn').disabled = false;
  }
}

async function step(hours = 1, durationMs = 1000) {
  if (state.stepInFlight) return false;
  state.stepInFlight = true;
  state.motionDurationMs = durationMs;
  try {
    const d = await post('/api/simulation/step', { hours });
    render(d);
    return true;
  } catch (e) {
    console.error('Simulation step error:', e);
    setStatus('NETWORK ISSUE · PAUSED', false);
    toast(`Step error: ${e.message}`);
    return false;
  } finally {
    state.stepInFlight = false;
  }
}

/* ── Rock-Solid 1-Second Cadence Movement Engine ── */
async function simulationTick() {
  if (!state.running) return;
  const started = performance.now();
  // Hours per second: at speed=1 => 1.0 sim hours every 1 second
  const hours = Math.max(0.25, Math.min(12.5, state.speed));
  const durationMs = 1000;
  const ok = await step(hours, durationMs);
  const elapsed = performance.now() - started;
  if (state.running) {
    // Deliver continuous visual motion: trigger next step at exactly 1000ms boundary
    const delay = Math.max(30, 1000 - elapsed);
    state.timer = setTimeout(simulationTick, delay);
  }
}

function run() {
  if (!state.vessels.length) {
    start();
    return;
  }
  if (state.running) return;
  state.running = true;
  state.lastStepDurationMs = 1000;
  $('playBtn').hidden = false;
  $('playBtn').textContent = '■ STOP SIMULATION';
  setStatus('FLEET SIMULATION RUNNING');
  simulationTick();
}

function stop() {
  state.running = false;
  clearTimeout(state.timer);
  state.timer = null;
  $('playBtn').hidden = true;
  $('playBtn').textContent = '■ STOP SIMULATION';
  setStatus('SIMULATION PAUSED');
}

async function refreshPolicy() {
  try {
    const d = await get('/api/simulation/policy');
    state.controller = d.controller || (d.trained ? 'PPO checkpoint' : 'bootstrap heading heuristic');
    $('policyState').textContent = d.trained ? 'TRAINED CHECKPOINT' : 'BOOTSTRAP HEURISTIC';
    $('policyAlgorithm').textContent = state.controller;
    $('policyDevice').textContent = (d.gpu_name && d.gpu_name !== 'CPU') ? d.gpu_name : 'CPU';
    $('gpuPill').textContent = d.gpu_name && d.gpu_name !== 'CPU' ? `AI · ${d.gpu_name.replace('NVIDIA ', '')}` : 'AI · CPU';
    $('fleetTitle').textContent = `${state.vessels.length} vessels · ${state.controller}`;
  } catch (e) {}
}

function setPanelOpen(panelId, open) {
  const panel = $(panelId), button = $(panelId === 'leftPanel' ? 'leftToggle' : 'rightToggle');
  panel.classList.toggle('hidden', !open);
  panel.setAttribute('aria-hidden', String(!open));
  button.setAttribute('aria-expanded', String(open));
  if (!open && panel.contains(document.activeElement)) button.focus();
}

function togglePanel(panelId) {
  const panel = $(panelId), open = panel.classList.contains('hidden');
  if (open && matchMedia('(max-width: 700px)').matches) setPanelOpen(panelId === 'leftPanel' ? 'rightPanel' : 'leftPanel', false);
  setPanelOpen(panelId, open);
}

// Controls bindings
$('vesselCount').oninput = e => {
  $('vesselCountValue').textContent = e.target.value;
  state.scenarioConfig = null;
};
$('scenarioSeed').oninput = () => {
  state.scenarioConfig = null;
  const seed = Number($('scenarioSeed').value);
  if (Number.isInteger(seed) && seed >= 0 && seed <= 2147483647) {
    state.weatherSeed = seed;
    queueWeatherTime(state.simHours);
  }
};

state.speed = Number($('simulationSpeed').value) || 1;
$('simulationSpeed').onchange = e => {
  state.speed = Number(e.target.value) || 1;
};

// Search & Filter Bindings
if ($('vesselSearch')) {
  $('vesselSearch').oninput = e => {
    state.searchQuery = e.target.value;
    applyVesselFilters();
  };
}

document.querySelectorAll('.filter-chip').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.filter-chip').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.filter = btn.dataset.filter || 'all';
    applyVesselFilters();
  };
});

$('startBtn').onclick = start;
$('stepBtn').onclick = () => step(1, 1000);
$('playBtn').onclick = () => state.running ? stop() : run();

$('weatherToggle').onclick = () => {
  state.weatherVisible = !state.weatherVisible;
  $('weatherToggle').classList.toggle('active', state.weatherVisible);
  $('weatherToggle').setAttribute('aria-expanded', String(state.weatherVisible));
  $('weatherState').textContent = state.weatherVisible ? 'ON' : 'OFF';
  $('weatherHud').hidden = !state.weatherVisible;
  weatherCanvas.classList.toggle('visible', state.weatherVisible);
  if (state.weatherVisible) {
    state.weatherProjectionDirty = true;
    state.weatherBaseDirty = true;
    if (!state.weatherField.length || state.weatherFieldSeed !== state.weatherSeed) {
      loadWeatherField(Math.floor(state.simHours / 3) * 3, state.weatherSeed);
    } else {
      queueWeatherTime(state.simHours);
      if (!state.weatherFrame) drawWeather();
    }
  } else {
    clearTimeout(state.weatherRequestTimer);
    if (state.weatherFrame) cancelAnimationFrame(state.weatherFrame);
    state.weatherFrame = null;
    weatherCtx.clearRect(0, 0, innerWidth, innerHeight);
  }
};

$('weatherOpacity').oninput = e => {
  state.weatherOpacity = Number(e.target.value) / 100;
  weatherCanvas.style.setProperty('--weather-opacity', String(state.weatherOpacity));
};
$('weatherParticlesToggle').onchange = e => {
  state.weatherParticlesEnabled = e.target.checked;
  if (state.weatherFrame) {
    cancelAnimationFrame(state.weatherFrame);
    state.weatherFrame = null;
  }
  if (state.weatherVisible) drawWeather();
};

$('traceCursor').oninput = e => {
  state.traceScrubbing = true;
  showTracePoint(Number(e.target.value));
};
$('traceLive').onclick = () => {
  state.traceScrubbing = false;
  const vessel = state.vessels.find(item => item.id === state.selected);
  if (vessel) updateTraceControls(vessel);
};

$('exportTrack').onclick = exportTrack;
$('exportRun').onclick = exportRun;

$('resetViewBtn').onclick = () => {
  viewer.camera.cancelFlight();
  viewer.camera.setView(defaultView);
  state.weatherProjectionDirty = true;
  state.weatherBaseDirty = true;
};

viewer.container.addEventListener('pointermove', event => {
  if (!state.weatherVisible || !state.weatherField.length) return;
  const point = viewer.camera.pickEllipsoid(new Cesium.Cartesian2(event.clientX, event.clientY), viewer.scene.globe.ellipsoid);
  if (!point) return;
  const coords = viewer.scene.globe.ellipsoid.cartesianToCartographic(point);
  const lat = Cesium.Math.toDegrees(coords.latitude), lon = Cesium.Math.toDegrees(coords.longitude);
  const sample = interpolateWeather(lat, lon);
  if (!sample) return;
  const layer = WEATHER_LAYERS[state.weatherLayer], value = layer.value(sample);
  $('weatherReadout').textContent = `${lat.toFixed(1)}°, ${lon.toFixed(1)}° · ${layer.title}: ${value.toFixed(1)} ${layer.unit}`;
});

viewer.container.addEventListener('pointerleave', () => {
  $('weatherReadout').textContent = 'Move over the ocean to inspect the synthetic field';
});

$('weatherClose').onclick = () => $('weatherToggle').click();

$('resetBtn').onclick = async () => {
  stop();
  try {
    const d = await post('/api/simulation/reset', {});
    clearVessels();
    state.trackHistory.clear();
    state.traceScrubbing = false;
    state.selected = null;
    $('selectedEmpty').hidden = false;
    $('selectedShip').hidden = true;
    $('traceCursor').disabled = true;
    $('tracePointCount').textContent = '0 samples';
    $('traceReadout').textContent = 'Select a vessel to inspect its recent history.';
    $('exportTrack').disabled = true;
    $('traceLive').disabled = true;
    render(d);
    setStatus('SIMULATION READY');
  } catch (e) {
    toast(`Could not reset simulation: ${e.message}`);
  }
};

$('collapseLeft').onclick = () => setPanelOpen('leftPanel', false);
$('collapseRight').onclick = () => setPanelOpen('rightPanel', false);
$('leftToggle').onclick = () => togglePanel('leftPanel');
$('rightToggle').onclick = () => togglePanel('rightPanel');

$('cinemaBtn').onclick = () => {
  state.cinema = !state.cinema;
  document.body.classList.toggle('cinema', state.cinema);
  $('cinemaBtn').setAttribute('aria-pressed', String(state.cinema));
};

if (matchMedia('(max-width: 700px)').matches) setPanelOpen('rightPanel', false);

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    document.body.classList.remove('cinema');
    state.cinema = false;
    $('cinemaBtn').setAttribute('aria-pressed', 'false');
    state.selected = null;
    $('selectedEmpty').hidden = false;
    $('selectedShip').hidden = true;
    if (state.vectorEntity) viewer.entities.remove(state.vectorEntity);
    state.vectorEntity = null;
    return;
  }
  if (event.target.closest('input,select,textarea,button,a')) return;
  if (event.code === 'Space') { event.preventDefault(); state.running ? stop() : run(); }
  if (event.key === 's' || event.key === 'S') step(1, 1000);
  if (event.key === 'r' || event.key === 'R') $('resetBtn').click();
  if (event.key === 'n' || event.key === 'N') $('resetViewBtn').click();
  if (event.key === '1') setWeatherLayer('waves');
  if (event.key === '2') setWeatherLayer('wind');
  if (event.key === '3') setWeatherLayer('current');
  if (event.key === '4') setWeatherLayer('storm');
  if (event.key === 'h' || event.key === 'H') $('cinemaBtn').click();
  if (event.key === 'm' || event.key === 'M') togglePanel('leftPanel');
  if (event.key === 'i' || event.key === 'I') togglePanel('rightPanel');
  if (event.key === 'w' || event.key === 'W') $('weatherToggle').click();
  if (event.key === 't' || event.key === 'T') location.href = 'training.html';
});

(async () => {
  await loadMap();
  const weatherOK = await loadWeatherField();
  await refreshPolicy();
  try {
    const d = await get('/api/simulation/state');
    render(d);
  } catch (e) {}
  setStatus(weatherOK ? 'SIMULATION READY' : 'SIMULATION READY · WEATHER API OFFLINE', weatherOK);
  setInterval(refreshPolicy, 2500);
})();

function downloadJson(filename, value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function validateScenarioShape(s) {
  const keys = value => Object.keys(value || {}).sort().join(',');
  if (s.format !== 'navruna.scenario' || s.version !== 2 || !Number.isInteger(s.seed) || s.seed < 0 || s.seed > 2147483647 || !Number.isInteger(s.vessel_count) || s.vessel_count < 10 || s.vessel_count > 300) {
    throw Error('Unsupported or invalid NAVRUNA scenario');
  }
  if (keys(s) !== 'environment,format,seed,version,vessel_count,weather' || keys(s.weather) !== 'mode,model_version,seed' || keys(s.environment) !== 'app_version,datasets' || keys(s.environment.datasets) !== 'coastline,ports') {
    throw Error('Scenario fields do not match the v2 format');
  }
  if (s.weather.mode !== 'procedural' || s.weather.seed !== s.seed || typeof s.weather.model_version !== 'string' || typeof s.environment.app_version !== 'string' || !s.environment.datasets || !['coastline', 'ports'].every(k => /^[a-f0-9]{64}$/i.test(s.environment.datasets[k] || ''))) {
    throw Error('Scenario provenance or weather settings are invalid');
  }
  return s;
}

$('saveScenario').onclick = async () => {
  const button = $('saveScenario');
  button.disabled = true;
  try {
    const seed = Number($('scenarioSeed').value), vesselCount = Number($('vesselCount').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647 || !Number.isInteger(vesselCount) || vesselCount < 10 || vesselCount > 300) {
      throw Error('Seed or vessel count is outside the supported range');
    }
    const scenario = await get(`/api/simulation/scenario/template?seed=${encodeURIComponent(seed)}&vessel_count=${encodeURIComponent(vesselCount)}`);
    state.scenarioConfig = validateScenarioShape(scenario);
    downloadJson(`navruna-scenario-${seed}.json`, scenario);
    toast('Versioned scenario downloaded');
  } catch (error) {
    toast(`Could not save scenario: ${error.message}`);
  } finally {
    button.disabled = false;
  }
};

$('loadScenario').onclick = () => $('scenarioFile').click();

$('scenarioFile').onchange = async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (file.size > 1000000) throw Error('Scenario file is larger than 1 MB');
    const scenario = JSON.parse(await file.text());
    if (scenario.version === 1) {
      if (!Number.isInteger(scenario.seed) || scenario.seed < 0 || scenario.seed > 2147483647 || !Number.isInteger(scenario.vessel_count) || scenario.vessel_count < 10 || scenario.vessel_count > 300) {
        throw Error('Legacy scenario values are invalid');
      }
      state.scenarioConfig = await get(`/api/simulation/scenario/template?seed=${scenario.seed}&vessel_count=${scenario.vessel_count}`);
      toast('Legacy scenario loaded and upgraded to v2');
    } else {
      validateScenarioShape(scenario);
      const local = await get(`/api/simulation/scenario/template?seed=${scenario.seed}&vessel_count=${scenario.vessel_count}`);
      if (scenario.weather.model_version !== local.weather.model_version) throw Error('Scenario uses a different synthetic weather model version');
      if (scenario.environment.datasets.coastline !== local.environment.datasets.coastline || scenario.environment.datasets.ports !== local.environment.datasets.ports) {
        throw Error('Scenario data fingerprints do not match this NAVRUNA installation');
      }
      state.scenarioConfig = scenario;
      toast('Scenario loaded; data versions match this installation');
    }
    $('scenarioSeed').value = state.scenarioConfig.seed;
    $('vesselCount').value = state.scenarioConfig.vessel_count;
    $('vesselCountValue').textContent = state.scenarioConfig.vessel_count;
    state.weatherSeed = state.scenarioConfig.seed;
    queueWeatherTime(state.simHours);
  } catch (error) {
    state.scenarioConfig = null;
    toast(`Could not load scenario: ${error.message}`);
  } finally {
    event.target.value = '';
  }
};
