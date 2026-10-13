// Same-origin in production (the API serves this folder). The local launcher runs
// the UI on :8080 and the API on :8000, so point at :8000 when the ports differ.
const API = (location.port && location.port !== '8000') ? `${location.protocol}//${location.hostname}:8000` : '';
const $ = id => document.getElementById(id);

async function get(path) {
  const response = await fetch(API + path);
  if (!response.ok) throw Error(await response.text());
  return response.json();
}

function renderBenchmarkTable(data) {
  const container = $('benchmarkTableContainer');
  if (!container || !data.controllers) return;
  const controllers = data.controllers;
  const names = Object.keys(controllers);
  if (!names.length) return;

  let html = `
    <table class="benchmark-table">
      <thead>
        <tr>
          <th>CONTROLLER</th>
          <th>SUCCESS RATE</th>
          <th>COLLISIONS</th>
          <th>MIN CLEARANCE</th>
          <th>AVG FUEL</th>
          <th>FUEL / NM</th>
          <th>AVG DURATION</th>
          <th>RECOVERIES</th>
          <th>RETURN</th>
        </tr>
      </thead>
      <tbody>
  `;

  names.forEach(name => {
    const r = controllers[name];
    const isPPO = name.toLowerCase().includes('ppo');
    const badgeClass = isPPO ? 'badge-ppo' : 'badge-baseline';
    const badgeLabel = isPPO ? 'PPO NEURAL' : 'BOOTSTRAP HEURISTIC';
    
    html += `
      <tr>
        <td>
          <span class="${badgeClass}">${badgeLabel}</span>
          <div style="font-size:10px;color:var(--muted);margin-top:4px;">${name}</div>
        </td>
        <td style="color:${r.success_rate >= 0.8 ? 'var(--green)' : 'var(--amber)'};font-weight:700;">
          ${(r.success_rate * 100).toFixed(1)}% (${r.successes}/${r.sample_count})
        </td>
        <td style="color:${r.collision_rate === 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">
          ${(r.collision_rate * 100).toFixed(1)}%
        </td>
        <td>${r.mean_minimum_clearance_km ? r.mean_minimum_clearance_km.toFixed(1) + ' km' : '—'}</td>
        <td>${r.mean_fuel_used_t ? r.mean_fuel_used_t.toFixed(1) + ' t' : '—'}</td>
        <td>${r.mean_fuel_per_nm_t ? (r.mean_fuel_per_nm_t * 1000).toFixed(2) + ' kg/nm' : '—'}</td>
        <td>${r.mean_duration_hours ? r.mean_duration_hours.toFixed(1) + ' h' : '—'}</td>
        <td style="color:var(--cyan);font-weight:600;">${r.mean_safe_recoveries ? r.mean_safe_recoveries.toFixed(1) : '0.0'}</td>
        <td style="color:var(--indigo-bright);font-weight:700;">${r.mean_return ? r.mean_return.toFixed(3) : '—'}</td>
      </tr>
    `;
  });

  html += `
      </tbody>
    </table>
  `;

  container.innerHTML = html;
}

function renderFleetSummary(data) {
  const container = $('fleetMetricsGrid');
  if (!container) return;
  const f = data.fleet || data;
  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:14px;">
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>VOYAGES STARTED</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;">${f.voyages_started ?? 0}</b>
      </div>
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>COMPLETED</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;color:var(--green)">${f.voyages_completed ?? 0}</b>
      </div>
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>FAILED</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;color:var(--red)">${f.voyages_failed ?? 0}</b>
      </div>
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>SUCCESS RATE</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;color:var(--cyan)">${f.success_rate ?? f.voyage_success_rate ?? 0}%</b>
      </div>
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>MEAN CLEARANCE</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;">${f.mean_min_clearance_km ?? 0} km</b>
      </div>
      <div class="glass" style="padding:10px;border-radius:var(--radius-sm);">
        <small>COLLISION RATE</small>
        <b style="display:block;font-size:16px;font-family:var(--font-mono);margin-top:2px;color:var(--green)">${f.collision_rate ?? 0}%</b>
      </div>
    </div>
  `;
}

async function load() {
  try {
    const data = await get('/api/simulation/analytics');
    const training = data.training || {};
    $('trained').textContent = training.trained ? 'TRAINED PPO' : 'UNTRAINED';
    $('device').textContent = training.gpu_name && training.gpu_name !== 'CPU' ? training.gpu_name : 'CPU';
    $('episodes').textContent = (training.episodes || 0).toLocaleString();
    $('ports').textContent = data.ports || 279;
    
    renderFleetSummary(data);
    
    if ($('report')) {
      $('report').textContent = JSON.stringify({
        training,
        fleet: {
          voyages_started: data.voyages_started,
          voyages_completed: data.voyages_completed,
          voyages_failed: data.voyages_failed,
          success_rate: data.voyage_success_rate,
          collision_rate: data.collision_rate,
          fuel_failure_rate: data.fuel_failure_rate,
          timeout_rate: data.timeout_rate,
          no_safe_action_rate: data.no_safe_action_rate,
          mean_min_clearance_km: data.mean_min_clearance_km,
          fuel_per_nm: data.fuel_per_nm,
          mean_safe_actions: data.mean_safe_actions,
          no_safe_action_steps: data.no_safe_action_steps,
          failure_reasons: data.failure_reasons
        }
      }, null, 2);
    }
  } catch (error) {
    console.error(error);
    if ($('report')) $('report').textContent = 'Backend analytics unavailable: ' + error.message;
  }
}

load();
setInterval(load, 5000);

let lastEvaluation = null;

$('downloadEvaluation').onclick = () => {
  if (!lastEvaluation) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastEvaluation, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `navruna-evaluation-seed-${lastEvaluation.seed_start}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

$('evaluateBtn').onclick = async () => {
  const button = $('evaluateBtn');
  const status = $('evaluationStatus');
  const report = $('evaluationReport');
  const episodes = Number($('evaluationEpisodes').value);
  const seed = Number($('evaluationSeed').value);

  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 32 || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    status.textContent = 'Enter 1–32 voyages and a valid non-negative seed.';
    return;
  }

  button.disabled = true;
  status.textContent = `Evaluating ${episodes} paired scenarios across identical procedural seeds…`;
  status.style.color = 'var(--cyan)';
  report.textContent = 'Monte Carlo paired evaluation running. Keep this tab open...';

  try {
    const response = await fetch(API + '/api/simulation/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ episodes, seed })
    });
    const data = await response.json().catch(() => ({ detail: 'The API returned an invalid response.' }));
    if (!response.ok) throw Error(data.detail || `HTTP ${response.status}`);
    if (!data.accepted) throw Error(data.message || 'Evaluation was not accepted');

    lastEvaluation = data;
    $('downloadEvaluation').disabled = false;
    renderBenchmarkTable(data);
    report.textContent = JSON.stringify(data, null, 2);
    status.textContent = `Evaluation complete · ${data.episode_count} voyages per controller benchmarked.`;
    status.style.color = 'var(--green)';
  } catch (error) {
    status.textContent = `Evaluation failed: ${error.message}`;
    status.style.color = 'var(--red)';
    report.textContent = 'Check that training is idle and the local API is available.';
    console.error('NAVRUNA evaluation:', error);
  } finally {
    button.disabled = false;
  }
};