// Same-origin in production (the API serves this folder). The local launcher runs
// the UI on :8080 and the API on :8000, so point at :8000 when the ports differ.
const API = (location.port && location.port !== '8000') ? `${location.protocol}//${location.hostname}:8000` : '';
const $ = id => document.getElementById(id);

async function get(p) {
  const r = await fetch(API + p);
  if (!r.ok) throw Error(await r.text());
  return r.json();
}

async function post(p, b) {
  const r = await fetch(API + p, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(b)
  });
  if (!r.ok) throw Error(await r.text());
  return r.json();
}

// Chart hover state tracker
const chartHovers = new Map();

/* ── Industry-Grade High-Precision HTML5 Canvas Chart Engine ── */
function drawAdvancedChart(canvasId, seriesList, labels, options = {}) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || canvas.clientWidth || 400;
  const h = rect.height || canvas.clientHeight || 190;
  
  if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
  }
  
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // Filter valid data series
  const cleanSeries = seriesList.map(s => ({
    ...s,
    values: (s.values || []).map(v => (v !== null && v !== undefined && Number.isFinite(v)) ? Number(v) : null)
  })).filter(s => s.values.some(v => v !== null));

  if (!cleanSeries.length || !labels.length) {
    ctx.fillStyle = '#64748b';
    ctx.font = '12px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Awaiting neural optimization telemetry...', w / 2, h / 2);
    ctx.restore();
    return;
  }

  // Calculate global min and max across all series
  const allVals = cleanSeries.flatMap(s => s.values.filter(v => v !== null));
  let minVal = options.minVal !== undefined ? options.minVal : Math.min(...allVals);
  let maxVal = options.maxVal !== undefined ? options.maxVal : Math.max(...allVals);
  if (minVal === maxVal) {
    minVal -= 1;
    maxVal += 1;
  }
  // Generous padding
  const padTop = 22;
  const padBottom = 26;
  const padLeft = 44;
  const padRight = 18;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const getX = i => padLeft + (plotW * (i / Math.max(1, labels.length - 1)));
  const getY = v => padTop + plotH * (1 - (v - minVal) / (maxVal - minVal));

  // 1. Draw subtle tactical grid
  const gridLines = 4;
  ctx.strokeStyle = 'rgba(0, 242, 254, 0.07)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= gridLines; i++) {
    const gy = padTop + (plotH * (i / gridLines));
    ctx.moveTo(padLeft, gy);
    ctx.lineTo(w - padRight, gy);
    
    // Y-axis numerical labels
    const valAtY = maxVal - ((maxVal - minVal) * (i / gridLines));
    ctx.fillStyle = '#64748b';
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const labelStr = Math.abs(valAtY) >= 100 ? Math.round(valAtY).toString() : valAtY.toFixed(2);
    ctx.fillText(labelStr, padLeft - 6, gy);
  }
  ctx.stroke();

  // 2. Optional target benchmark line
  if (options.targetLine !== undefined && options.targetLine >= minVal && options.targetLine <= maxVal) {
    const ty = getY(options.targetLine);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(5, 255, 161, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(padLeft, ty);
    ctx.lineTo(w - padRight, ty);
    ctx.stroke();
    ctx.fillStyle = '#05ffa1';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`TARGET: ${options.targetLine}%`, padLeft + 6, ty - 5);
    ctx.restore();
  }

  // 3. Draw Each Series Curve & Gradients
  cleanSeries.forEach((series, sIndex) => {
    const color = series.color || (sIndex === 0 ? '#00f2fe' : sIndex === 1 ? '#ffb703' : '#05ffa1');
    const validPoints = [];
    series.values.forEach((v, idx) => {
      if (v !== null) validPoints.push({ x: getX(idx), y: getY(v), val: v, idx });
    });
    if (validPoints.length < 2) return;

    // Gradient Area Fill under primary curve
    if (series.fillArea !== false && sIndex === 0) {
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
      grad.addColorStop(0, series.fillColor || 'rgba(0, 242, 254, 0.28)');
      grad.addColorStop(0.65, 'rgba(0, 242, 254, 0.08)');
      grad.addColorStop(1, 'rgba(0, 242, 254, 0.00)');

      ctx.beginPath();
      ctx.moveTo(validPoints[0].x, padTop + plotH);
      ctx.lineTo(validPoints[0].x, validPoints[0].y);
      for (let i = 1; i < validPoints.length; i++) {
        const prev = validPoints[i - 1];
        const curr = validPoints[i];
        const mx = (prev.x + curr.x) / 2;
        ctx.bezierCurveTo(mx, prev.y, mx, curr.y, curr.x, curr.y);
      }
      ctx.lineTo(validPoints[validPoints.length - 1].x, padTop + plotH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // High-contrast glowing Bezier curve
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = series.lineWidth || 2.2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(validPoints[0].x, validPoints[0].y);
    for (let i = 1; i < validPoints.length; i++) {
      const prev = validPoints[i - 1];
      const curr = validPoints[i];
      const mx = (prev.x + curr.x) / 2;
      ctx.bezierCurveTo(mx, prev.y, mx, curr.y, curr.x, curr.y);
    }
    ctx.stroke();
    ctx.restore();

    // Pulse dot on terminal point
    const lastPt = validPoints[validPoints.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lastPt.x, lastPt.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  });

  // 4. X-Axis Labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`U#${labels[0] || '0'}`, padLeft, h - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`U#${labels[labels.length - 1] || '0'}`, w - padRight, h - 8);

  // 5. Interactive Mouse Crosshair & Readout Tooltip
  const hoverX = chartHovers.get(canvasId);
  if (hoverX !== undefined && hoverX >= padLeft && hoverX <= w - padRight) {
    const fraction = (hoverX - padLeft) / plotW;
    const closestIdx = Math.min(labels.length - 1, Math.max(0, Math.round(fraction * (labels.length - 1))));
    const snapX = getX(closestIdx);

    // Vertical dashed marker
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(snapX, padTop);
    ctx.lineTo(snapX, padTop + plotH);
    ctx.stroke();
    ctx.restore();

    // Floating telemetry tooltip badge
    const badgeText = cleanSeries.map(s => {
      const v = s.values[closestIdx];
      return `${s.name || ''}: ${v !== null && v !== undefined ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3)) : '—'}`;
    }).join(' · ');
    const headerText = `Update ${labels[closestIdx]} | ${badgeText}`;

    ctx.font = '10px "JetBrains Mono", monospace';
    const textW = ctx.measureText(headerText).width + 16;
    const badgeX = Math.max(padLeft, Math.min(w - padRight - textW, snapX - textW / 2));
    const badgeY = padTop + 2;

    ctx.fillStyle = 'rgba(6, 14, 34, 0.92)';
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, textW, 20, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(headerText, badgeX + textW / 2, badgeY + 10);
  }

  ctx.restore();
}

// Attach hover listener for interactive tooltips
function setupChartInteractivity(canvasId) {
  const canvas = $(canvasId);
  if (!canvas || canvas._hasListener) return;
  canvas._hasListener = true;
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    chartHovers.set(canvasId, e.clientX - rect.left);
    if (latestTrainingData) render(latestTrainingData);
  });
  canvas.addEventListener('mouseleave', () => {
    chartHovers.delete(canvasId);
    if (latestTrainingData) render(latestTrainingData);
  });
}

let logClearedAt = 0;

function render(d) {
  const r = d.report || {}, history = d.history || [];
  $('trainStatus').textContent = (d.status || 'idle').toUpperCase();
  $('trainDot').style.background = ['starting', 'training', 'stopping'].includes(d.status) ? '#ffb703' : d.status === 'error' ? '#ff2a5f' : '#05ffa1';
  $('device').textContent = r.device || 'CPU';
  
  const checkpointLabel = {
    compatible: 'compatible checkpoint',
    missing: 'no checkpoint',
    incompatible_weather_model: 'checkpoint uses an older weather model',
    incompatible_navigation_model: 'checkpoint uses an older action set · retrain required',
    legacy_unversioned: 'checkpoint lacks weather provenance',
    invalid: 'checkpoint could not be loaded'
  }[r.checkpoint_status] || 'checkpoint status unknown';
  
  $('deviceMeta').textContent = `${r.cuda_available ? 'CUDA enabled' : 'CPU'} · ${checkpointLabel}`;
  $('episodes').textContent = (r.episodes || 0).toLocaleString();
  $('updates').textContent = (r.updates || 0).toLocaleString();
  $('success').textContent = `${((r.success_rate || 0) * 100).toFixed(1)}%`;
  
  const last = history[history.length - 1] || {};
  $('safeActions').textContent = last.mean_safe_actions?.toFixed(2) || '—';
  $('noSafeActions').textContent = `Recovery states: ${last.no_safe_action_steps || 0} · recovery moves: ${last.safe_recovery_maneuvers || 0}`;
  $('sps').textContent = last.steps_per_sec ? Math.round(last.steps_per_sec).toLocaleString() : '0';
  $('currentLr').textContent = last.learning_rate ? Number(last.learning_rate).toExponential(2) : '—';
  if ($('explVar')) $('explVar').textContent = last.explained_variance !== undefined ? last.explained_variance.toFixed(2) : '—';
  if ($('policyEntropy')) $('policyEntropy').textContent = last.entropy !== undefined ? last.entropy.toFixed(3) : '—';
  $('params').textContent = `${(r.parameter_count || '175,233').toLocaleString()} parameters`;
  
  if (r.elapsed_s) $('elapsed').textContent = `${Number(r.elapsed_s).toFixed(1)} s`;
  if (r.device) $('reportText').textContent = `Total: ${Number(r.transitions || 0).toLocaleString()} transitions · mean return ${Number(r.mean_return || 0).toFixed(3)}`;
  
  const run = r.training_run;
  $('runMetadata').textContent = run
    ? `Run ${run.run_id} · seed ${run.seed} · target ${Number(run.requested_steps || 0).toLocaleString()} transitions · ${run.parallel_environments} envs · LR ${run.learning_rate} · ${run.status || 'in progress'} · NAVRUNA ${run.app_version}`
    : 'No persisted run configuration is available yet.';
    
  const logs = (d.logs || []).filter(entry => Date.parse(entry.time) > logClearedAt);
  $('aiLog').textContent = logs.map(entry => `[${entry.time}] ${entry.kind.toUpperCase()}  ${entry.message}`).join('\n') || 'Waiting for training activity...';
  
  const chartHistory = run?.run_id ? history.filter(point => point.run_id === run.run_id) : history;
  const labels = chartHistory.map(point => String(point.update));

  // Initialize interactive listeners
  ['returnChart', 'successChart', 'lossChart', 'entropyChart'].forEach(setupChartInteractivity);

  // 1. Mean Return Chart (with smooth area gradient)
  drawAdvancedChart('returnChart', [
    { name: 'Return', values: chartHistory.map(p => p.mean_return), color: '#00f2fe', fillColor: 'rgba(0, 242, 254, 0.28)' }
  ], labels);

  // 2. Success Rate Chart (with 80% & 95% target indicator)
  drawAdvancedChart('successChart', [
    { name: 'Success %', values: chartHistory.map(p => p.success_rate * 100), color: '#05ffa1', fillColor: 'rgba(5, 255, 161, 0.25)' }
  ], labels, { targetLine: 90, minVal: 0, maxVal: 100 });

  // 3. Loss Dynamics (Policy loss & Value loss dual plot)
  drawAdvancedChart('lossChart', [
    { name: 'Policy Loss', values: chartHistory.map(p => p.loss_policy), color: '#00f2fe', fillArea: false },
    { name: 'Value Loss', values: chartHistory.map(p => p.loss_value), color: '#ffb703', fillArea: false }
  ], labels);

  // 4. Entropy & Trust Region Stability
  drawAdvancedChart('entropyChart', [
    { name: 'Entropy', values: chartHistory.map(p => p.entropy), color: '#818cf8', fillArea: false },
    { name: 'KL Div', values: chartHistory.map(p => p.approx_kl), color: '#ff2a5f', fillArea: false }
  ], labels);
}

let refreshInFlight = false, latestTrainingData = null;

function showTrainingFeedback(message, state = 'info') {
  const feedback = $('trainingFeedback');
  feedback.textContent = message;
  feedback.dataset.state = state;
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const data = await get('/api/simulation/training');
    latestTrainingData = data;
    render(data);
    const running = ['starting', 'training', 'stopping'].includes(data.status);
    $('startTrain').disabled = running;
    $('stopTrain').disabled = !running;
    $('stopTrain').textContent = data.status === 'stopping' ? 'STOPPING…' : 'STOP TRAINING';
    $('exportTraining').disabled = !data.report?.training_run;
    if (data.status === 'error' && data.report?.error) {
      showTrainingFeedback(`Training failed: ${data.report.error}`, 'error');
    }
  } catch (error) {
    console.error('NAVRUNA training status:', error);
    $('reportText').textContent = `Training status unavailable: ${error.message}`;
    showTrainingFeedback(`Training status unavailable: ${error.message}`, 'error');
  } finally {
    refreshInFlight = false;
  }
}

$('exportTraining').onclick = () => {
  if (!latestTrainingData) return;
  const run = latestTrainingData.report?.training_run, id = run?.run_id || 'latest';
  const url = URL.createObjectURL(new Blob([JSON.stringify(latestTrainingData, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `navruna-training-${id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

$('startTrain').onclick = async () => {
  const b = $('startTrain');
  b.disabled = true;
  b.textContent = 'STARTING…';
  showTrainingFeedback('Submitting PPO training run to backend worker…');
  try {
    const request = {
      steps: Number($('steps').value),
      envs: Number($('envs').value),
      rollout: 128,
      learning_rate: Number($('lr').value),
      continuous: $('continuous').checked
    };
    const result = await post('/api/simulation/train', request);
    if (!result.accepted) {
      const message = result.message || 'Training was not started';
      $('reportText').textContent = message;
      showTrainingFeedback(message, 'error');
      return;
    }
    showTrainingFeedback(`Training session accepted · ${Number(result.requested_steps || request.steps).toLocaleString()} transitions queued across ${request.envs} parallel workers.`, 'success');
    await refresh();
  } catch (e) {
    console.error('NAVRUNA training start:', e);
    $('reportText').textContent = `Training failed: ${e.message}`;
    $('trainStatus').textContent = 'ERROR';
    showTrainingFeedback(`Training failed: ${e.message}`, 'error');
  } finally {
    b.disabled = false;
    b.textContent = 'START TRAINING';
    refresh();
  }
};

$('stopTrain').onclick = async () => {
  const b = $('stopTrain');
  b.disabled = true;
  b.textContent = 'STOPPING…';
  try {
    const result = await post('/api/simulation/train/stop', {});
    $('reportText').textContent = result.message || result.status;
    await refresh();
  } catch (e) {
    $('reportText').textContent = `Could not stop training: ${e.message}`;
  } finally {
    refresh();
  }
};

$('refresh').onclick = refresh;
$('clearLog').onclick = () => {
  logClearedAt = Date.now();
  $('aiLog').textContent = 'Log view cleared. New telemetry will appear here.';
};

refresh();
setInterval(refresh, 1500);
window.addEventListener('resize', refresh);
