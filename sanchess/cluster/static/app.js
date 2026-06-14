// Dashboard public San-o1 Cluster. Aucune dépendance externe : graphiques en canvas
// natif (compatible Railway / hors-ligne). Poll /cluster/stats (live) + /cluster/history.
'use strict';

const fmt = (n) => Math.round(n ?? 0).toLocaleString('fr-FR');
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function humanDuration(sec) {
  sec = Math.max(0, Math.floor(sec));
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d) return `${d}j ${h}h`;
  if (h) return `${h}h ${m}min`;
  if (m) return `${m}min`;
  return `${sec}s`;
}

function timeAgo(epoch) {
  if (!epoch) return 'jamais';
  return 'il y a ' + humanDuration(Date.now() / 1000 - epoch);
}

// --- Graphique en ligne (canvas, responsive, HiDPI) --------------------------
function drawChart(canvas, pts, { color = '#6ea8fe', fill = true, area = false } = {}) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.parentNode.clientWidth || 300;
  const cssH = canvas.height;                 // hauteur CSS fixée par l'attribut
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 6, r: 6, t: 10, b: 10 };
  const W = cssW - pad.l - pad.r, H = cssH - pad.t - pad.b;

  if (!pts || pts.length === 0) {
    ctx.fillStyle = '#8b94a7'; ctx.font = '13px system-ui'; ctx.textAlign = 'center';
    ctx.fillText('en attente de données…', cssW / 2, cssH / 2);
    return;
  }
  const ys = pts.map((p) => p.v);
  const maxV = Math.max(1, ...ys), minV = Math.min(0, ...ys);
  const n = pts.length;
  const X = (i) => pad.l + (n === 1 ? W / 2 : (i / (n - 1)) * W);
  const Y = (v) => pad.t + H - ((v - minV) / (maxV - minV || 1)) * H;

  // grille horizontale
  ctx.strokeStyle = '#232834'; ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g++) {
    const y = pad.t + (g / 3) * H;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(cssW - pad.r, y); ctx.stroke();
  }

  // aire
  if (area) {
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + H);
    grad.addColorStop(0, color + '55'); grad.addColorStop(1, color + '00');
    ctx.beginPath(); ctx.moveTo(X(0), Y(ys[0]));
    pts.forEach((p, i) => ctx.lineTo(X(i), Y(p.v)));
    ctx.lineTo(X(n - 1), pad.t + H); ctx.lineTo(X(0), pad.t + H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
  }

  // ligne
  ctx.beginPath(); ctx.moveTo(X(0), Y(ys[0]));
  pts.forEach((p, i) => ctx.lineTo(X(i), Y(p.v)));
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();

  // dernier point
  const lx = X(n - 1), ly = Y(ys[n - 1]);
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(lx, ly, 3, 0, 7); ctx.fill();
}

// --- Rafraîchissements -------------------------------------------------------
let lastHistory = 0;
let histRate = [], histCum = [];

async function refreshStats() {
  try {
    const r = await fetch('/cluster/stats', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    setOnline(true, s);

    $('model-version').textContent = s.has_model ? 'v' + fmt(s.model_version) : '—';
    $('model-step').textContent = fmt(s.model_step);
    $('total-samples').textContent = fmt(s.total_samples);
    $('total-games').textContent = fmt(s.total_games);
    $('active-workers').textContent = fmt(s.active_workers);
    $('total-contributors').textContent = fmt(s.total_contributors);
    $('games-min').textContent = fmt(s.games_per_min);
    $('pending').textContent = fmt(s.pending_shards);

    renderDevices(s.device_breakdown || []);
    renderLeaderboard(s.leaderboard || []);

    $('footer-status').textContent =
      `${fmt(s.samples_last_hour)} positions · ${fmt(s.games_last_hour)} parties (1 h)`;
    $('footer-uptime').textContent =
      `serveur up ${humanDuration(s.uptime_sec)} · modèle publié ${timeAgo(s.model_published_at)}`;
  } catch (e) {
    setOnline(false);
  }
}

async function refreshHistory() {
  try {
    const r = await fetch('/cluster/history?minutes=180', { cache: 'no-store' });
    if (!r.ok) return;
    const data = await r.json();
    const series = data.series || [];
    histRate = series.map((p) => ({ v: p.games }));
    histCum = series.map((p) => ({ v: p.cum_samples }));
    if (series.length) {
      $('chart-rate-now').textContent = fmt(series[series.length - 1].games) + '/min';
      $('chart-cum-now').textContent = '+' + fmt(series.reduce((a, p) => a + p.samples, 0)) + ' (3 h)';
    }
    drawAll();
  } catch (e) { /* silencieux : le live continue */ }
}

function drawAll() {
  drawChart($('chart-rate'), histRate, { color: '#6ea8fe', area: true });
  drawChart($('chart-cum'), histCum, { color: '#4ade80', area: true });
}

function renderDevices(devs) {
  const box = $('devices');
  const known = devs.filter((d) => d.samples > 0 || d.workers > 0);
  if (!known.length) { box.innerHTML = '<p class="muted">Aucun contributeur pour l\'instant.</p>'; return; }
  const max = Math.max(1, ...known.map((d) => d.samples));
  const label = (d) => ({ cuda: 'GPU', mps: 'M1', cpu: 'CPU' }[d] || d);
  box.innerHTML = known.map((d) => {
    const cls = ['cuda', 'mps', 'cpu'].includes(d.device) ? d.device : '';
    const w = Math.max(2, (d.samples / max) * 100);
    return `<div class="dev-row">
      <span class="dev-name ${cls}">${escapeHtml(label(d.device))}</span>
      <span class="dev-bar"><span style="width:${w}%"></span></span>
      <span class="dev-meta">${fmt(d.workers)} machine(s) · ${fmt(d.samples)} pos.</span>
    </div>`;
  }).join('');
}

function renderLeaderboard(rows) {
  const tbody = document.querySelector('#leaderboard tbody');
  const label = (d) => ({ cuda: 'GPU', mps: 'M1', cpu: 'CPU' }[d] || (d || '?'));
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">Sois le premier à contribuer !</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((row, i) =>
    `<tr><td>${i + 1}</td><td>${escapeHtml(row.name)}</td>` +
    `<td><span class="pill">${escapeHtml(label(row.device))}</span></td>` +
    `<td>${fmt(row.games)}</td><td>${fmt(row.samples)}</td></tr>`).join('');
}

function setOnline(ok, s) {
  const dot = $('live-dot'), txt = $('live-text');
  dot.className = 'dot ' + (ok ? 'on' : 'off');
  if (ok) {
    txt.textContent = s.active_workers
      ? `${fmt(s.active_workers)} worker(s) actif(s)`
      : (s.has_model ? `modèle v${fmt(s.model_version)}` : 'en attente de workers');
  } else {
    txt.textContent = 'hors ligne';
  }
}

// --- Init --------------------------------------------------------------------
const srv = $('server-url'); if (srv) srv.textContent = window.location.origin;
$('copy-btn')?.addEventListener('click', () => {
  navigator.clipboard.writeText($('join-cmd').innerText).then(() => {
    const b = $('copy-btn'); b.textContent = 'Copié ✓';
    setTimeout(() => (b.textContent = 'Copier la commande'), 1500);
  });
});
window.addEventListener('resize', drawAll);

refreshStats(); refreshHistory();
setInterval(refreshStats, 4000);
setInterval(refreshHistory, 12000);
