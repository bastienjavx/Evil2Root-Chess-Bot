// Dashboard San-o1 Cluster : poll /cluster/stats et rafraîchit l'affichage.
const fmt = (n) => (n ?? 0).toLocaleString('fr-FR');

function setServerUrl() {
  const span = document.getElementById('server-url');
  if (span) span.textContent = window.location.origin;
}

async function refresh() {
  const status = document.getElementById('status');
  try {
    const r = await fetch('/cluster/stats', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();

    document.getElementById('model-version').textContent = fmt(s.model_version);
    document.getElementById('model-step').textContent = fmt(s.model_step);
    document.getElementById('total-games').textContent = fmt(s.total_games);
    document.getElementById('total-samples').textContent = fmt(s.total_samples);
    document.getElementById('active-workers').textContent = fmt(s.active_workers);
    document.getElementById('games-min').textContent = fmt(Math.round(s.games_per_min));

    const tbody = document.querySelector('#leaderboard tbody');
    tbody.innerHTML = '';
    (s.leaderboard || []).forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${i + 1}</td><td>${escapeHtml(row.name)}</td>` +
        `<td>${escapeHtml(row.device || '?')}</td>` +
        `<td>${fmt(row.games)}</td><td>${fmt(row.samples)}</td>`;
      tbody.appendChild(tr);
    });
    if (!(s.leaderboard || []).length) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">Aucun contributeur pour l\'instant — sois le premier !</td></tr>';
    }
    const v = s.has_model ? `modèle v${s.model_version}` : 'aucun modèle publié';
    status.textContent = `À jour · ${v} · ${s.pending_shards} shards en attente d'entraînement`;
  } catch (e) {
    status.textContent = 'Hors ligne : ' + e.message;
  }
}

function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

document.getElementById('copy-btn')?.addEventListener('click', () => {
  const cmd = document.getElementById('join-cmd').innerText;
  navigator.clipboard.writeText(cmd).then(() => {
    const b = document.getElementById('copy-btn');
    b.textContent = 'Copié ✓';
    setTimeout(() => (b.textContent = 'Copier la commande'), 1500);
  });
});

setServerUrl();
refresh();
setInterval(refresh, 5000);
