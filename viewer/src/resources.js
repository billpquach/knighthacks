const fields = [
  ['memory', 'Shared memory', 'ram_used_gb', 'GiB', '#bcbcbc'],
  ['gpu', 'GPU', 'gpu_pct', '%', '#76b900'],
  ['cpu', 'CPU', 'cpu_pct', '%', '#dddddd'],
  ['temperature', 'Temperature', 'temp_c', 'C', '#f4d06c'],
  ['power', 'Board power', 'power_w', 'W', '#f2ad91'],
  ['swap', 'Swap', 'swap_used_mb', 'MiB', '#929292'],
];

export function normalizeMetrics(raw) {
  if (raw?.version !== 1 || raw.memory_kind !== 'shared' || typeof raw.available !== 'boolean') throw new Error('Invalid metrics');
  if (!raw.available) return null;
  if (typeof raw.device !== 'string' || !raw.device || typeof raw.sample_id !== 'string'
    || !Number.isFinite(raw.age_ms) || raw.age_ms < 0 || raw.age_ms >= 5000) throw new Error('Stale metrics');
  const m = raw.metrics || {};
  const values = {};
  for (const key of ['ram_used_gb', 'ram_total_gb', 'gpu_pct', 'cpu_pct', 'temp_c', 'power_w', 'swap_used_mb']) {
    const value = m[key];
    values[key] = Number.isFinite(value) && value >= 0 && (!key.endsWith('_pct') || value <= 100) ? value : null;
  }
  if (!values.ram_total_gb || values.ram_used_gb > values.ram_total_gb) values.ram_used_gb = null;
  return { ...raw, metrics: values };
}

export function metricsUrl(value, source) {
  const url = new URL(value.trim() || `${source}/metrics`);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Use an HTTP(S) metrics URL without credentials.');
  return url.href;
}

export class ResourcePanel {
  constructor(root) {
    this.root = root;
    this.rows = fields.map(([id, label, key, unit, color]) => {
      const row = document.createElement('div'); row.className = 'resource-metric'; row.dataset.metric = id;
      const name = document.createElement('span'); name.className = 'resource-name'; name.textContent = label;
      if (id === 'memory') name.title = 'Physical memory shared by CPU and GPU; not dedicated VRAM';
      const value = document.createElement('strong'); value.textContent = '--';
      const bar = document.createElement('progress'); bar.max = 100; bar.hidden = !['memory', 'gpu', 'cpu'].includes(id);
      bar.setAttribute('aria-label', `${label} utilization`); bar.style.setProperty('--metric-color', color);
      const canvas = document.createElement('canvas'); canvas.width = 240; canvas.height = 28;
      canvas.setAttribute('aria-label', `${label}, last 60 seconds`); canvas.setAttribute('role', 'img');
      canvas.title = `${label}, last 60 seconds`;
      row.append(name, value, bar, canvas); root.querySelector('.resource-metrics').append(row);
      return { id, key, unit, color, row, value, bar, canvas, history: [] };
    });
    this.status('NOT CONNECTED');
  }

  status(text) {
    const status = this.root.querySelector('.resource-status');
    if (status.textContent !== text) status.textContent = text;
    this.root.dataset.status = text.toLowerCase();
  }

  clear(text) {
    this.status(text);
    for (const row of this.rows) {
      row.value.textContent = '--'; row.bar.value = 0; row.bar.setAttribute('aria-valuetext', 'Unavailable');
      row.history = []; this.draw(row);
    }
  }

  draw(row) {
    const ctx = row.canvas.getContext('2d');
    ctx.clearRect(0, 0, 240, 28);
    const valid = row.history.filter(p => p.value !== null);
    if (!valid.length) return;
    const max = ['memory', 'gpu', 'cpu'].includes(row.id) ? 100 : Math.max(...valid.map(p => p.value), 1) * 1.15;
    const now = performance.now(); ctx.strokeStyle = row.color; ctx.lineWidth = 1.5; ctx.beginPath();
    let connected = false, last = 0;
    for (const point of row.history) {
      if (point.value === null) { connected = false; continue; }
      const x = 240 - (now - point.t) / 60000 * 240, y = 26 - point.value / max * 24;
      if (!connected || point.t - last > 2500) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      connected = true; last = point.t;
    }
    ctx.stroke();
    const latest = valid[valid.length - 1];
    ctx.fillStyle = row.color; ctx.beginPath();
    ctx.arc(Math.min(238, 240 - (now - latest.t) / 60000 * 240), 26 - latest.value / max * 24, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }

  connect(url) {
    const token = this.token = (this.token || 0) + 1;
    clearTimeout(this.timer); this.controller?.abort();
    this.lastId = null; this.changedAt = 0;
    this.clear('CONNECTING');
    this.root.querySelector('.resource-heading').title = url;
    const poll = async () => {
      if (token !== this.token) return;
      if (document.hidden) { this.clear('PAUSED'); this.timer = setTimeout(poll, 1000); return; }
      this.controller = new AbortController();
      const timeout = setTimeout(() => this.controller?.abort(), 2500);
      try {
        const response = await fetch(url, { cache: 'no-store', signal: this.controller.signal });
        if (!response.ok) throw new Error('Unavailable');
        const raw = await response.json();
        if (token !== this.token) return;
        const sample = normalizeMetrics(raw);
        if (!sample) { this.clear(raw.reason === 'stale' ? 'STALE' : 'NOT CONNECTED'); return; }
        const now = performance.now();
        const fresh = sample.sample_id !== this.lastId;
        if (fresh) { this.lastId = sample.sample_id; this.changedAt = now; }
        if (now - this.changedAt >= 5000) { this.clear('STALE'); return; }
        this.root.querySelector('.resource-heading').title = `${sample.device} | ${url}`;
        this.status(Object.values(sample.metrics).some(v => v === null) ? 'PARTIAL' : 'LIVE');
        for (const row of this.rows) {
          const value = sample.metrics[row.key];
          row.value.textContent = value === null ? '--' : row.id === 'memory'
            ? `${value.toFixed(1)} / ${sample.metrics.ram_total_gb.toFixed(1)} GiB`
            : `${value.toFixed(row.unit === '%' || row.unit === 'MiB' ? 0 : 1)} ${row.unit}`;
          row.bar.value = value === null ? 0 : row.id === 'memory' ? value / sample.metrics.ram_total_gb * 100 : value;
          row.bar.setAttribute('aria-valuetext', value === null ? 'Unavailable' : `${row.bar.value.toFixed(0)} percent`);
          if (fresh) row.history.push({ t: now, value: row.id === 'memory' && value !== null ? row.bar.value : value });
          row.history = row.history.filter(p => now - p.t <= 60000);
          this.draw(row);
        }
      } catch (error) {
        if (token === this.token) this.clear(error.message === 'Stale metrics' ? 'STALE' : 'UNAVAILABLE');
      } finally {
        clearTimeout(timeout);
        if (token === this.token) this.timer = setTimeout(poll, 1000);
      }
    };
    poll();
  }
}
