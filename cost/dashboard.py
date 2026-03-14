"""
CostDashboard: serves a live HTML dashboard showing cost metrics.

Add to any FastAPI app:
    from cost.dashboard import mount_dashboard
    mount_dashboard(app, store, alerter)

Then visit http://localhost:8000/cost/dashboard
"""

from __future__ import annotations
from typing import Optional

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FRAMEWORM COST</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0c0906; --bg-card: #130e09; --border: #2a1e12;
    --orange: #CC6B2C; --amber: #d4a55a; --cream: #f5eedf;
    --cream-dim: #c8b99a; --cream-muted: #8a7a65;
    --green: #7ab87a; --red: #c85a5a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--cream); font-family: system-ui,-apple-system,sans-serif; min-height: 100vh; }
  header { padding: 24px 40px; border-bottom: 0.5px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
  header h1 span { color: var(--orange); }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 8px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; box-shadow: 0 0 8px var(--green); } 50% { opacity:0.4; box-shadow: none; } }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 32px 40px 0; }
  .card { background: var(--bg-card); border: 0.5px solid var(--border); border-radius: 14px; padding: 20px 22px; }
  .card .label { font-size: 11px; color: var(--cream-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px; font-family: monospace; }
  .card .value { font-size: 36px; font-weight: 700; line-height: 1; }
  .card .sub { font-size: 12px; color: var(--cream-muted); margin-top: 6px; font-family: monospace; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 40px; }
  .chart-card { background: var(--bg-card); border: 0.5px solid var(--border); border-radius: 14px; padding: 20px; }
  .chart-card h3 { font-size: 11px; color: var(--cream-muted); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 16px; font-family: monospace; }
  .full-width { grid-column: 1 / -1; }
  .hints { padding: 0 40px 16px; }
  .hint-card { background: var(--bg-card); border: 0.5px solid var(--border); border-radius: 14px; padding: 20px 24px; margin-bottom: 12px; }
  .hint-card .hint-title { font-size: 14px; font-weight: 600; color: var(--cream); margin-bottom: 6px; }
  .hint-card .hint-desc { font-size: 13px; color: var(--cream-muted); line-height: 1.6; }
  .hint-card .hint-saving { font-size: 13px; color: var(--green); margin-top: 6px; font-family: monospace; }
  .effort { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; margin-left: 8px; }
  .effort.low { background: rgba(122,184,122,0.12); color: var(--green); }
  .effort.medium { background: rgba(212,165,90,0.12); color: var(--amber); }
  .effort.high { background: rgba(200,90,90,0.12); color: var(--red); }
  .section-title { font-size: 11px; color: var(--cream-muted); letter-spacing: 0.12em; text-transform: uppercase; padding: 24px 40px 12px; font-family: monospace; }
  .alert-row { display: flex; align-items: flex-start; gap: 14px; padding: 14px 24px; border-bottom: 0.5px solid var(--border); }
  .alert-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
  .alert-dot.critical { background: var(--red); }
  .alert-dot.warning { background: var(--amber); }
  .alert-msg { font-size: 13px; color: var(--cream-muted); line-height: 1.5; }
  .alert-time { font-size: 11px; color: var(--border); font-family: monospace; margin-top: 3px; }
  .empty { text-align: center; padding: 32px; color: var(--cream-muted); font-size: 13px; }
  footer { padding: 24px 40px; border-top: 0.5px solid var(--border); margin-top: 32px; color: var(--cream-muted); font-size: 12px; font-family: monospace; }
</style>
</head>
<body>

<header>
  <h1>FRAMEWORM <span>COST</span></h1>
  <div style="display:flex;align-items:center;gap:20px">
    <span style="font-size:13px;color:var(--cream-muted);font-family:monospace">
      <span class="live-dot"></span>Live
    </span>
    <span id="last-update" style="font-size:12px;color:var(--cream-muted);font-family:monospace">--</span>
  </div>
</header>

<div class="grid">
  <div class="card">
    <div class="label">Total requests</div>
    <div class="value" id="total-requests" style="color:var(--cream)">--</div>
    <div class="sub" id="model-name">loading...</div>
  </div>
  <div class="card">
    <div class="label">Avg cost / request</div>
    <div class="value" id="avg-cost" style="color:var(--orange)">--</div>
    <div class="sub">per inference call</div>
  </div>
  <div class="card">
    <div class="label">Projected monthly</div>
    <div class="value" id="monthly" style="color:var(--red)">--</div>
    <div class="sub">at current rate · 10 rps</div>
  </div>
  <div class="card">
    <div class="label">Avg latency</div>
    <div class="value" id="avg-latency" style="color:var(--amber)">--</div>
    <div class="sub" id="p95-latency">p95: --</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>Cost per request (last 50)</h3>
    <canvas id="cost-chart" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Latency ms (last 50)</h3>
    <canvas id="latency-chart" height="180"></canvas>
  </div>
  <div class="chart-card full-width">
    <h3>Cost by model</h3>
    <canvas id="model-chart" height="120"></canvas>
  </div>
</div>

<div class="section-title">💡 Savings Opportunities</div>
<div class="hints" id="hints-container">
  <div class="empty">No data yet — run some predictions first.</div>
</div>

<div class="section-title">⚠️ Recent Alerts</div>
<div class="card" style="margin: 0 40px 16px; padding: 0;" id="alerts-container">
  <div class="empty">No alerts fired.</div>
</div>

<footer>FRAMEWORM-COST · github.com/aakash0440/frameworm · auto-refresh 5s</footer>

<script>
const FMT = {
  cost: v => v < 0.0001 ? `$${(v*1e6).toFixed(2)}µ` : `$${v.toFixed(6)}`,
  ms:   v => `${v.toFixed(0)}ms`,
  usd:  v => v >= 1000 ? `$${(v/1000).toFixed(1)}k` : `$${v.toFixed(0)}`,
};

const chartDefaults = {
  responsive: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { display: false },
    y: { ticks: { color: '#8a7a65', font: { family: 'monospace', size: 11 } }, grid: { color: '#2a1e12' } }
  },
  elements: { point: { radius: 2 }, line: { tension: 0.3 } },
  animation: { duration: 300 },
};

const costChart = new Chart(document.getElementById('cost-chart'), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: '#CC6B2C', backgroundColor: 'rgba(204,107,44,0.08)', fill: true, borderWidth: 1.5 }] },
  options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `$${v.toFixed(6)}` } } } },
});

const latencyChart = new Chart(document.getElementById('latency-chart'), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: '#d4a55a', backgroundColor: 'rgba(212,165,90,0.08)', fill: true, borderWidth: 1.5 }] },
  options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `${v}ms` } } } },
});

const modelChart = new Chart(document.getElementById('model-chart'), {
  type: 'bar',
  data: { labels: [], datasets: [{ data: [], backgroundColor: '#CC6B2C', borderRadius: 6 }] },
  options: { ...chartDefaults, scales: { ...chartDefaults.scales, x: { display: true, ticks: { color: '#8a7a65', font: { family: 'monospace', size: 11 } }, grid: { color: '#2a1e12' } } } },
});

async function refresh() {
  try {
    const [summary, records, hints, alerts] = await Promise.all([
      fetch('/cost/summary').then(r => r.json()),
      fetch('/cost/records').then(r => r.json()),
      fetch('/cost/hints').then(r => r.json()),
      fetch('/cost/alerts').then(r => r.json()).catch(() => ({ total_alerts: 0, by_type: {} })),
    ]);

    // Stat cards
    document.getElementById('total-requests').textContent = summary.total_requests ?? '--';
    document.getElementById('avg-cost').textContent = summary.avg_cost_usd ? FMT.cost(summary.avg_cost_usd) : '--';
    document.getElementById('monthly').textContent = summary.projected_monthly_10rps_usd ? FMT.usd(summary.projected_monthly_10rps_usd) : '--';
    document.getElementById('avg-latency').textContent = summary.avg_latency_ms ? FMT.ms(summary.avg_latency_ms) : '--';
    document.getElementById('p95-latency').textContent = `p95: ${summary.p95_latency_ms ? FMT.ms(summary.p95_latency_ms) : '--'}`;

    // Model name
    const models = Object.keys(summary.by_model ?? {});
    if (models.length) document.getElementById('model-name').textContent = models.join(', ');

    // Charts
    const recs = (records.records ?? []).slice(-50);
    costChart.data.labels = recs.map((_, i) => i);
    costChart.data.datasets[0].data = recs.map(r => r.total_cost_usd);
    costChart.update();

    latencyChart.data.labels = recs.map((_, i) => i);
    latencyChart.data.datasets[0].data = recs.map(r => r.latency_ms);
    latencyChart.update();

    // Model chart
    const byModel = summary.by_model ?? {};
    modelChart.data.labels = Object.keys(byModel);
    modelChart.data.datasets[0].data = Object.values(byModel).map(m => m.total_cost_usd);
    modelChart.update();

    // Hints
    const hintsEl = document.getElementById('hints-container');
    const opps = hints.savings_opportunities ?? [];
    if (opps.length) {
      hintsEl.innerHTML = opps.map(o => `
        <div class="hint-card">
          <div class="hint-title">${o.title} <span class="effort ${o.effort}">${o.effort}</span></div>
          <div class="hint-desc">${o.description}</div>
          <div class="hint-saving">Est. saving: $${o.estimated_monthly_saving_usd.toLocaleString()}/month</div>
        </div>
      `).join('');
    }

    // Alerts
    const alertsEl = document.getElementById('alerts-container');
    if (alerts.total_alerts > 0) {
      const hist = alerts.history ?? [];
      alertsEl.innerHTML = hist.slice(-10).reverse().map(a => `
        <div class="alert-row">
          <div class="alert-dot ${a.level}"></div>
          <div>
            <div class="alert-msg">${a.message}</div>
            <div class="alert-time">${new Date(a.timestamp * 1000).toLocaleTimeString()}</div>
          </div>
        </div>
      `).join('') || '<div class="empty">No alerts fired.</div>';
    }

    document.getElementById('last-update').textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    console.error('Refresh failed:', e);
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def mount_dashboard(app, store, alerter=None) -> None:
    """
    Mount the cost dashboard onto a FastAPI app.

    Usage:
        from cost.dashboard import mount_dashboard
        mount_dashboard(app, tracker.store, alerter)

    Visit: http://localhost:8000/cost/dashboard
    """
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.requests import Request

    @app.get("/cost/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return HTMLResponse(DASHBOARD_HTML)

    if alerter:
        @app.get("/cost/alerts")
        def alerts_endpoint():
            data = alerter.summary()
            data["history"] = [a.to_dict() for a in alerter.alert_history[-50:]]
            return JSONResponse(data)
