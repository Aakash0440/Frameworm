
"""
Generates HTML and JSON drift reports.
HTML report shows per-feature histogram overlays + severity badges.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from shift.core.drift_engine import DriftResult, DriftSeverity
from shift.core.feature_profiles import DatasetProfile


class ReportGenerator:

    def generate_html(
        self,
        result: DriftResult,
        reference: DatasetProfile,
        current: DatasetProfile,
        output_path: str = "shift_report.html",
        model_name: str = "model",
    ) -> str:
        html = self._build_html(result, reference, current, model_name)
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    def generate_json(
        self,
        result: DriftResult,
        output_path: str = "shift_report.json",
    ) -> str:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return output_path

    # ──────────────────────────────────────────────── HTML builder

    def _build_html(
        self,
        result: DriftResult,
        reference: DatasetProfile,
        current: DatasetProfile,
        model_name: str,
    ) -> str:
        severity_colours = {
            "NONE":   "#22c55e",
            "LOW":    "#eab308",
            "MEDIUM": "#f97316",
            "HIGH":   "#ef4444",
        }
        sev = result.overall_severity.value
        sev_colour = severity_colours.get(sev, "#6b7280")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        feature_cards = "\n".join(
            self._feature_card(name, report, reference, current, severity_colours)
            for name, report in sorted(
                result.features.items(),
                key=lambda x: x[1].severity.value,
                reverse=True,
            )
        )

        drifted_n = len(result.drifted_features)
        total_n   = result.n_features_checked

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FRAMEWORM SHIFT — {model_name}</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8;
    --ref: #3b82f6; --cur: #f97316;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: 'Inter', system-ui, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }}
  .badge {{ display: inline-block; padding: 0.25rem 0.75rem;
            border-radius: 9999px; font-size: 0.75rem;
            font-weight: 700; color: #fff; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
                   gap: 1rem; margin-bottom: 2rem; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border);
               border-radius: 0.75rem; padding: 1.25rem; }}
  .stat-label {{ font-size: 0.75rem; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; }}
  .features-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                    gap: 1rem; }}
  .feature-card {{ background: var(--surface); border: 1px solid var(--border);
                   border-radius: 0.75rem; padding: 1.25rem; }}
  .feature-name {{ font-weight: 600; margin-bottom: 0.5rem; }}
  .feature-meta {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 0.75rem; }}
  .bar-chart {{ display: flex; align-items: flex-end; gap: 2px;
                height: 60px; margin: 0.5rem 0; }}
  .bar-wrap {{ flex: 1; display: flex; flex-direction: column;
               align-items: center; gap: 1px; }}
  .bar {{ width: 100%; border-radius: 2px 2px 0 0; min-height: 2px; }}
  .bar.ref {{ background: var(--ref); opacity: 0.7; }}
  .bar.cur {{ background: var(--cur); opacity: 0.7; }}
  .legend {{ display: flex; gap: 1rem; font-size: 0.7rem; color: var(--muted); }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%;
                  display: inline-block; margin-right: 4px; }}
  .metric-row {{ display: flex; justify-content: space-between;
                 font-size: 0.8rem; padding: 0.2rem 0; border-bottom: 1px solid var(--border); }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ color: var(--muted); }}
  .delta-pos {{ color: #f97316; }} .delta-neg {{ color: #3b82f6; }}
  section h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 1rem;
                color: var(--muted); text-transform: uppercase;
                letter-spacing: 0.05em; }}
</style>
</head>
<body>

<h1>⚡ FRAMEWORM SHIFT</h1>
<p class="subtitle">Model: <strong>{model_name}</strong> &nbsp;·&nbsp; {timestamp}</p>

<div class="summary-grid">
  <div class="stat-card">
    <div class="stat-label">Overall Status</div>
    <div class="stat-value">
      <span class="badge" style="background:{sev_colour}">{sev}</span>
    </div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Features Drifted</div>
    <div class="stat-value" style="color:{sev_colour}">{drifted_n} / {total_n}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Drift Fraction</div>
    <div class="stat-value">{result.drift_fraction*100:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Reference Samples</div>
    <div class="stat-value">{reference.n_samples:,}</div>
  </div>
</div>

<section>
<h2>Per-Feature Breakdown</h2>
<div class="features-grid">
{feature_cards}
</div>
</section>

<p style="margin-top:2rem;font-size:0.75rem;color:#475569">
  Generated by FRAMEWORM SHIFT v0.1.0
</p>
</body>
</html>"""

    def _feature_card(self, name, report, reference, current, colours) -> str:
        sev = report.severity.value
        colour = colours.get(sev, "#6b7280")
        test_info = f"{report.test_used} · stat={report.statistic:.4f} · p={report.p_value:.4f}"

        # Build histogram bars if numerical
        hist_html = ""
        if report.feature_type == "numerical" and name in reference.numerical:
            ref_counts = reference.numerical[name].histogram_counts
            cur_counts = current.numerical[name].histogram_counts if name in current.numerical else []
            hist_html = self._histogram_html(ref_counts, cur_counts)

        # Delta metrics
        delta_rows = ""
        if report.mean_delta is not None:
            sign = "pos" if report.mean_delta > 0 else "neg"
            delta_rows += f"""
            <div class="metric-row">
              <span class="metric-label">Mean delta</span>
              <span class="delta-{sign}">{report.mean_delta:+.4f}</span>
            </div>"""
        if report.std_delta is not None:
            sign = "pos" if report.std_delta > 0 else "neg"
            delta_rows += f"""
            <div class="metric-row">
              <span class="metric-label">Std delta</span>
              <span class="delta-{sign}">{report.std_delta:+.4f}</span>
            </div>"""
        if report.missing_rate_delta is not None:
            delta_rows += f"""
            <div class="metric-row">
              <span class="metric-label">Missing rate Δ</span>
              <span>{report.missing_rate_delta:+.4f}</span>
            </div>"""

        drifted_icon = "🔴" if report.drifted else "🟢"

        return f"""<div class="feature-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div class="feature-name">{drifted_icon} {name}</div>
    <span class="badge" style="background:{colour};font-size:0.65rem">{sev}</span>
  </div>
  <div class="feature-meta">{test_info} · {report.feature_type}</div>
  {hist_html}
  {delta_rows}
</div>"""

    def _histogram_html(self, ref_counts, cur_counts) -> str:
        if not ref_counts:
            return ""
        max_val = max(max(ref_counts, default=1), max(cur_counts, default=1), 1)
        bins = min(len(ref_counts), len(cur_counts)) if cur_counts else len(ref_counts)
        bars = ""
        for i in range(bins):
            rh = int((ref_counts[i] / max_val) * 56) + 2
            ch = int((cur_counts[i] / max_val) * 56) + 2 if cur_counts else 0
            bars += f"""<div class="bar-wrap">
              <div class="bar ref" style="height:{rh}px"></div>
              <div class="bar cur" style="height:{ch}px"></div>
            </div>"""
        return f"""<div class="bar-chart">{bars}</div>
<div class="legend">
  <span><span class="legend-dot" style="background:#3b82f6"></span>Reference</span>
  <span><span class="legend-dot" style="background:#f97316"></span>Current</span>
</div>"""

