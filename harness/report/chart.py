"""chart.html: one self-contained file (no CDN), light and dark themes, hover tooltips, table view.

Chart 1 answers the decision question: accuracy vs cost per correct answer (log x),
one dot per model x prompt, whisker = min-max accuracy across repeats.
Chart 2: latency p50 -> p95 range per cell, one shared ms axis.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from harness.report.analysis import RunData, cell_summary

# Categorical slots 1-3 of the reference palette (validated all-pairs in both modes).
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70"]
SHAPES = ["circle", "square", "triangle", "diamond"]

W, H = 720, 380
ML, MR, MT, MB = 64, 200, 24, 52


def _marker(shape: str, x: float, y: float, r: float = 6) -> str:
    if shape == "square":
        return f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" rx="2"/>'
    if shape == "triangle":
        return f'<path d="M{x} {y - r - 1}L{x + r + 1} {y + r}L{x - r - 1} {y + r}Z"/>'
    if shape == "diamond":
        return f'<path d="M{x} {y - r - 1}L{x + r + 1} {y}L{x} {y + r + 1}L{x - r - 1} {y}Z"/>'
    return f'<circle cx="{x}" cy="{y}" r="{r}"/>'


def _nice_log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10**e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks or [lo, hi]


def _fmt_usd(v: float) -> str:
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.01:
        return f"${v:.3f}"
    return f"${v:.2g}"


def _scatter(
    summary: pd.DataFrame, model_slot: dict[str, int], prompt_shape: dict[str, str]
) -> str:
    df = summary[summary.cost_per_correct_usd.apply(lambda v: math.isfinite(v) and v > 0)]
    if df.empty:
        return "<p>No cell has a correct answer, so cost per correct answer is undefined.</p>"
    lo, hi = df.cost_per_correct_usd.min(), df.cost_per_correct_usd.max()
    lo_l, hi_l = math.log10(lo) - 0.25, math.log10(hi) + 0.25
    pw, ph = W - ML - MR, H - MT - MB

    def x(v: float) -> float:
        return ML + (math.log10(v) - lo_l) / (hi_l - lo_l) * pw

    def y(v: float) -> float:
        return MT + (1 - v) * ph

    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Accuracy versus cost per correct answer">'
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{ML}" x2="{ML + pw}" y1="{y(t)}" y2="{y(t)}"/>')
        parts.append(
            f'<text class="tick" x="{ML - 8}" y="{y(t) + 4}" text-anchor="end">{t * 100:.0f}%</text>'
        )
    for t in _nice_log_ticks(10**lo_l, 10**hi_l):
        parts.append(f'<line class="grid" x1="{x(t)}" x2="{x(t)}" y1="{MT}" y2="{MT + ph}"/>')
        parts.append(
            f'<text class="tick" x="{x(t)}" y="{MT + ph + 18}" text-anchor="middle">{_fmt_usd(t)}</text>'
        )
    parts.append(f'<line class="axis" x1="{ML}" x2="{ML + pw}" y1="{MT + ph}" y2="{MT + ph}"/>')
    parts.append(
        f'<text class="label" x="{ML + pw / 2}" y="{H - 10}" text-anchor="middle">'
        "Cost per correct answer (USD, log scale) — further left is cheaper</text>"
    )
    parts.append(
        f'<text class="label" x="14" y="{MT + ph / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {MT + ph / 2})">Accuracy (all fields)</text>'
    )
    placed: list[float] = []
    for r in df.sort_values("accuracy", ascending=False).itertuples():
        cx, cy = x(r.cost_per_correct_usd), y(r.accuracy)
        slot = model_slot[r.model_key]
        tip = html.escape(
            f"{r.model_key} / {r.prompt}|Accuracy {r.accuracy * 100:.1f}% "
            f"({r.accuracy_min * 100:.0f}–{r.accuracy_max * 100:.0f}%)|"
            f"Cost per correct {_fmt_usd(r.cost_per_correct_usd)}|"
            f"Cost per item {_fmt_usd(r.cost_per_item_usd)}|p95 {r.latency_p95_ms or 0:,.0f} ms"
        )
        parts.append(f'<g class="mark s{slot}" data-tip="{tip}" tabindex="0">')
        parts.append(
            f'<line class="whisker" x1="{cx}" x2="{cx}" y1="{y(r.accuracy_min)}" y2="{y(r.accuracy_max)}"/>'
        )
        parts.append(_marker(prompt_shape[r.prompt], cx, cy))
        parts.append(f'<circle class="hit" cx="{cx}" cy="{cy}" r="16"/>')
        parts.append("</g>")
        ly = cy + 4
        while any(abs(ly - p) < 14 for p in placed):
            ly += 14
        placed.append(ly)
        parts.append(
            f'<text class="direct" x="{ML + pw + 12}" y="{ly}">{html.escape(r.model_key)} / '
            f"{html.escape(r.prompt)} · {r.accuracy * 100:.0f}%</text>"
        )
        parts.append(
            f'<line class="leader" x1="{cx + 9}" x2="{ML + pw + 8}" y1="{cy}" y2="{ly - 4}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _latency(summary: pd.DataFrame, model_slot: dict[str, int]) -> str:
    df = summary.dropna(subset=["latency_p50_ms", "latency_p95_ms"])
    if df.empty:
        return "<p>No live (uncached, successful) calls to measure latency from.</p>"
    rows = len(df)
    row_h, top = 34, 16
    h = top + rows * row_h + 44
    lw, rw = 220, 40
    pw = W - lw - rw
    hi = df.latency_p95_ms.max() * 1.1 or 1.0

    def x(v: float) -> float:
        return lw + v / hi * pw

    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Latency p50 to p95 per cell">']
    step = 10 ** math.floor(math.log10(hi)) if hi > 0 else 1
    if hi / step < 3:
        step /= 2
    t = 0.0
    while t <= hi:
        parts.append(
            f'<line class="grid" x1="{x(t)}" x2="{x(t)}" y1="{top}" y2="{top + rows * row_h}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x(t)}" y="{top + rows * row_h + 18}" text-anchor="middle">{t:,.0f}</text>'
        )
        t += step
    parts.append(
        f'<text class="label" x="{lw + pw / 2}" y="{h - 8}" text-anchor="middle">'
        "Latency, ms (dot = p50, bar end = p95)</text>"
    )
    for i, r in enumerate(df.sort_values("latency_p50_ms").itertuples()):
        cy = top + i * row_h + row_h / 2
        slot = model_slot[r.model_key]
        tip = html.escape(
            f"{r.model_key} / {r.prompt}|p50 {r.latency_p50_ms:,.0f} ms|p95 {r.latency_p95_ms:,.0f} ms"
            + (f"|TTFT p50 {r.ttft_p50_ms:,.0f} ms" if r.ttft_p50_ms else "")
        )
        parts.append(
            f'<text class="direct" x="{lw - 12}" y="{cy + 4}" text-anchor="end">'
            f"{html.escape(r.model_key)} / {html.escape(r.prompt)}</text>"
        )
        parts.append(f'<g class="mark s{slot}" data-tip="{tip}" tabindex="0">')
        parts.append(
            f'<line class="range" x1="{x(r.latency_p50_ms)}" x2="{x(r.latency_p95_ms)}" y1="{cy}" y2="{cy}"/>'
        )
        parts.append(f'<circle cx="{x(r.latency_p50_ms)}" cy="{cy}" r="5"/>')
        parts.append(
            f'<rect class="hit" x="{lw}" y="{cy - row_h / 2}" width="{pw}" height="{row_h}"/>'
        )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def _table(summary: pd.DataFrame) -> str:
    head = (
        "<tr><th>Model</th><th>Prompt</th><th>Accuracy</th><th>Schema-valid</th>"
        "<th>p50 ms</th><th>p95 ms</th><th>Cost / item</th><th>Cost / correct</th></tr>"
    )
    body = []
    for r in summary.itertuples():
        cpc = _fmt_usd(r.cost_per_correct_usd) if math.isfinite(r.cost_per_correct_usd) else "–"
        body.append(
            f"<tr><td>{html.escape(r.model_key)}</td><td>{html.escape(r.prompt)}</td>"
            f"<td>{r.accuracy * 100:.1f}%</td><td>{r.schema_valid * 100:.1f}%</td>"
            f"<td>{(r.latency_p50_ms or 0):,.0f}</td><td>{(r.latency_p95_ms or 0):,.0f}</td>"
            f"<td>{_fmt_usd(r.cost_per_item_usd)}</td><td>{cpc}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def _legend(model_slot: dict[str, int], prompt_shape: dict[str, str]) -> str:
    items = [
        f'<span class="key"><svg width="14" height="14" class="s{s}"><circle cx="7" cy="7" r="5"/></svg>'
        f"{html.escape(m)}</span>"
        for m, s in model_slot.items()
    ]
    items += [
        f'<span class="key"><svg width="14" height="14" class="shape">{_marker(sh, 7, 7, 4.5)}</svg>'
        f"{html.escape(p)}</span>"
        for p, sh in prompt_shape.items()
    ]
    return f'<div class="legend">{"".join(items)}</div>'


CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--grid:#e6e5e1;--axis:#a8a79f;%LIGHT%}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2e2e2c;--axis:#6b6a64;%DARK%}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--grid:#2e2e2c;--axis:#6b6a64;%DARK%}
body{margin:0;background:var(--surface-1)}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,
"Segoe UI",sans-serif;max-width:760px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 4px}
p.sub{color:var(--text-secondary);margin:0 0 12px}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--text-secondary);font-size:11px}.label{fill:var(--text-secondary);font-size:12px}
.direct{fill:var(--text-primary);font-size:12px}
.leader{stroke:var(--grid);stroke-width:1}
.mark circle:not(.hit),.mark rect:not(.hit),.mark path{stroke:var(--surface-1);stroke-width:2}
.mark .whisker,.mark .range{stroke-width:2;stroke-linecap:round}
.hit{fill:transparent;stroke:none}.legend svg{width:14px;height:14px;flex:none}.mark{cursor:default;outline:none}
.mark:hover circle:not(.hit),.mark:focus circle:not(.hit){r:7}
.s1 :not(.hit){fill:var(--series-1)}.s1 line{stroke:var(--series-1)}
.s2 :not(.hit){fill:var(--series-2)}.s2 line{stroke:var(--series-2)}
.s3 :not(.hit){fill:var(--series-3)}.s3 line{stroke:var(--series-3)}
.s0 :not(.hit){fill:var(--axis)}.s0 line{stroke:var(--axis)}
.shape *{fill:none;stroke:var(--text-secondary);stroke-width:1.5}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:4px 0 8px;color:var(--text-secondary);font-size:12px}
.key{display:inline-flex;align-items:center;gap:6px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);
border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;display:none;
box-shadow:0 2px 8px rgba(0,0,0,.12);max-width:280px}
#tip b{display:block;margin-bottom:2px}
details{margin-top:24px}summary{cursor:pointer;color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}
.warn{color:var(--text-secondary);border-left:3px solid var(--axis);padding:4px 10px;margin:8px 0}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,el){const p=el.dataset.tip.split('|');tip.innerHTML='<b>'+p[0]+'</b>'+p.slice(1).join('<br>');
tip.style.display='block';const r=el.getBoundingClientRect();
const x=(e&&e.clientX)||r.right,y=(e&&e.clientY)||r.top;
tip.style.left=Math.min(x+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.max(8,y-tip.offsetHeight-8)+'px';}
document.querySelectorAll('.mark').forEach(el=>{el.addEventListener('mousemove',e=>show(e,el));
el.addEventListener('focus',()=>show(null,el));el.addEventListener('mouseleave',()=>tip.style.display='none');
el.addEventListener('blur',()=>tip.style.display='none');});
"""


def build_html(run: RunData) -> str:
    summary = cell_summary(run)
    models = list(dict.fromkeys(summary.model_key)) if not summary.empty else []
    prompts = list(dict.fromkeys(summary.prompt)) if not summary.empty else []
    model_slot = {m: (i + 1 if i < 3 else 0) for i, m in enumerate(models)}
    prompt_shape = {p: SHAPES[i % len(SHAPES)] for i, p in enumerate(prompts)}
    light = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_DARK))
    css = CSS.replace("%LIGHT%", light).replace("%DARK%", dark)
    simulated = "fake" in set(run.meta.get("providers", {}).values())
    warn = (
        '<p class="warn">Simulated run (fake provider): these numbers test the harness only.</p>'
        if simulated
        else ""
    )
    extra = (
        '<p class="warn">More than 3 models: models past the third are drawn in gray.</p>'
        if len(models) > 3
        else ""
    )
    body = "<p>No results.</p>"
    if not summary.empty:
        body = (
            "<h2>Accuracy vs cost per correct answer</h2>"
            '<p class="sub">Up and to the left is better. Whisker = accuracy range across repeats.</p>'
            + _legend(model_slot, prompt_shape)
            + _scatter(summary, model_slot, prompt_shape)
            + "<h2>Latency</h2>"
            + _latency(summary, model_slot)
            + "<details><summary>Table view</summary>"
            + _table(summary)
            + "</details>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Model comparison · {html.escape(run.meta['experiment'])}</title>"
        f"<style>{css}</style></head><body><div class='viz-root'>"
        f"<h1>{html.escape(run.meta['experiment'])}</h1>"
        f"<p class='sub'>{run.meta['n_items']} items · run {html.escape(run.meta['run_id'])} · "
        f"{html.escape(run.meta['started_at'])} UTC</p>{warn}{extra}{body}"
        f"<div id='tip'></div></div><script>{JS}</script></body></html>"
    )


def write_html(run: RunData, path: Path) -> Path:
    path.write_text(build_html(run), encoding="utf-8")
    return path
