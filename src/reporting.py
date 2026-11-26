from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


CHART_NOTES = {
    "Growth of Capital": "Compare how the same starting capital grew in the portfolio and benchmark.",
    "Drawdowns": "Measure each series’ decline from its previous peak.",
    "Rolling Risk": "Track changes in volatility and the estimated downside threshold over time.",
    "Rolling Benchmark Exposure": "Compare the strength and direction of the portfolio’s relationship to its benchmark.",
    "Risk Contribution": "Compare the share of capital allocated to each asset with its contribution to portfolio volatility.",
    "Stress Tests": "Compare portfolio and benchmark returns in selected historical periods.",
}

REPORT_CSS = """
:root {
  color-scheme: light;
  --ink: #14253c;
  --muted: #46576b;
  --rule: #dce5ed;
  --paper: #f2f6f9;
  font-family: Inter, system-ui, sans-serif;
}
* { box-sizing: border-box }
body { margin: 0; color: var(--ink); background: var(--paper); line-height: 1.55 }
.wrap { width: min(1120px, calc(100% - 48px)); margin-inline: auto }
header { background: #102942; color: white; border-bottom: 4px solid #31a99b; padding: 54px 0 46px }
h1 { margin: 0 0 12px; font-size: clamp(2rem, 6vw, 3.75rem); line-height: 1.05; letter-spacing: -.045em }
header p { color: #d5e7f2; margin: 0; max-width: 62ch }
nav { background: white; border-bottom: 1px solid var(--rule) }
nav ul { display: flex; flex-wrap: wrap; gap: 8px 22px; padding: 14px 0; list-style: none; margin-block: 0 }
a { color: #11685f; text-underline-offset: .2em }
nav a { display: inline-block; font-size: .875rem; font-weight: 650 }
a:focus-visible { outline: 2px solid #11685f; outline-offset: 3px }
main { padding-block: 28px 36px; display: grid; gap: 20px }
section { min-width: 0; background: white; border: 1px solid var(--rule); border-radius: 14px; padding: 20px 24px 10px; scroll-margin: 14px }
h2 { font-size: 1.3rem; margin: 0; letter-spacing: -.025em }
section p, footer { color: var(--muted); font-size: .9rem }
section p { margin: 4px 0 0 }
footer { border-top: 1px solid var(--rule); padding-block: 16px 30px }
noscript { display: block; padding-block: 16px }
@media (max-width: 600px) {
  .wrap { width: calc(100% - 24px) }
  header { padding: 36px 0 30px }
  section { padding-inline: 10px }
}
"""


def export_tables(tables: dict[str, pd.DataFrame | pd.Series], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = output / f"{name}.csv"
        if isinstance(table, pd.Series):
            table.to_frame("Value").to_csv(path)
        else:
            table.to_csv(path)
        paths.append(path)
    return paths


def export_figures_html(
    figures: dict[str, go.Figure],
    output_file: str | Path,
    *,
    description: str = "Explore portfolio performance, benchmark exposure, and downside risk.",
) -> Path:
    """Export an interactive report that can be opened offline in a browser."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    links = []
    sections = []
    for index, (title, figure) in enumerate(figures.items()):
        anchor = f"chart-{index + 1}"
        safe_title = escape(title)
        links.append(f'<li><a href="#{anchor}">{safe_title}</a></li>')
        note = CHART_NOTES.get(title)
        description_html = f"<p>{escape(note)}</p>" if note else ""
        chart = go.Figure(figure)
        chart.update_layout(
            title_text=None, autosize=True, width=None, height=350,
            paper_bgcolor="#ffffff", plot_bgcolor="#f6f9fc",
            font=dict(family="system-ui, sans-serif", color="#243a52", size=12),
            margin=dict(l=58, r=16, t=18, b=75),
            legend=dict(orientation="h", yanchor="top", y=-0.20, x=0, title_text=""),
        )
        chart.update_xaxes(gridcolor="#dce5ed", automargin=True)
        chart.update_yaxes(gridcolor="#dce5ed", automargin=True)
        chart_html = chart.to_html(
            full_html=False,
            include_plotlyjs=index == 0,
            config={"responsive": True, "displaylogo": False},
            div_id=f"{anchor}-plot",
        )
        sections.append(
            f'<section id="{anchor}" aria-labelledby="{anchor}-title">'
            f'<h2 id="{anchor}-title">{safe_title}</h2>{description_html}{chart_html}</section>'
        )
    navigation = (
        f'<nav aria-label="Report sections"><ul class="wrap">{"".join(links)}</ul></nav>'
        if links else ""
    )
    content = "".join(sections) or "<p>No charts were generated.</p>"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Analytics Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<header><div class="wrap"><h1>Portfolio Analytics</h1><p>{escape(description)}</p></div></header>
{navigation}
<noscript><p class="wrap">Enable JavaScript to view the interactive charts.</p></noscript>
<main class="wrap">{content}</main>
<footer><div class="wrap">Historical analysis for educational use.</div></footer>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output
