from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

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

def export_figures_html(figures: dict[str, go.Figure], output_file: str | Path) -> Path:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fragments = []
    for i, (title, fig) in enumerate(figures.items()):
        fragments.append(f"<h2>{title}</h2>")
        fragments.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html = """<!doctype html><html><head><meta charset="utf-8">
    <title>Portfolio Analytics Report</title></head><body>
    <h1>Portfolio Analytics Report</h1>""" + "\n".join(fragments) + "</body></html>"
    output.write_text(html, encoding="utf-8")
    return output
