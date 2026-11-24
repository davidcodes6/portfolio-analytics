# Multi-Asset Portfolio Analytics

A Python research project for comparing a weighted portfolio with a benchmark. It measures performance, downside risk, and asset-level risk contributions, and compares optimization and rebalancing strategies.

[Open the example report](reports/portfolio_analytics_report.html) · [View the analysis notebook](notebooks/portfolio_analytics_colab.ipynb)

The example report is a historical snapshot. Download the HTML file and open it in a browser to zoom, inspect values, or toggle series. The charts work offline.

## Run the analysis

From the extracted repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt notebook
python -m notebook notebooks/portfolio_analytics_colab.ipynb
```

Run the notebook from top to bottom. Edit the `PortfolioConfig` cell to change assets, weights, benchmark, or date range. The notebook downloads price data, caches it locally, and writes an interactive HTML report and CSV tables under `reports/`.

For Google Colab, upload and extract the full repository, open the notebook, and set its working directory to the extracted repository root. Run the installation cell before the imports.

## What the analysis covers

| Area | Outputs |
| --- | --- |
| Performance | Cumulative growth, drawdowns, Sharpe ratio, CAPM alpha and beta, and rolling benchmark exposure |
| Risk | VaR and CVaR estimates, asset-level risk contribution, and historical and hypothetical stress tests |
| Optimization | Minimum-variance and maximum-Sharpe portfolios and an efficient frontier |
| Rebalancing | Buy-and-hold and calendar rebalancing with transaction costs |

## Project structure

- `src/` contains reusable data, analytics, optimization, and reporting modules.
- `notebooks/` contains the end-to-end workflow and editable example configuration.
- `config/portfolio_config.json` documents an example configuration.
- `reports/` contains the sample report. Notebook runs also write CSV exports.
- `tests/` contains known-answer checks for analytics and optimization.

Run the tests with `python -m pytest -q`.

The analysis uses adjusted Yahoo Finance prices. Historical estimates depend on the selected assets, period, and assumptions. The workflow does not account for survivorship bias or delisting returns. See the notebook for additional modeling assumptions.

MIT License. See [LICENSE](LICENSE).
