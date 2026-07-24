# Institutional Multi-Asset Portfolio Analytics Platform

A modular Google Colab and Python project for analyzing multi-asset portfolios across equities, ETFs, bonds, commodities and digital assets.

## Features

- Auto-adjusted market-data download with retry logic and local caching
- Data validation and calendar alignment
- Simple/log returns and wealth indices
- Sharpe, Sortino, Calmar, drawdown, skewness, kurtosis and tail metrics
- CAPM alpha/beta, tracking error, information ratio and capture ratios
- Historical, normal, Student-t and Monte Carlo VaR
- Historical and normal CVaR
- Return and risk attribution
- Rolling risk/performance analytics
- Historical and hypothetical stress testing
- Buy-and-hold and calendar-rebalancing simulations with transaction costs
- Interactive Plotly charts and HTML export
- Unit tests with known-answer checks

## Quick start in Google Colab

1. Upload the project ZIP and extract it, or clone the repository.
2. Open `notebooks/portfolio_analytics_colab.ipynb`.
3. Run the installation cell.
4. Edit the `PortfolioConfig` weights and dates.
5. Run all cells from top to bottom.

## Example configuration

```python
PortfolioConfig(
    weights={
        "AAPL": 0.20,
        "MSFT": 0.20,
        "SPY": 0.25,
        "TLT": 0.15,
        "GLD": 0.10,
        "BTC-USD": 0.10,
    },
    benchmark="SPY",
    start_date="2018-01-01",
    end_date="2026-07-01",
    initial_capital=100_000,
)
```

## Data limitations

Yahoo Finance is convenient for education and prototyping but is not an exchange-grade or point-in-time institutional feed. The project does not remove survivorship bias, model delisting returns, or guarantee corporate-action completeness. FRED observations may be revised. Results are research outputs, not investment advice.

## Resume bullet

Built a modular multi-asset portfolio analytics platform in Python that computes benchmark-relative performance, CAPM attribution, rolling risk, VaR/CVaR, historical stress tests, rebalancing costs and asset-level risk contributions, with automated data validation and interactive reporting.

## Institutional upgrades

Replace free market data with point-in-time licensed data; add holdings and trade databases, factor-risk models, shrinkage covariance, constrained optimization, live position feeds, liquidity and market-impact models, CI/CD, Docker, authentication, monitoring and scheduled risk-limit alerts.
