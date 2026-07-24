from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_rebalancing(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
    frequency: str | None = "ME",
    transaction_cost_bps: float = 0.0,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate buy-and-hold or calendar rebalancing with linear turnover costs."""
    returns = asset_returns.dropna().copy()
    target = target_weights.reindex(returns.columns).astype(float)
    values = initial_capital * target
    portfolio_values = []
    history = []
    rebalance_dates = set()
    if frequency:
        rebalance_dates = set(returns.resample(frequency).last().index)

    for dt, row in returns.iterrows():
        values = values * (1 + row)
        gross_value = float(values.sum())
        turnover = 0.0
        cost = 0.0

        should_rebalance = frequency is not None and any(
            (dt.year, dt.month) == (x.year, x.month) for x in rebalance_dates
        )
        # Rebalance on the final available observation in each selected period.
        if should_rebalance:
            period_end = returns.loc[:dt].resample(frequency).last().index.max()
            if (dt.year, dt.month, dt.day) == (period_end.year, period_end.month, period_end.day):
                current_weights = values / gross_value
                turnover = float(np.abs(target - current_weights).sum())
                cost = gross_value * turnover * transaction_cost_bps / 10_000
                gross_value -= cost
                values = gross_value * target

        portfolio_values.append(gross_value)
        history.append({"Date": dt, "Turnover": turnover, "Cost": cost})

    wealth = pd.Series(portfolio_values, index=returns.index, name="Portfolio Value")
    daily_returns = wealth.pct_change().dropna().rename("Portfolio")
    return daily_returns, pd.DataFrame(history).set_index("Date")
