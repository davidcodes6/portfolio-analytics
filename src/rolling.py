from __future__ import annotations
import numpy as np
import pandas as pd
from .performance import drawdown_series

def rolling_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series,
    annual_rf: float,
    short_window: int = 63,
    long_window: int = 126,
    periods: int = 252,
) -> pd.DataFrame:
    frame = pd.concat([portfolio.rename("Portfolio"), benchmark.rename("Benchmark")], axis=1).dropna()
    rf_period = (1 + annual_rf) ** (1 / periods) - 1
    out = pd.DataFrame(index=frame.index)
    out["Rolling Return"] = (
        (1 + frame["Portfolio"]).rolling(short_window).apply(np.prod, raw=True)
        ** (periods / short_window) - 1
    )
    out["Rolling Volatility"] = frame["Portfolio"].rolling(short_window).std() * np.sqrt(periods)
    out["Rolling Sharpe"] = (
        (frame["Portfolio"] - rf_period).rolling(short_window).mean() * periods
        / out["Rolling Volatility"]
    )
    downside = np.minimum(frame["Portfolio"] - rf_period, 0)
    downside_dev = pd.Series(downside, index=frame.index).rolling(short_window).apply(
        lambda x: np.sqrt(np.mean(np.square(x))) * np.sqrt(periods), raw=True
    )
    out["Rolling Sortino"] = (
        (frame["Portfolio"] - rf_period).rolling(short_window).mean() * periods / downside_dev
    )
    cov = frame["Portfolio"].rolling(long_window).cov(frame["Benchmark"])
    var = frame["Benchmark"].rolling(long_window).var()
    out["Rolling Beta"] = cov / var
    out["Rolling Correlation"] = frame["Portfolio"].rolling(long_window).corr(frame["Benchmark"])
    out["Rolling VaR 95%"] = -frame["Portfolio"].rolling(short_window).quantile(0.05)
    out["Drawdown"] = drawdown_series(frame["Portfolio"])
    return out
