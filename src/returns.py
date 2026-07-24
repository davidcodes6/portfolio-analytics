from __future__ import annotations
import numpy as np
import pandas as pd

def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change(fill_method=None).dropna(how="all")

def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices / prices.shift(1)).dropna(how="all")

def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(asset_returns.columns)
    if aligned_weights.isna().any():
        raise ValueError("Weights do not cover all return columns.")
    return asset_returns.dot(aligned_weights).rename("Portfolio")

def wealth_index(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    return (initial_capital * (1.0 + returns.fillna(0.0)).cumprod()).rename("Wealth")

def active_returns(portfolio: pd.Series, benchmark: pd.Series) -> pd.Series:
    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1]).rename("Active")
