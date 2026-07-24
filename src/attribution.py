from __future__ import annotations
import numpy as np
import pandas as pd

def return_contributions(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    return asset_returns.mul(weights.reindex(asset_returns.columns), axis=1)

def risk_contributions(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    periods: int = 252,
) -> pd.DataFrame:
    w = weights.reindex(asset_returns.columns).astype(float)
    cov = asset_returns.cov() * periods
    port_var = float(w.T @ cov @ w)
    port_vol = np.sqrt(port_var)
    if port_vol <= 0:
        raise ValueError("Portfolio volatility must be positive.")
    marginal = (cov @ w) / port_vol
    component = w * marginal
    pct = component / port_vol
    out = pd.DataFrame({
        "Weight": w,
        "Asset Volatility": np.sqrt(np.diag(cov)),
        "Marginal Risk Contribution": marginal,
        "Component Risk Contribution": component,
        "Percent Risk Contribution": pct,
    })
    return out

def diversification_ratio(asset_returns: pd.DataFrame, weights: pd.Series, periods: int = 252) -> float:
    rc = risk_contributions(asset_returns, weights, periods)
    weighted_asset_vol = float((rc["Weight"] * rc["Asset Volatility"]).sum())
    portfolio_vol = float(rc["Component Risk Contribution"].sum())
    return weighted_asset_vol / portfolio_vol
