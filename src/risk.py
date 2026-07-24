from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def historical_var(r: pd.Series, confidence: float = 0.95) -> float:
    return float(-np.quantile(r.dropna(), 1 - confidence))

def historical_cvar(r: pd.Series, confidence: float = 0.95) -> float:
    x = r.dropna()
    threshold = np.quantile(x, 1 - confidence)
    tail = x[x <= threshold]
    return float(-tail.mean()) if len(tail) else np.nan

def normal_var(r: pd.Series, confidence: float = 0.95) -> float:
    x = r.dropna()
    q = stats.norm.ppf(1 - confidence, loc=x.mean(), scale=x.std(ddof=1))
    return float(-q)

def normal_cvar(r: pd.Series, confidence: float = 0.95) -> float:
    x = r.dropna()
    mu, sigma = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu - sigma * stats.norm.pdf(z) / (1 - confidence)))

def student_t_var(r: pd.Series, confidence: float = 0.95) -> float:
    x = r.dropna()
    df, loc, scale = stats.t.fit(x)
    return float(-stats.t.ppf(1 - confidence, df, loc=loc, scale=scale))

def monte_carlo_var(
    r: pd.Series, confidence: float = 0.95, simulations: int = 100_000, seed: int = 42
) -> float:
    x = r.dropna()
    rng = np.random.default_rng(seed)
    draws = rng.normal(x.mean(), x.std(ddof=1), simulations)
    return float(-np.quantile(draws, 1 - confidence))

def semivariance(r: pd.Series, target: float = 0.0) -> float:
    downside = np.minimum(r.dropna() - target, 0)
    return float(np.mean(downside ** 2))

def risk_summary(r: pd.Series, confidence: float = 0.95) -> pd.Series:
    return pd.Series({
        f"Historical VaR {confidence:.0%}": historical_var(r, confidence),
        f"Historical CVaR {confidence:.0%}": historical_cvar(r, confidence),
        f"Normal VaR {confidence:.0%}": normal_var(r, confidence),
        f"Normal CVaR {confidence:.0%}": normal_cvar(r, confidence),
        f"Student-t VaR {confidence:.0%}": student_t_var(r, confidence),
        f"Monte Carlo VaR {confidence:.0%}": monte_carlo_var(r, confidence),
        "Daily Semivariance": semivariance(r),
    })
