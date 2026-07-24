from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _aligned(portfolio: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    out = pd.concat([portfolio.rename("Portfolio"), benchmark.rename("Benchmark")], axis=1).dropna()
    if len(out) < 30:
        raise ValueError("At least 30 aligned observations are required.")
    return out

def capm_regression(portfolio: pd.Series, benchmark: pd.Series, annual_rf: float = 0.0, periods: int = 252):
    frame = _aligned(portfolio, benchmark)
    rf = (1 + annual_rf) ** (1 / periods) - 1
    y = frame["Portfolio"] - rf
    X = sm.add_constant(frame["Benchmark"] - rf)
    return sm.OLS(y, X, missing="drop").fit()

def beta(portfolio: pd.Series, benchmark: pd.Series) -> float:
    frame = _aligned(portfolio, benchmark)
    return float(frame["Portfolio"].cov(frame["Benchmark"]) / frame["Benchmark"].var())

def tracking_error(portfolio: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    frame = _aligned(portfolio, benchmark)
    return float((frame["Portfolio"] - frame["Benchmark"]).std(ddof=1) * np.sqrt(periods))

def information_ratio(portfolio: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    frame = _aligned(portfolio, benchmark)
    active = frame["Portfolio"] - frame["Benchmark"]
    te = active.std(ddof=1) * np.sqrt(periods)
    return float(active.mean() * periods / te) if te > 0 else np.nan

def capture_ratio(portfolio: pd.Series, benchmark: pd.Series, direction: str) -> float:
    frame = _aligned(portfolio, benchmark)
    mask = frame["Benchmark"] >= 0 if direction == "up" else frame["Benchmark"] < 0
    sample = frame.loc[mask]
    denom = sample["Benchmark"].mean()
    return float(sample["Portfolio"].mean() / denom) if len(sample) and denom != 0 else np.nan

def benchmark_summary(portfolio: pd.Series, benchmark: pd.Series, annual_rf: float = 0.0) -> pd.Series:
    frame = _aligned(portfolio, benchmark)
    model = capm_regression(portfolio, benchmark, annual_rf)
    b_name = [n for n in model.params.index if n != "const"][0]
    return pd.Series({
        "Beta": float(model.params[b_name]),
        "Annualized Jensen Alpha": float(model.params["const"] * 252),
        "Alpha p-value": float(model.pvalues["const"]),
        "Beta p-value": float(model.pvalues[b_name]),
        "CAPM R-squared": float(model.rsquared),
        "Correlation": float(frame.corr().iloc[0, 1]),
        "Tracking Error": tracking_error(portfolio, benchmark),
        "Information Ratio": information_ratio(portfolio, benchmark),
        "Upside Capture": capture_ratio(portfolio, benchmark, "up"),
        "Downside Capture": capture_ratio(portfolio, benchmark, "down"),
    })
