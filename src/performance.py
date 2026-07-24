from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

def total_return(r: pd.Series) -> float:
    return float((1 + r.dropna()).prod() - 1)

def annualized_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    return float((1 + r).prod() ** (periods / len(r)) - 1)

def annualized_arithmetic_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.dropna().mean() * periods)

def annualized_volatility(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(periods))

def downside_deviation(r: pd.Series, target_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    target_period = (1 + target_annual) ** (1 / periods) - 1
    downside = np.minimum(r.dropna() - target_period, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods))

def sharpe_ratio(r: pd.Series, annual_rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    rf_period = (1 + annual_rf) ** (1 / periods) - 1
    excess = r.dropna() - rf_period
    vol = r.dropna().std(ddof=1) * np.sqrt(periods)
    return float(excess.mean() * periods / vol) if vol > 0 else np.nan

def sortino_ratio(r: pd.Series, annual_rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ann_excess = annualized_arithmetic_return(r, periods) - annual_rf
    dd = downside_deviation(r, annual_rf, periods)
    return float(ann_excess / dd) if dd > 0 else np.nan

def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return (wealth / wealth.cummax() - 1).rename("Drawdown")

def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())

def calmar_ratio(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return float(annualized_return(r) / mdd) if mdd > 0 else np.nan

def ulcer_index(r: pd.Series) -> float:
    dd = 100 * drawdown_series(r)
    return float(np.sqrt(np.mean(dd ** 2)))

def longest_drawdown_days(r: pd.Series) -> int:
    dd = drawdown_series(r)
    underwater = dd < 0
    groups = (~underwater).cumsum()
    lengths = underwater.groupby(groups).sum()
    return int(lengths.max()) if not lengths.empty else 0

def tail_ratio(r: pd.Series) -> float:
    x = r.dropna()
    q95, q05 = np.quantile(x, [0.95, 0.05])
    return float(q95 / abs(q05)) if q05 != 0 else np.nan

def summary(r: pd.Series, annual_rf: float = 0.0) -> pd.Series:
    x = r.dropna()
    monthly = (1 + x).resample("ME").prod() - 1
    return pd.Series({
        "Total Return": total_return(x),
        "Annualized Return": annualized_return(x),
        "Annualized Arithmetic Return": annualized_arithmetic_return(x),
        "Annualized Volatility": annualized_volatility(x),
        "Sharpe Ratio": sharpe_ratio(x, annual_rf),
        "Sortino Ratio": sortino_ratio(x, annual_rf),
        "Calmar Ratio": calmar_ratio(x),
        "Maximum Drawdown": max_drawdown(x),
        "Longest Drawdown (days)": longest_drawdown_days(x),
        "Downside Deviation": downside_deviation(x, annual_rf),
        "Ulcer Index": ulcer_index(x),
        "Tail Ratio": tail_ratio(x),
        "Skewness": float(stats.skew(x, bias=False)),
        "Excess Kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
        "Positive Day %": float((x > 0).mean()),
        "Best Day": float(x.max()),
        "Worst Day": float(x.min()),
        "Best Month": float(monthly.max()) if not monthly.empty else np.nan,
        "Worst Month": float(monthly.min()) if not monthly.empty else np.nan,
    })
