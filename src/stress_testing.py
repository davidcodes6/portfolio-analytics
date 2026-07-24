from __future__ import annotations
import pandas as pd
from .performance import max_drawdown

DEFAULT_SCENARIOS = {
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "US Debt Ceiling Crisis": ("2011-07-22", "2011-10-03"),
    "Q4 2018 Selloff": ("2018-09-20", "2018-12-24"),
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock": ("2022-01-03", "2022-10-12"),
}

def historical_stress_tests(
    portfolio: pd.Series,
    benchmark: pd.Series,
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    scenarios = scenarios or DEFAULT_SCENARIOS
    records = []
    for name, (start, end) in scenarios.items():
        p = portfolio.loc[start:end].dropna()
        b = benchmark.loc[start:end].dropna()
        if len(p) < 2 or len(b) < 2:
            continue
        records.append({
            "Scenario": name,
            "Start": start,
            "End": end,
            "Portfolio Return": (1 + p).prod() - 1,
            "Benchmark Return": (1 + b).prod() - 1,
            "Portfolio Max Drawdown": max_drawdown(p),
            "Worst Portfolio Day": p.min(),
            "Observations": len(p),
        })
    return pd.DataFrame(records).set_index("Scenario") if records else pd.DataFrame()

def hypothetical_shock(weights: pd.Series, shocks: dict[str, float]) -> pd.DataFrame:
    shock_series = pd.Series(shocks, dtype=float).reindex(weights.index).fillna(0.0)
    contributions = weights * shock_series
    table = pd.DataFrame({
        "Weight": weights,
        "Shock": shock_series,
        "Contribution": contributions,
    })
    table.loc["TOTAL"] = [weights.sum(), float("nan"), contributions.sum()]
    return table
