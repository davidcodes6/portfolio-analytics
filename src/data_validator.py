from __future__ import annotations
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

def validate_and_align_prices(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    max_missing_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Validate, clean and align portfolio and benchmark prices."""
    if prices.empty or benchmark.empty:
        raise ValueError("Prices and benchmark must be non-empty.")
    prices = prices.copy()
    benchmark = benchmark.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    benchmark = benchmark[~benchmark.index.duplicated(keep="last")].sort_index()

    if (prices <= 0).any().any() or (benchmark <= 0).any():
        raise ValueError("Price data contain non-positive values.")

    quality = pd.DataFrame({
        "missing_fraction": prices.isna().mean(),
        "first_valid": prices.apply(lambda s: s.first_valid_index()),
        "last_valid": prices.apply(lambda s: s.last_valid_index()),
        "observations": prices.notna().sum(),
    })
    bad = quality.index[quality["missing_fraction"] > max_missing_fraction].tolist()
    if bad:
        LOGGER.warning("High missing-data fraction for %s", bad)

    # Controlled forward fill only; leading missing values remain missing.
    prices = prices.ffill()
    joined = prices.join(benchmark.rename("Benchmark"), how="inner").dropna()
    if len(joined) < 60:
        raise ValueError("Fewer than 60 aligned observations remain after cleaning.")

    asset_prices = joined.drop(columns="Benchmark")
    benchmark_prices = joined["Benchmark"]
    returns = asset_prices.pct_change()
    extreme = (returns.abs() > 0.50).sum()
    for ticker, count in extreme[extreme > 0].items():
        LOGGER.warning("%s has %d absolute daily returns above 50%%.", ticker, count)

    return asset_prices, benchmark_prices, quality
