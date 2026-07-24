from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import logging
import os
import time
import requests
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into a date-indexed close-price DataFrame."""
    if raw is None or raw.empty:
        raise ValueError("The market-data response was empty.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Adj Close" in level0:
            close = raw["Adj Close"].copy()
        else:
            raise ValueError(f"Could not locate Close data. Fields: {sorted(set(level0))}")
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            close = raw[["Adj Close"]].copy()
        else:
            raise ValueError(f"Could not locate Close data. Columns: {list(raw.columns)}")
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    if len(tickers) == 1 and list(close.columns) == ["Close"]:
        close.columns = tickers

    close.columns = [str(c).upper() for c in close.columns]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    return close

def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    retries: int = 3,
    pause_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download auto-adjusted close prices with local CSV caching and retries."""
    ticker_list = sorted({str(t).strip().upper() for t in tickers})
    if not ticker_list:
        raise ValueError("No tickers supplied.")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = "_".join(t.replace("^", "IDX") for t in ticker_list)
    file_path = cache_path / f"prices_{key}_{start}_{end}.csv"

    if file_path.exists() and not refresh:
        LOGGER.info("Loading cached prices from %s", file_path)
        cached = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if not cached.empty:
            return cached

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info("Downloading %s (attempt %s/%s)", ticker_list, attempt, retries)
            raw = yf.download(
                ticker_list,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                repair=True,
                progress=False,
                threads=True,
                group_by="column",
                multi_level_index=True,
            )
            close = _extract_close(raw, ticker_list)
            missing = [t for t in ticker_list if t not in close.columns]
            if missing:
                raise ValueError(f"No usable price series returned for: {missing}")
            close = close[ticker_list]
            close.to_csv(file_path)
            return close
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Download attempt failed: %s", exc)
            if attempt < retries:
                time.sleep(pause_seconds * attempt)

    raise RuntimeError(f"Price download failed after {retries} attempts: {last_error}")

def download_fred_series(
    series_id: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    """Download a FRED series using the official observations endpoint."""
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED_API_KEY is not configured. Use a fixed rate or add the key to Colab secrets."
        )
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for FRED series {series_id}.")
    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.set_index("date")["value"].dropna().rename(series_id)
