from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Tuple
import math

@dataclass(frozen=True)
class PortfolioConfig:
    """Validated configuration for the portfolio analytics pipeline."""
    weights: Dict[str, float]
    benchmark: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    initial_capital: float = 100_000.0
    annual_risk_free_rate: float = 0.04
    confidence_level: float = 0.95
    rolling_short: int = 63
    rolling_long: int = 126
    rebalancing_frequency: str = "M"
    transaction_cost_bps: float = 5.0

    def __post_init__(self) -> None:
        cleaned = {str(k).strip().upper(): float(v) for k, v in self.weights.items()}
        object.__setattr__(self, "weights", cleaned)
        object.__setattr__(self, "benchmark", self.benchmark.strip().upper())

        if not cleaned:
            raise ValueError("At least one portfolio asset is required.")
        if any(not math.isfinite(v) for v in cleaned.values()):
            raise ValueError("All weights must be finite numbers.")
        if any(v < 0 for v in cleaned.values()):
            raise ValueError("Long-only base version does not permit negative weights.")
        if not math.isclose(sum(cleaned.values()), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Weights must sum to 1.0; received {sum(cleaned.values()):.8f}.")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0.5 and 1.")
        if self.rolling_short < 2 or self.rolling_long < 2:
            raise ValueError("Rolling windows must be at least 2 observations.")
        if self.transaction_cost_bps < 0:
            raise ValueError("Transaction costs cannot be negative.")

    @property
    def tickers(self) -> Tuple[str, ...]:
        return tuple(self.weights.keys())
