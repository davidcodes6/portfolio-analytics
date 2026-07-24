import numpy as np
import pandas as pd

from src.optimization import (
    estimate_covariance_matrix,
    estimate_expected_returns,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    efficient_frontier,
)


def sample_returns() -> pd.DataFrame:
    np.random.seed(42)

    return pd.DataFrame(
        {
            "A": np.random.normal(0.0005, 0.01, 500),
            "B": np.random.normal(0.0003, 0.006, 500),
            "C": np.random.normal(0.0007, 0.015, 500),
        }
    )


def test_minimum_variance_weights_sum_to_one():
    returns = sample_returns()

    expected = estimate_expected_returns(returns)
    covariance = estimate_covariance_matrix(returns)

    result = minimum_variance_portfolio(
        expected,
        covariance,
        max_weight=1.0,
    )

    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= -1e-8).all()


def test_maximum_sharpe_weights_sum_to_one():
    returns = sample_returns()

    expected = estimate_expected_returns(returns)
    covariance = estimate_covariance_matrix(returns)

    result = maximum_sharpe_portfolio(
        expected,
        covariance,
        risk_free_rate=0.02,
        max_weight=1.0,
    )

    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= -1e-8).all()


def test_efficient_frontier_returns_results():
    returns = sample_returns()

    expected = estimate_expected_returns(returns)
    covariance = estimate_covariance_matrix(returns)

    frontier = efficient_frontier(
        expected,
        covariance,
        points=10,
    )

    assert not frontier.empty
    assert "Expected Return" in frontier.columns
    assert "Volatility" in frontier.columns