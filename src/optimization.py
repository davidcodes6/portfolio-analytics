"""Portfolio optimization tools based on Modern Portfolio Theory."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class OptimizedPortfolio:
    """Results returned by a portfolio optimization model."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float


def estimate_expected_returns(
    asset_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Estimate annualized expected returns using historical arithmetic means.

    Parameters
    ----------
    asset_returns:
        DataFrame containing periodic asset returns.
    periods_per_year:
        Number of return observations per year.

    Returns
    -------
    pd.Series
        Annualized expected return for each asset.
    """
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty.")

    return asset_returns.mean() * periods_per_year


def estimate_covariance_matrix(
    asset_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Estimate an annualized covariance matrix.

    Parameters
    ----------
    asset_returns:
        DataFrame containing periodic asset returns.
    periods_per_year:
        Number of return observations per year.

    Returns
    -------
    pd.DataFrame
        Annualized asset covariance matrix.
    """
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty.")

    return asset_returns.cov() * periods_per_year


def calculate_portfolio_statistics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """
    Calculate expected return, volatility, and Sharpe ratio.

    Returns
    -------
    tuple[float, float, float]
        Expected annual return, annual volatility, and Sharpe ratio.
    """
    aligned_weights = weights.reindex(expected_returns.index)

    if aligned_weights.isna().any():
        raise ValueError("Weights do not align with expected-return assets.")

    weight_array = aligned_weights.to_numpy(dtype=float)
    mean_array = expected_returns.to_numpy(dtype=float)
    covariance_array = covariance_matrix.loc[
        expected_returns.index,
        expected_returns.index,
    ].to_numpy(dtype=float)

    expected_return = float(weight_array @ mean_array)

    portfolio_variance = float(
        weight_array.T @ covariance_array @ weight_array
    )
    volatility = float(np.sqrt(max(portfolio_variance, 0.0)))

    sharpe_ratio = (
        (expected_return - risk_free_rate) / volatility
        if volatility > 0
        else np.nan
    )

    return expected_return, volatility, float(sharpe_ratio)


def minimum_variance_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> OptimizedPortfolio:
    """
    Find the long-only portfolio with minimum variance.

    Constraints
    -----------
    - Portfolio weights sum to one.
    - No short selling.
    - Each asset weight is limited by max_weight.
    """
    assets = expected_returns.index
    covariance = covariance_matrix.loc[assets, assets].to_numpy(dtype=float)

    number_of_assets = len(assets)
    weights_variable = cp.Variable(number_of_assets)

    objective = cp.Minimize(
        cp.quad_form(weights_variable, covariance)
    )

    constraints = [
        cp.sum(weights_variable) == 1,
        weights_variable >= 0,
        weights_variable <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights_variable.value is None:
        raise RuntimeError(
            f"Minimum-variance optimization failed: {problem.status}"
        )

    weights = pd.Series(
        np.asarray(weights_variable.value).flatten(),
        index=assets,
        name="Minimum Variance Weight",
    )

    weights[weights.abs() < 1e-8] = 0.0
    weights = weights / weights.sum()

    expected_return, volatility, sharpe = (
        calculate_portfolio_statistics(
            weights,
            expected_returns,
            covariance_matrix,
            risk_free_rate,
        )
    )

    return OptimizedPortfolio(
        weights=weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def maximum_sharpe_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> OptimizedPortfolio:
    """
    Approximate the maximum-Sharpe long-only portfolio.

    Uses a transformed convex optimization problem based on excess returns.
    """
    assets = expected_returns.index
    excess_returns = expected_returns - risk_free_rate

    covariance = covariance_matrix.loc[assets, assets].to_numpy(dtype=float)
    excess_array = excess_returns.to_numpy(dtype=float)

    if np.all(excess_array <= 0):
        raise ValueError(
            "No asset has a positive expected excess return."
        )

    number_of_assets = len(assets)

    transformed_weights = cp.Variable(number_of_assets)
    scale = cp.Variable(nonneg=True)

    objective = cp.Minimize(
        cp.quad_form(transformed_weights, covariance)
    )

    constraints = [
        excess_array @ transformed_weights == 1,
        cp.sum(transformed_weights) == scale,
        transformed_weights >= 0,
        transformed_weights <= max_weight * scale,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if transformed_weights.value is None or scale.value is None:
        raise RuntimeError(
            f"Maximum-Sharpe optimization failed: {problem.status}"
        )

    raw_weights = (
        np.asarray(transformed_weights.value).flatten()
        / float(scale.value)
    )

    weights = pd.Series(
        raw_weights,
        index=assets,
        name="Maximum Sharpe Weight",
    )

    weights[weights.abs() < 1e-8] = 0.0
    weights = weights / weights.sum()

    expected_return, volatility, sharpe = (
        calculate_portfolio_statistics(
            weights,
            expected_returns,
            covariance_matrix,
            risk_free_rate,
        )
    )

    return OptimizedPortfolio(
        weights=weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    points: int = 40,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate portfolios along the long-only efficient frontier.

    For each target return, the optimizer finds the portfolio with the
    lowest possible volatility.
    """
    if points < 2:
        raise ValueError("points must be at least 2.")

    assets = expected_returns.index
    mean_returns = expected_returns.to_numpy(dtype=float)
    covariance = covariance_matrix.loc[assets, assets].to_numpy(dtype=float)

    minimum_target = float(expected_returns.min())
    maximum_target = float(expected_returns.max())

    target_returns = np.linspace(
        minimum_target,
        maximum_target,
        points,
    )

    frontier_records: list[dict[str, float]] = []

    for target_return in target_returns:
        weights_variable = cp.Variable(len(assets))

        objective = cp.Minimize(
            cp.quad_form(weights_variable, covariance)
        )

        constraints = [
            cp.sum(weights_variable) == 1,
            weights_variable >= 0,
            weights_variable <= max_weight,
            mean_returns @ weights_variable >= target_return,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if weights_variable.value is None:
            continue

        weights = np.asarray(weights_variable.value).flatten()

        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(
            weights.T @ covariance @ weights
        )
        portfolio_volatility = float(
            np.sqrt(max(portfolio_variance, 0.0))
        )

        record: dict[str, float] = {
            "Expected Return": portfolio_return,
            "Volatility": portfolio_volatility,
        }

        for asset, weight in zip(assets, weights):
            record[f"Weight {asset}"] = float(weight)

        frontier_records.append(record)

    if not frontier_records:
        raise RuntimeError(
            "No feasible portfolios were found for the efficient frontier."
        )

    return pd.DataFrame(frontier_records)