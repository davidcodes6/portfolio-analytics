import numpy as np
import pandas as pd
from src.returns import portfolio_returns
from src.performance import total_return, max_drawdown
from src.risk import historical_var
from src.attribution import risk_contributions

def test_weighted_return_known_answer():
    returns = pd.DataFrame({"A": [0.10], "B": [0.00]})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    assert np.isclose(portfolio_returns(returns, weights).iloc[0], 0.05)

def test_total_return_compounds():
    r = pd.Series([0.10, -0.10])
    assert np.isclose(total_return(r), -0.01)

def test_max_drawdown():
    r = pd.Series([0.10, -0.20, 0.05])
    assert np.isclose(max_drawdown(r), -0.20)

def test_historical_var_positive_loss_number():
    r = pd.Series([-0.05, -0.02, 0.01, 0.02, 0.03])
    assert historical_var(r, 0.80) >= 0

def test_risk_contributions_sum_to_volatility():
    idx = pd.date_range("2020-01-01", periods=100)
    returns = pd.DataFrame({
        "A": np.linspace(-0.01, 0.01, 100),
        "B": np.linspace(0.008, -0.008, 100),
    }, index=idx)
    weights = pd.Series({"A": 0.6, "B": 0.4})
    table = risk_contributions(returns, weights)
    assert np.isclose(
        table["Percent Risk Contribution"].sum(), 1.0, atol=1e-8
    )
