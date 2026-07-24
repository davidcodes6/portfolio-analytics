from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def growth_chart(portfolio_wealth: pd.Series, benchmark_wealth: pd.Series) -> go.Figure:
    frame = pd.concat([
        portfolio_wealth.rename("Portfolio"),
        benchmark_wealth.rename("Benchmark")
    ], axis=1).dropna()
    fig = px.line(frame, title="Growth of Initial Capital")
    fig.update_layout(yaxis_title="Portfolio Value", xaxis_title="")
    return fig

def drawdown_chart(portfolio_dd: pd.Series, benchmark_dd: pd.Series) -> go.Figure:
    frame = pd.concat([
        portfolio_dd.rename("Portfolio"),
        benchmark_dd.rename("Benchmark")
    ], axis=1).dropna()
    fig = px.area(frame, title="Drawdown (Underwater) Analysis")
    fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Drawdown", xaxis_title="")
    return fig

def rolling_chart(rolling: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = px.line(rolling[columns], title=title)
    fig.update_layout(xaxis_title="")
    return fig

def risk_contribution_chart(table: pd.DataFrame) -> go.Figure:
    frame = table[["Weight", "Percent Risk Contribution"]].reset_index(names="Asset")
    long = frame.melt("Asset", var_name="Measure", value_name="Value")
    fig = px.bar(long, x="Asset", y="Value", color="Measure", barmode="group",
                 title="Capital Weight vs. Risk Contribution")
    fig.update_layout(yaxis_tickformat=".1%")
    return fig

def stress_chart(stress: pd.DataFrame) -> go.Figure:
    cols = ["Portfolio Return", "Benchmark Return"]
    frame = stress[cols].reset_index().melt("Scenario", var_name="Series", value_name="Return")
    fig = px.bar(frame, x="Scenario", y="Return", color="Series", barmode="group",
                 title="Historical Stress-Test Performance")
    fig.update_layout(yaxis_tickformat=".1%")
    return fig
