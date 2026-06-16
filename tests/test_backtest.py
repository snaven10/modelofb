"""
test_backtest.py — Pruebas del backtesting temporal.

Verificamos que el split respete el orden temporal y que la comparación contra el
mercado se arme bien. Usamos un modelo DUMMY (probabilidad fija) para aislar la
lógica del backtest de cualquier modelo real.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest_vs_market, temporal_split


class _DummyModel:
    """Modelo de mentira: ignora todo y devuelve una probabilidad fija."""

    def __init__(self, probs=(0.45, 0.25, 0.30)):
        self.probs = probs
        self.fitted_on = None

    def fit(self, df):
        self.fitted_on = len(df)
        return self

    def predict_proba(self, home, away):
        return self.probs


def _df():
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]),
        "home_team": ["A", "B", "C", "D"],
        "away_team": ["B", "C", "D", "A"],
        "result": ["H", "D", "A", "H"],
        "p_market_h": [0.50, 0.30, 0.20, 0.55],
        "p_market_d": [0.25, 0.40, 0.30, 0.25],
        "p_market_a": [0.25, 0.30, 0.50, 0.20],
    })


def test_split_respeta_la_fecha():
    train, test = temporal_split(_df(), "2022-01-01")
    assert len(train) == 2          # 2020 y 2021
    assert len(test) == 2           # 2022 y 2023
    assert train["date"].max() < pd.Timestamp("2022-01-01")
    assert test["date"].min() >= pd.Timestamp("2022-01-01")


def test_backtest_entrena_solo_en_train():
    train, test = temporal_split(_df(), "2022-01-01")
    model = _DummyModel()
    backtest_vs_market(model, train, test)
    # El modelo debe haber visto SOLO los partidos de train (no los de test).
    assert model.fitted_on == 2


def test_backtest_estructura_y_metricas():
    train, test = temporal_split(_df(), "2022-01-01")
    res = backtest_vs_market(_DummyModel(), train, test)
    assert set(res) == {"classes", "model", "market", "logloss_gap", "n_test"}
    assert res["n_test"] == 2
    # gap = log-loss modelo − log-loss mercado, debe ser un número finito.
    assert np.isfinite(res["logloss_gap"])


def test_backtest_falla_sin_odds():
    """Si el test no tiene odds, debe avisar con un error claro, no dar basura."""
    train, test = temporal_split(_df(), "2022-01-01")
    test = test.assign(p_market_h=np.nan, p_market_d=np.nan, p_market_a=np.nan)
    with pytest.raises(ValueError):
        backtest_vs_market(_DummyModel(), train, test)
