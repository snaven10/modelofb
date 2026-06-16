"""
test_compare_market.py — Pruebas del comparador modelo vs mercado (sin red).

Mockeamos el loader de odds y el de internacionales para probar la lógica de
comparación partido a partido: que arme la tabla, detecte el favorito de cada lado
y marque la coincidencia. Sin red, sin créditos.

Correlo con:  pytest -q
"""

import pandas as pd
import pytest

import src.predict.compare_market as CM


def _fake_internacionales():
    """Histórico mínimo: Spain fuerte, Haiti flojo, para que el Elo tenga ratings."""
    filas = []
    for i in range(6):
        filas += [("Spain", "Haiti", 4, 0), ("Haiti", "Spain", 0, 3)]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime("2024-01-01")
    df["neutral"] = False
    df["is_competitive"] = True
    return df


def _patch(monkeypatch, odds_df):
    class _FakeOdds:
        def __init__(self, *a, **k): pass
        def world_cup_match_market(self): return odds_df
    class _FakeIntl:
        def __init__(self, *a, **k): pass
        def load(self, include_unplayed=False): return _fake_internacionales()
    monkeypatch.setattr(CM, "OddsApiLoader", _FakeOdds)
    monkeypatch.setattr(CM, "InternationalResultsLoader", _FakeIntl)


def test_compare_matches_arma_tabla_y_coincidencia(monkeypatch):
    odds = pd.DataFrame([{
        "home_team": "Spain", "away_team": "Haiti", "commence": "x",
        "p_market_h": 0.85, "p_market_d": 0.10, "p_market_a": 0.05,
    }])
    _patch(monkeypatch, odds)
    t = CM.compare_matches()
    assert len(t) == 1
    row = t.iloc[0]
    # Spain fuerte → ambos (modelo y mercado) la hacen favorita local
    assert row["fav_modelo"] == "H"
    assert row["fav_mercado"] == "H"
    assert row["coincide"]
    assert set(t.columns) >= {"partido", "fav_modelo", "fav_mercado", "coincide"}


def test_compare_matches_falla_sin_odds(monkeypatch):
    _patch(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError):
        CM.compare_matches()
