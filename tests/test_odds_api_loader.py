"""
test_odds_api_loader.py — Pruebas del loader de The Odds API (sin red ni créditos).

Mockeamos la respuesta de la API y probamos NUESTRA lógica: la conversión de odds
decimales a probabilidad implícita normalizada (sin overround) y el parseo del mercado
outright. No tocamos la red ni gastamos cuota.

Correlo con:  pytest -q
"""

import pytest

from src.data_loaders.odds_api_loader import OddsApiLoader, _implied_probs


def test_implied_probs_quita_overround_y_suma_uno():
    # Odds decimales con margen: 1/2.0 + 1/4.0 + 1/4.0 = 1.0 justo, sin margen.
    # Metemos margen: odds más bajas → suma >1 → normalizamos.
    probs = _implied_probs({"A": [2.0], "B": [4.0], "C": [4.0]})
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["A"] > probs["B"]            # menor odd = mayor probabilidad
    assert probs["A"] == pytest.approx(0.5)   # 0.5/(0.5+0.25+0.25)=0.5


def test_implied_probs_promedia_varias_casas():
    # dos casas para A: 2.0 y 3.0 → promedio 2.5
    probs = _implied_probs({"A": [2.0, 3.0], "B": [2.0]})
    # avg A=2.5 -> 0.4 ; B=2.0 -> 0.5 ; normalizado
    assert probs["B"] > probs["A"]


def test_world_cup_winner_market_parsea_outrights(monkeypatch, tmp_path):
    loader = OddsApiLoader(cache_dir=tmp_path, use_cache=False)
    fake = [{
        "bookmakers": [{
            "markets": [{
                "key": "outrights",
                "outcomes": [
                    {"name": "Spain", "price": 5.0},
                    {"name": "Brazil", "price": 10.0},
                    {"name": "Haiti", "price": 100.0},
                ],
            }],
        }],
    }]
    monkeypatch.setattr(loader, "_get", lambda *a, **k: fake)
    probs = loader.world_cup_winner_market()
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Spain"] > probs["Brazil"] > probs["Haiti"]   # favorito → más prob


def test_world_cup_match_market_arma_1x2(monkeypatch, tmp_path):
    loader = OddsApiLoader(cache_dir=tmp_path, use_cache=False)
    fake = [{
        "home_team": "Spain", "away_team": "Haiti", "commence_time": "2026-06-20T18:00:00Z",
        "bookmakers": [{
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Spain", "price": 1.2},
                    {"name": "Haiti", "price": 15.0},
                    {"name": "Draw", "price": 7.0},
                ],
            }],
        }],
    }]
    monkeypatch.setattr(loader, "_get", lambda *a, **k: fake)
    df = loader.world_cup_match_market()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["p_market_h"] > row["p_market_d"] > row["p_market_a"]
    assert row[["p_market_h", "p_market_d", "p_market_a"]].sum() == pytest.approx(1.0)
