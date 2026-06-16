"""
test_world_cup.py — Pruebas de la predicción del Mundial 2026.

Verificamos los CONCEPTOS sin tocar red ni el dataset real: sembramos un mini-dataset
con partidos jugados (para entrenar) + fixtures del Mundial sin marcador (a predecir),
y comprobamos que el split separe bien, que solo se prediga lo no jugado, que las
probabilidades sumen 1 y que la `neutral` de cada fixture se respete.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import src.predict.world_cup as wc
from src.predict.world_cup import _split_played_vs_fixtures, predict_world_cup


def _fake_df():
    """Jugados (con marcador) + 2 fixtures del Mundial 2026 (sin marcador)."""
    filas = [
        # date, home, away, hs, as, tournament, neutral
        ("2025-09-01", "Brazil", "Chile", 3, 0, "FIFA World Cup qualification", False),
        ("2025-09-05", "Argentina", "Chile", 2, 0, "FIFA World Cup qualification", False),
        ("2025-09-10", "Brazil", "Argentina", 1, 1, "Friendly", True),
        ("2026-06-12", "Brazil", "Chile", None, None, "FIFA World Cup", True),
        ("2026-06-13", "Argentina", "Brazil", None, None, "FIFA World Cup", True),
    ]
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team",
                                      "home_score", "away_score", "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["is_competitive"] = df["tournament"] != "Friendly"
    df["result"] = pd.NA
    return df


def _patch_loader(monkeypatch, df):
    """Hace que el loader devuelva nuestro mini-dataset en vez de leer del disco."""
    class _FakeLoader:
        def __init__(self, *a, **k): pass
        def load(self, include_unplayed=False):
            return df if include_unplayed else df.dropna(subset=["home_score", "away_score"])
    monkeypatch.setattr(wc, "InternationalResultsLoader", _FakeLoader)


def test_split_separa_jugados_de_fixtures():
    played, fixtures = _split_played_vs_fixtures(_fake_df())
    assert len(played) == 3                      # los 3 con marcador
    assert len(fixtures) == 2                     # los 2 del Mundial sin marcador
    assert fixtures["home_score"].isna().all()    # ningún fixture tiene marcador
    assert (fixtures["tournament"] == "FIFA World Cup").all()


def test_predict_solo_los_fixtures_y_suma_uno(monkeypatch):
    _patch_loader(monkeypatch, _fake_df())
    pred = predict_world_cup()
    assert len(pred) == 2
    sumas = pred[["p_home", "p_draw", "p_away"]].sum(axis=1).to_numpy()
    assert np.allclose(sumas, 1.0)
    assert set(pred["favorito"]).issubset({"H", "D", "A"})


def test_neutral_se_respeta(monkeypatch):
    """Ambos fixtures son neutral=True → el predictor debe pasarlo al Elo."""
    _patch_loader(monkeypatch, _fake_df())
    pred = predict_world_cup()
    assert pred["neutral"].all()


def test_falla_si_no_hay_fixtures(monkeypatch):
    """Sin fixtures del Mundial, debe avisar claro en vez de devolver vacío."""
    solo_jugados = _fake_df().dropna(subset=["home_score", "away_score"])
    _patch_loader(monkeypatch, solo_jugados)
    with pytest.raises(ValueError):
        predict_world_cup()
