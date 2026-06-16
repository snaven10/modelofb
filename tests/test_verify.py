"""
test_verify.py — Pruebas de la verificación retrospectiva del Mundial.

Lo CRÍTICO a verificar es que NO haya leakage: el modelo se entrena solo con partidos
anteriores al cutoff, y se evalúa sobre los del Mundial ya jugados. Mockeamos el loader
con un mini-dataset (sin red) y comprobamos el split, el conteo de aciertos y que las
métricas se calculen.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import src.predict.verify as V
from src.predict.verify import verify_world_cup


def _fake_df():
    filas = [
        # PRE-Mundial (entrenamiento): Spain fuerte, Cape Verde flojo
        ("2025-09-01", "Spain", "Cape Verde", 4, 0, "Friendly", False),
        ("2025-10-01", "Spain", "Cape Verde", 3, 0, "Friendly", False),
        ("2025-11-01", "Cape Verde", "Spain", 0, 2, "Friendly", False),
        # MUNDIAL jugado (a verificar)
        ("2026-06-12", "Spain", "Cape Verde", 1, 0, "FIFA World Cup", True),
        ("2026-06-13", "Cape Verde", "Spain", 0, 3, "FIFA World Cup", True),
        # MUNDIAL futuro (sin marcador) — NO debe entrar a la verificación
        ("2026-06-20", "Spain", "Cape Verde", None, None, "FIFA World Cup", True),
    ]
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team",
                                      "home_score", "away_score", "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["is_competitive"] = df["tournament"] != "Friendly"
    df["result"] = pd.NA
    return df


def _patch(monkeypatch, df):
    class _FakeLoader:
        def __init__(self, *a, **k): pass
        def load(self, include_unplayed=False):
            return df if include_unplayed else df.dropna(subset=["home_score", "away_score"])
    monkeypatch.setattr(V, "InternationalResultsLoader", _FakeLoader)


def test_solo_verifica_partidos_jugados_del_mundial(monkeypatch):
    _patch(monkeypatch, _fake_df())
    tabla, m = verify_world_cup(cutoff="2026-06-11")
    # 2 partidos del Mundial jugados (el futuro sin marcador NO cuenta).
    assert m["n"] == 2
    assert len(tabla) == 2


def test_no_hay_leakage_acierta_a_spain(monkeypatch):
    """Entrenado solo pre-Mundial, Spain debe ser favorita y ganar sus 2 partidos."""
    _patch(monkeypatch, _fake_df())
    tabla, m = verify_world_cup(cutoff="2026-06-11")
    # Spain ganó ambos (1-0 y 0-3 como visita) → el modelo debería acertar los 2.
    assert m["aciertos"] == 2
    assert m["accuracy"] == pytest.approx(1.0)


def test_metricas_presentes(monkeypatch):
    _patch(monkeypatch, _fake_df())
    _, m = verify_world_cup(cutoff="2026-06-11")
    assert set(m) >= {"n", "aciertos", "accuracy", "log_loss", "log_loss_azar"}
    assert np.isfinite(m["log_loss"])


def test_falla_sin_partidos_jugados(monkeypatch):
    solo_pre = _fake_df()
    solo_pre = solo_pre[solo_pre["date"] < "2026-06-01"]
    _patch(monkeypatch, solo_pre)
    with pytest.raises(ValueError):
        verify_world_cup(cutoff="2026-06-11")
