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


def _walk_df():
    """Warmup (pre-2018) + período de test, con un fuerte y un flojo claros."""
    filas = []
    # warmup: Fuerte le gana a Flojo repetido (2015-2017)
    for i in range(8):
        filas.append((f"201{5+i%3}-0{1+i%9}-01", "Fuerte", "Flojo", 3, 0, "Friendly", False))
    # test (>=2018): más partidos para predecir walk-forward
    for i in range(6):
        filas.append((f"2018-0{1+i}-01", "Fuerte", "Flojo", 2, 0, "FIFA World Cup qualification", False))
        filas.append((f"2019-0{1+i}-01", "Flojo", "Fuerte", 0, 2, "Friendly", False))
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team",
                                      "home_score", "away_score", "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["is_competitive"] = df["tournament"] != "Friendly"
    df["result"] = pd.NA
    return df


def test_walk_forward_devuelve_metricas_y_calibracion(monkeypatch):
    from src.predict.verify import walk_forward_verify
    _patch(monkeypatch, _walk_df())
    m, calib = walk_forward_verify(start="2018-01-01")
    assert set(m) >= {"n", "accuracy", "log_loss", "brier", "log_loss_azar", "pct_empates"}
    assert m["n"] == 12                      # 12 partidos de test (>=2018)
    assert calib.shape[1] == 3               # tabla de calibración (prob, freq, n) por bin


def test_walk_forward_acierta_al_fuerte(monkeypatch):
    """Fuerte siempre le gana a Flojo → el walk-forward debe acertar casi todo."""
    from src.predict.verify import walk_forward_verify
    _patch(monkeypatch, _walk_df())
    m, _ = walk_forward_verify(start="2018-01-01")
    assert m["accuracy"] >= 0.9
    assert m["log_loss"] < m["log_loss_azar"]


def _wc_repetido_df():
    """Mundial donde Spain juega DOS veces (para que el walk-forward tenga efecto)."""
    filas = [
        # warmup pre-Mundial: Spain fuerte
        ("2025-09-01", "Spain", "Qatar", 3, 0, "Friendly", False),
        ("2025-10-01", "Spain", "Cape Verde", 2, 0, "Friendly", False),
        ("2025-11-01", "Qatar", "Cape Verde", 1, 1, "Friendly", False),
        # Mundial: Spain juega 2 veces (golea primero → su rating sube)
        ("2026-06-12", "Spain", "Cape Verde", 5, 0, "FIFA World Cup", True),
        ("2026-06-17", "Spain", "Qatar", 1, 0, "FIFA World Cup", True),
    ]
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team",
                                      "home_score", "away_score", "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    df["is_competitive"] = df["tournament"] != "Friendly"
    df["result"] = pd.NA
    return df


def test_walk_forward_tiene_efecto_con_equipo_repetido(monkeypatch):
    """
    Si un equipo juega 2 veces, el walk-forward debe cambiar la predicción del 2º
    partido (ya incorporó el 1º). Con corte fijo, ambos usan el mismo Elo congelado.
    """
    _patch(monkeypatch, _wc_repetido_df())
    from src.predict.verify import verify_world_cup
    t_fijo, _ = verify_world_cup(cutoff="2026-06-11", walk_forward=False)
    t_wf, _ = verify_world_cup(cutoff="2026-06-11", walk_forward=True)

    # La fila del 2º partido de Spain (vs Qatar) debe tener p_real distinto.
    p_fijo = t_fijo[t_fijo["partido"].str.contains("Qatar")]["p_real"].iloc[0]
    p_wf = t_wf[t_wf["partido"].str.contains("Qatar")]["p_real"].iloc[0]
    assert p_fijo != p_wf            # el walk-forward incorporó la goleada previa
