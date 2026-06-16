"""
test_talent.py — Pruebas del ajuste por talento de plantel (sin red).

Probamos adjust_elo_by_talent: w=0 deja el Elo igual, w=1 ordena por valor de plantel,
y un equipo de plantel caro con Elo bajo sube su rating. squad_values es estático
(sin red), así que el test es puro.

Correlo con:  pytest -q
"""

import pytest

from src.data_loaders.squad_values import squad_values
from src.models.elo import EloModel
from src.predict.talent import adjust_elo_by_talent


def _elo_con_ratings():
    """Elo con ratings inventados para las selecciones del Mundial."""
    elo = EloModel()
    # a todas las del Mundial les damos un rating arbitrario distinto del base
    for i, t in enumerate(squad_values()):
        elo.ratings[t] = 1500.0 + i  # ratings crecientes, arbitrarios
    return elo


def test_w_cero_no_cambia_el_elo():
    elo = _elo_con_ratings()
    antes = {t: elo.get_rating(t) for t in squad_values()}
    adjust_elo_by_talent(elo, w=0.0)
    for t, r in antes.items():
        assert elo.get_rating(t) == pytest.approx(r, abs=1e-6)


def test_w_uno_ordena_por_valor_de_plantel():
    elo = _elo_con_ratings()
    adjust_elo_by_talent(elo, w=1.0)
    sv = squad_values()
    # Francia (mayor valor) debe quedar con mayor rating que Qatar (menor valor)
    assert elo.get_rating("France") > elo.get_rating("Qatar")
    # el orden de ratings debe seguir el orden de valores
    top_val = max(sv, key=sv.get)
    top_rat = max(sv, key=lambda t: elo.get_rating(t))
    assert top_val == top_rat


def test_plantel_caro_con_elo_bajo_sube():
    """Un equipo de valor alto pero Elo bajo debe SUBIR su rating al mezclar talento."""
    elo = _elo_con_ratings()
    # forzamos: Ivory Coast (valor alto, 522M) con Elo bajo
    elo.ratings["Ivory Coast"] = 1500.0
    antes = elo.get_rating("Ivory Coast")
    adjust_elo_by_talent(elo, w=0.5)
    assert elo.get_rating("Ivory Coast") > antes


def test_rechaza_w_invalido():
    elo = _elo_con_ratings()
    with pytest.raises(ValueError):
        adjust_elo_by_talent(elo, w=1.5)


def test_verify_acepta_talento_y_produce_metricas(monkeypatch):
    """
    verify_world_cup(talent_w>0) debe correr, ajustar el Elo por talento y dar métricas
    válidas con efecto medible. (Que el log-loss MEJORE es un hallazgo de datos REALES
    —1.125→1.095 sobre los 16 partidos—, no algo que un juguete reproduzca robustamente.)
    """
    import numpy as np
    import pandas as pd
    import src.predict.verify as V

    filas = []
    for i in range(8):
        filas.append((f"2025-0{i + 1}-01", "France", "Qatar", 2, 1, "Friendly", False))
        filas.append((f"2025-0{i + 1}-15", "Spain", "Haiti", 3, 0, "Friendly", False))
    filas.append(("2026-06-12", "France", "Spain", 1, 0, "FIFA World Cup", True))
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"]); df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64"); df["is_competitive"] = df["tournament"] != "Friendly"
    df["result"] = pd.NA

    class _Fake:
        def __init__(self, *a, **k): pass
        def load(self, include_unplayed=False): return df
    monkeypatch.setattr(V, "InternationalResultsLoader", _Fake)

    _, m0 = V.verify_world_cup(cutoff="2026-06-11", talent_w=0.0)
    _, m1 = V.verify_world_cup(cutoff="2026-06-11", talent_w=0.5)
    assert m1["n"] == m0["n"] == 1
    assert np.isfinite(m1["log_loss"])
    assert m1["log_loss"] != m0["log_loss"]   # el talento cambia la predicción
