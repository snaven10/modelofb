"""
test_elo_poisson.py — Pruebas del modelo de goles ajustado por rival (EloGoalModel).

Verificamos el CONCEPTO que arregla el sesgo: la supremacía de goles sale de la
diferencia de Elo, así que un equipo fuerte genera más λ contra uno débil, y la
localía sube el λ del local. Y que predict_proba sume 1.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.elo_poisson import EloGoalModel


def _toy_df():
    """Liga de 3 niveles: Fuerte le gana a Medio que le gana a Flojo, repetido."""
    filas = []
    for _ in range(10):
        filas += [
            ("Fuerte", "Flojo", 3, 0), ("Flojo", "Fuerte", 0, 3),
            ("Fuerte", "Medio", 2, 0), ("Medio", "Fuerte", 0, 2),
            ("Medio", "Flojo", 2, 0), ("Flojo", "Medio", 0, 2),
        ]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime("2020-01-01")
    df["neutral"] = False
    return df


def test_calibra_total_y_pendiente():
    m = EloGoalModel().fit(_toy_df())
    assert m.total_base > 0            # media de goles totales
    assert m.b > 0                     # más Elo => más goles, pendiente positiva


def test_fuerte_genera_mas_goles_que_debil():
    """λ del equipo fuerte > λ del débil cuando se enfrentan (ajuste por rival)."""
    m = EloGoalModel().fit(_toy_df())
    lf, ld = m.expected_goals("Fuerte", "Flojo")
    assert lf > ld


def test_localia_sube_lambda_del_local():
    """El mismo cruce con roles invertidos: ser local aumenta tu λ."""
    m = EloGoalModel().fit(_toy_df())
    # Fuerte de local vs Fuerte de visita contra el mismo rival:
    lf_home, _ = m.expected_goals("Fuerte", "Medio")
    _, lf_away = m.expected_goals("Medio", "Fuerte")
    assert lf_home > lf_away           # la localía suma


def test_lambda_nunca_negativo():
    """Aun con una diferencia de Elo enorme, λ no baja de 0."""
    m = EloGoalModel().fit(_toy_df())
    lf, ld = m.expected_goals("Fuerte", "Flojo")
    assert lf >= 0.05 and ld >= 0.05


def test_predict_proba_suma_uno():
    m = EloGoalModel().fit(_toy_df())
    assert sum(m.predict_proba("Fuerte", "Flojo")) == pytest.approx(1.0)
