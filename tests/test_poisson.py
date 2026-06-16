"""
test_poisson.py — Pruebas del modelo Poisson.

Probamos los CONCEPTOS: la matriz de marcadores suma 1, las probabilidades 1X2
suman 1, un equipo goleador genera más λ, la localía da λ_local mayor, y el empate
sale de la diagonal. Construimos un mini-dataset a mano (sin red) con resultados
controlados para que las fuerzas sean predecibles.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson import PoissonModel


def _mini_df() -> pd.DataFrame:
    """
    Liga juguete de 3 equipos. 'Fuerte' golea, 'Flojo' encaja. Cada par juega
    de local y de visita para que haya stats en ambos roles.
    """
    filas = [
        # home, away, hs, as
        ("Fuerte", "Flojo", 5, 0),
        ("Fuerte", "Medio", 3, 1),
        ("Medio", "Flojo", 2, 0),
        ("Medio", "Fuerte", 1, 3),
        ("Flojo", "Fuerte", 0, 4),
        ("Flojo", "Medio", 1, 2),
    ]
    return pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])


def test_matriz_marcadores_suma_uno():
    """
    La matriz de Poisson (truncada en max_goals) debe sumar ~1 cuando el corte es
    holgado respecto a λ. Usamos un cruce parejo (λ moderado, como el fútbol real);
    un favorito extremo dispararía λ y necesitaría un corte mayor.
    """
    m = PoissonModel(max_goals=15).fit(_mini_df())
    matriz = m.score_matrix("Medio", "Medio")
    assert matriz.sum() == pytest.approx(1.0, abs=1e-4)


def test_predict_proba_suma_uno():
    m = PoissonModel().fit(_mini_df())
    p_h, p_d, p_a = m.predict_proba("Fuerte", "Flojo")
    assert p_h + p_d + p_a == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in (p_h, p_d, p_a))


def test_equipo_fuerte_mayor_lambda():
    """El ataque de 'Fuerte' debe generar más goles esperados que el de 'Flojo'."""
    m = PoissonModel().fit(_mini_df())
    lam_fuerte, _ = m.expected_goals("Fuerte", "Medio")
    lam_flojo, _ = m.expected_goals("Flojo", "Medio")
    assert lam_fuerte > lam_flojo


def test_favorito_claro_gana_la_mayor_probabilidad():
    """Fuerte de local vs Flojo: la probabilidad de victoria local debe dominar."""
    m = PoissonModel().fit(_mini_df())
    p_h, p_d, p_a = m.predict_proba("Fuerte", "Flojo")
    assert p_h > p_d and p_h > p_a


def test_empate_es_la_diagonal():
    """P(empate) de predict_proba debe igualar la traza (diagonal) de la matriz."""
    m = PoissonModel().fit(_mini_df())
    matriz = m.score_matrix("Medio", "Medio")
    _, p_draw, _ = m.predict_proba("Medio", "Medio")
    assert p_draw == pytest.approx(np.trace(matriz) / matriz.sum())


def test_equipo_desconocido_no_rompe():
    """Un equipo nunca visto usa fuerza neutral (1.0), no explota."""
    m = PoissonModel().fit(_mini_df())
    lam_h, lam_a = m.expected_goals("Inexistente", "OtroQueNoExiste")
    # Con fuerzas 1.0, λ debe ser exactamente las medias de la liga.
    assert lam_h == pytest.approx(m.home_mean)
    assert lam_a == pytest.approx(m.away_mean)


def test_over_entre_cero_y_uno():
    m = PoissonModel().fit(_mini_df())
    over = m.prob_over("Fuerte", "Flojo", 2.5)
    assert 0.0 <= over <= 1.0
