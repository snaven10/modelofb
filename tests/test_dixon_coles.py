"""
test_dixon_coles.py — Pruebas del modelo Dixon-Coles.

Verificamos las DOS contribuciones: (1) la corrección τ con ρ<0 infla los empates
bajos (0-0, 1-1) respecto al Poisson puro; (2) el decay temporal ξ hace que un
partido reciente pese más que uno viejo. Y que con ρ=0, ξ=0 recupere el Poisson.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel
from src.models.poisson import PoissonModel


def _mini_df():
    filas = [
        ("Fuerte", "Flojo", 5, 0),
        ("Fuerte", "Medio", 3, 1),
        ("Medio", "Flojo", 2, 0),
        ("Medio", "Fuerte", 1, 3),
        ("Flojo", "Fuerte", 0, 4),
        ("Flojo", "Medio", 1, 2),
    ]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    # fechas crecientes para poder probar el decay temporal
    df["date"] = pd.to_datetime(
        ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"])
    return df


def test_rho_cero_xi_cero_es_poisson():
    """Sin corrección ni decay, Dixon-Coles debe dar lo mismo que Poisson."""
    df = _mini_df()
    dc = DixonColesModel(rho=0.0, xi=0.0).fit(df)
    po = PoissonModel().fit(df)
    p_dc = dc.predict_proba("Medio", "Medio")
    p_po = po.predict_proba("Medio", "Medio")
    assert p_dc == pytest.approx(p_po, abs=1e-9)


def test_rho_negativo_infla_empates_bajos():
    """Con ρ<0, P(0-0) debe ser MAYOR que con el Poisson puro (ρ=0)."""
    df = _mini_df()
    base = DixonColesModel(rho=0.0).fit(df)
    corregido = DixonColesModel(rho=-0.15).fit(df)

    m_base = base.score_matrix("Medio", "Medio")
    m_corr = corregido.score_matrix("Medio", "Medio")
    assert m_corr[0, 0] > m_base[0, 0]   # 0-0 sube
    assert m_corr[1, 1] > m_base[1, 1]   # 1-1 sube


def test_matriz_corregida_suma_uno():
    """Tras τ y la renormalización, la matriz debe seguir sumando 1."""
    dc = DixonColesModel(rho=-0.12, max_goals=15).fit(_mini_df())
    m = dc.score_matrix("Medio", "Medio")
    assert m.sum() == pytest.approx(1.0, abs=1e-9)


def test_predict_proba_suma_uno():
    dc = DixonColesModel(rho=-0.10).fit(_mini_df())
    p_h, p_d, p_a = dc.predict_proba("Fuerte", "Flojo")
    assert p_h + p_d + p_a == pytest.approx(1.0)


def test_decay_da_mas_peso_a_lo_reciente():
    """
    Un equipo que mejoró con el tiempo debe verse MÁS fuerte con decay alto.
    'Subiendo' marca poco al principio y mucho al final; 'Constante' siempre igual
    (ancla la media de la liga). Con ξ alto, el ataque RELATIVO de Subiendo sube,
    porque sus goleadas recientes pesan más. (Hace falta más de un local: con uno
    solo, su ataque es 1.0 por definición — es su propia media.)
    """
    filas = [
        ("Subiendo", "Rival", 0, 0),   # viejo: flojo
        ("Constante", "Rival", 2, 0),
        ("Subiendo", "Rival", 0, 1),
        ("Constante", "Rival", 2, 0),
        ("Subiendo", "Rival", 4, 0),   # reciente: golea
        ("Constante", "Rival", 2, 0),
        ("Subiendo", "Rival", 5, 0),
        ("Constante", "Rival", 2, 0),
    ]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(
        ["2019-01-01", "2019-06-01", "2021-01-01", "2021-06-01",
         "2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"])

    sin_decay = DixonColesModel(rho=0.0, xi=0.0).fit(df)
    con_decay = DixonColesModel(rho=0.0, xi=1.0).fit(df)
    # El ataque relativo de Subiendo debe ser mayor cuando lo reciente pesa más.
    assert con_decay.attack_home["Subiendo"] > sin_decay.attack_home["Subiendo"]
