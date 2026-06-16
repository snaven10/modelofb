"""
test_elo.py — Pruebas del modelo Elo.

No probamos "que corre"; probamos que cada CONCEPTO se cumple: la curva logística,
la ventaja de localía (y que NO aplique en neutral), la actualización de suma cero,
que las sorpresas muevan más, y que las probabilidades 1X2 sumen 1.

Correlo con:  pytest -q
"""

import pandas as pd
import pytest

from src.models.elo import EloModel


def test_ratings_iguales_sin_localia_da_50_50():
    """Dos equipos con el mismo rating, en neutral, deben tener E = 0.5 exacto."""
    elo = EloModel()
    elo.ratings = {"A": 1500, "B": 1500}
    assert elo.expected_score("A", "B", neutral=True) == pytest.approx(0.5)


def test_diferencia_400_da_favorito_909():
    """La convención del Elo: 400 puntos de diferencia ≈ 0.909 esperado."""
    elo = EloModel(home_advantage=0)  # sin localía para aislar la diferencia pura
    elo.ratings = {"Fuerte": 1900, "Debil": 1500}
    assert elo.expected_score("Fuerte", "Debil", neutral=True) == pytest.approx(0.909, abs=1e-3)


def test_ventaja_localia_sube_el_esperado():
    """Con ratings iguales, el local debe esperar MÁS que 0.5... salvo en neutral."""
    elo = EloModel(home_advantage=100)
    elo.ratings = {"A": 1500, "B": 1500}
    assert elo.expected_score("A", "B", neutral=False) > 0.5
    # En cancha neutral la ventaja desaparece -> vuelve a 0.5.
    assert elo.expected_score("A", "B", neutral=True) == pytest.approx(0.5)


def test_actualizacion_es_suma_cero():
    """Lo que sube un equipo es exactamente lo que baja el otro."""
    elo = EloModel()
    elo.ratings = {"A": 1500, "B": 1500}
    elo.update_one("A", "B", home_score=2, away_score=0, neutral=True)
    assert elo.get_rating("A") + elo.get_rating("B") == pytest.approx(3000.0)


def test_la_sorpresa_mueve_mas():
    """
    Un batacazo (gana el débil) mueve MÁS rating que una victoria esperada
    (gana el fuerte). Es el corazón del Elo: (S − E).
    """
    # Victoria esperada: el fuerte le gana al débil.
    esperado = EloModel(home_advantage=0)
    esperado.ratings = {"Fuerte": 1900, "Debil": 1500}
    esperado.update_one("Fuerte", "Debil", 1, 0, neutral=True)
    movimiento_esperado = esperado.get_rating("Fuerte") - 1900

    # Batacazo: el débil le gana al fuerte.
    batacazo = EloModel(home_advantage=0)
    batacazo.ratings = {"Fuerte": 1900, "Debil": 1500}
    batacazo.update_one("Debil", "Fuerte", 1, 0, neutral=True)
    movimiento_batacazo = batacazo.get_rating("Debil") - 1500

    assert movimiento_batacazo > movimiento_esperado


def test_amistoso_mueve_menos_que_competitivo():
    """K modulado: el mismo resultado pesa menos si fue amistoso."""
    comp = EloModel(friendly_weight=0.5)
    comp.ratings = {"A": 1500, "B": 1500}
    comp.update_one("A", "B", 1, 0, neutral=True, is_competitive=True)
    delta_comp = comp.get_rating("A") - 1500

    amis = EloModel(friendly_weight=0.5)
    amis.ratings = {"A": 1500, "B": 1500}
    amis.update_one("A", "B", 1, 0, neutral=True, is_competitive=False)
    delta_amis = amis.get_rating("A") - 1500

    assert delta_amis == pytest.approx(delta_comp * 0.5)


def test_predict_proba_suma_uno():
    elo = EloModel()
    elo.ratings = {"A": 1800, "B": 1500}
    p_h, p_d, p_a = elo.predict_proba("A", "B")
    assert p_h + p_d + p_a == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in (p_h, p_d, p_a))


def test_empate_es_mayor_en_partido_parejo():
    """El empate debe ser más probable entre iguales que cuando hay claro favorito."""
    elo = EloModel()
    elo.ratings = {"Par1": 1500, "Par2": 1500, "Crack": 2100, "Flojo": 1300}

    _, draw_parejo, _ = elo.predict_proba("Par1", "Par2", neutral=True)
    _, draw_dispar, _ = elo.predict_proba("Crack", "Flojo", neutral=True)
    assert draw_parejo > draw_dispar


def test_fit_procesa_y_ordena():
    """fit() debe procesar en orden de fecha y saltar partidos sin marcador."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2019-01-01", "2021-01-01"]),
        "home_team": ["A", "A", "C"],
        "away_team": ["B", "B", "D"],
        "home_score": [3, 1, pd.NA],          # el tercero es un fixture futuro
        "away_score": [0, 1, pd.NA],
        "neutral": [True, True, True],
        "is_competitive": [True, True, True],
    })
    elo = EloModel().fit(df)
    # A jugó dos veces (ganó una, empató otra) -> rating por encima del base.
    assert elo.get_rating("A") > 1500
    # C y D nunca se procesaron (partido sin marcador) -> siguen en base.
    assert elo.get_rating("C") == 1500
    assert elo.get_rating("D") == 1500


# ──────────────────────────────────────────────────────────────────────────────
# Modelo de empate calibrado (curva p_draw(d) = c0 + c1·d + c2·d²)
# ──────────────────────────────────────────────────────────────────────────────
def test_draw_prob_usa_la_curva_calibrada_si_existe():
    """Con draw_coef seteado, _draw_prob usa la cuadrática (no la heurística)."""
    m = EloModel()
    m.draw_coef = (0.25, 0.15, -0.38)        # curva tipo la calibrada de datos
    # d = |2·0.5−1| = 0 (partido parejo) -> c0
    assert m._draw_prob(0.5) == pytest.approx(0.25, abs=1e-9)


def test_draw_prob_fallback_a_heuristica_sin_calibrar():
    """Sin draw_coef (None), usa la heurística vieja draw_max·(1−d)."""
    m = EloModel(draw_max=0.28)
    assert m.draw_coef is None
    assert m._draw_prob(0.5) == pytest.approx(0.28)          # d=0 -> draw_max
    assert m._draw_prob(1.0) == pytest.approx(0.0, abs=1e-9)  # d=1 -> 0


def test_draw_prob_clampea_a_rango_sano():
    """La cuadrática nunca devuelve negativos ni valores absurdos."""
    m = EloModel()
    m.draw_coef = (0.25, 0.15, -0.38)
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = m._draw_prob(e)
        assert 0.02 <= p <= 0.45


def test_curva_calibrada_es_mas_plana_que_la_heuristica():
    """
    El hallazgo clave: en favorito MEDIO la tasa real de empate es ~0.24, no ~0.15.
    La curva calibrada debe dar MÁS empate que la heurística vieja en ese rango.
    """
    cal = EloModel(); cal.draw_coef = (0.253, 0.151, -0.377)
    heur = EloModel(draw_max=0.28)            # sin calibrar
    e_medio = 0.725                            # d ≈ 0.45 (favorito medio)
    assert cal._draw_prob(e_medio) > heur._draw_prob(e_medio)


def test_predict_proba_suma_uno_con_curva_calibrada():
    m = EloModel(); m.draw_coef = (0.253, 0.151, -0.377)
    ph, pd_, pa = m.predict_proba("X", "Y")    # equipos nuevos -> parejo
    assert ph + pd_ + pa == pytest.approx(1.0)


def test_calibrate_no_actua_con_pocos_datos():
    """Con menos de 200 partidos, no calibra (deja la heurística) para no sobre-ajustar."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "home_team": ["A", "B"], "away_team": ["B", "A"],
        "home_score": [1, 2], "away_score": [1, 0],
        "neutral": [False, False],
    })
    m = EloModel().fit(df)
    assert m.draw_coef is None                 # muy pocos datos -> sin calibrar
