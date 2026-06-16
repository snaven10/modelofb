"""
test_simulate.py — Pruebas de las primitivas de simulación (fase de grupos).

Verificamos conceptos sin red: el neutral promedia orientaciones, la tabla acumula
puntos como FIFA (3/1/0), el ranking respeta puntos→GD→GF, y los 8 mejores terceros
salen ordenados. Usamos un PoissonModel entrenado con un mini-dataset determinista.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson import PoissonModel
from src.predict.simulate import (
    TeamRecord,
    best_eight_thirds,
    expected_goals,
    sample_score,
    simulate_group,
    simulate_group_stage,
)


def _toy_model():
    """Poisson chico: 'Fuerte' golea, 'Flojo' encaja, en casa y fuera."""
    filas = []
    for _ in range(6):
        filas += [("Fuerte", "Flojo", 3, 0), ("Flojo", "Fuerte", 0, 3),
                  ("Medio", "Flojo", 2, 1), ("Fuerte", "Medio", 2, 1)]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    return PoissonModel(max_goals=8).fit(df)


def test_neutral_promedia_orientaciones():
    """En neutral, λ(a,b) debe ser simétrico al rol: el mismo sin importar quién es 'home'."""
    m = _toy_model()
    la1, lb1 = expected_goals(m, "Fuerte", "Medio", neutral=True)
    lb2, la2 = expected_goals(m, "Medio", "Fuerte", neutral=True)
    # Invertir el orden en neutral da los mismos λ por equipo (promedio simétrico).
    assert la1 == pytest.approx(la2)
    assert lb1 == pytest.approx(lb2)


def test_teamrecord_puntos_fifa():
    r = TeamRecord("X")
    r.add(2, 0)   # gana → +3
    r.add(1, 1)   # empata → +1
    r.add(0, 2)   # pierde → +0
    assert r.points == 4
    assert r.gf == 3 and r.ga == 3 and r.gd == 0 and r.played == 3


def test_simulate_group_da_tabla_ordenada():
    m = _toy_model()
    rng = np.random.default_rng(0)
    fix = pd.DataFrame({
        "home_team": ["Fuerte", "Fuerte", "Medio", "Flojo", "Medio", "Flojo"],
        "away_team": ["Medio", "Flojo", "Flojo", "Fuerte", "Fuerte", "Medio"],
        "neutral": [True] * 6,
    })
    tabla = simulate_group(m, fix, rng)
    assert len(tabla) == 3
    # La tabla debe venir ordenada de mayor a menor por (puntos, GD, GF).
    claves = [(r.points, r.gd, r.gf) for r in tabla]
    assert claves == sorted(claves, reverse=True)


def test_best_eight_thirds_elige_ocho():
    """De 12 grupos toma exactamente los 8 mejores terceros (por puntos→GD→GF)."""
    standings = {}
    for i in range(12):
        # tercero de cada grupo con puntos decrecientes para tener orden claro
        r1, r2 = TeamRecord(f"1_{i}"), TeamRecord(f"2_{i}")
        r3 = TeamRecord(f"3_{i}"); r3.points = i  # 0..11
        r4 = TeamRecord(f"4_{i}")
        standings[chr(65 + i)] = [r1, r2, r3, r4]
    rng = np.random.default_rng(1)
    letras = best_eight_thirds(standings, rng)
    assert len(letras) == 8
    # Los 8 con más puntos son los grupos E..L (i=4..11) → letras 'E'..'L'.
    assert set(letras) == set("EFGHIJKL")


def test_sample_score_no_negativo():
    m = _toy_model()
    rng = np.random.default_rng(7)
    for _ in range(50):
        ga, gb = sample_score(m, "Fuerte", "Flojo", neutral=True, rng=rng)
        assert ga >= 0 and gb >= 0
