"""
test_bracket.py — Pruebas de la estructura oficial del bracket y su simulación.

Verificamos que la estructura codificada sea CONSISTENTE con las reglas oficiales:
los 48 equipos repartidos en 12 grupos sin repetidos, 16 partidos en R32 con 8 slots
de tercero, el árbol de llaves bien encadenado, y que la asignación de terceros respete
SIEMPRE la elegibilidad oficial. La simulación de un partido a muerte nunca empata.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson import PoissonModel
from src.predict import bracket as B


def test_grupos_cubren_48_equipos_sin_repetir():
    todos = [t for eq in B.GROUPS_BY_LETTER.values() for t in eq]
    assert len(B.GROUPS_BY_LETTER) == 12
    assert all(len(eq) == 4 for eq in B.GROUPS_BY_LETTER.values())
    assert len(todos) == 48
    assert len(set(todos)) == 48          # ningún equipo en dos grupos


def test_r32_tiene_16_partidos_y_8_slots_de_tercero():
    assert len(B.R32) == 16
    slots_tercero = [m for m, (h, a) in B.R32.items() if a[0] == "3" or h[0] == "3"]
    assert len(slots_tercero) == 8
    assert set(B.THIRD_ELIGIBLE) == set(slots_tercero)
    # cada slot de tercero declara 5 grupos elegibles
    assert all(len(e) == 5 for e in B.THIRD_ELIGIBLE.values())


def test_arbol_de_llaves_bien_encadenado():
    # Los feeds de cada partido deben existir como partidos previos.
    validos = set(B.R32) | set(B.KO_FEEDS)
    for m, (fh, fa) in B.KO_FEEDS.items():
        assert fh in validos and fa in validos
    assert 104 in B.KO_FEEDS              # la final existe
    # Eliminatorias: R32=16, R16=8, QF=4, SF=2, Final=1 = 31 partidos (sin 3er puesto).
    assert len(B.R32) == 16
    assert len(B.KO_FEEDS) == 8 + 4 + 2 + 1   # 15
    assert len(B.R32) + len(B.KO_FEEDS) == 31


def test_assign_thirds_respeta_elegibilidad():
    """Cualquier combinación válida de 8 grupos debe asignarse a slots elegibles."""
    rng = np.random.default_rng(3)
    # tomamos 8 letras que sabemos cubren un matching (todas menos A y B y dos más)
    for combo in (list("EFGHIJKL"), list("CDEFGHIJ"[:8]), list("ABEFHIJK")):
        asign = B.assign_thirds(combo, rng)
        assert len(asign) == 8
        for slot, letra in asign.items():
            assert letra in B.THIRD_ELIGIBLE[slot]    # respeta elegibilidad
        assert len(set(asign.values())) == 8          # biyección: sin repetir letra


def _toy_model():
    filas = []
    for _ in range(8):
        filas += [("A", "B", 2, 0), ("B", "A", 0, 2), ("A", "C", 3, 0), ("C", "A", 1, 2)]
    df = pd.DataFrame(filas, columns=["home_team", "away_team", "home_score", "away_score"])
    return PoissonModel(max_goals=8).fit(df)


def test_play_knockout_nunca_empata():
    m = _toy_model()
    rng = np.random.default_rng(5)
    for _ in range(100):
        w = B.play_knockout(m, "A", "B", rng)
        assert w in ("A", "B")            # siempre sale UN ganador


def test_resolve_spec():
    from src.predict.simulate import TeamRecord
    standings = {"E": [TeamRecord("g1"), TeamRecord("g2"), TeamRecord("g3"), TeamRecord("g4")]}
    assert B._resolve_spec(("W", "E"), standings, {}) == "g1"
    assert B._resolve_spec(("R", "E"), standings, {}) == "g2"
    assert B._resolve_spec(("3", 74), standings, {74: "E"}) == "g3"
