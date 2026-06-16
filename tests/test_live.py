"""
test_live.py — Pruebas de la simulación condicionada (sin red).

Lo clave: que respete los resultados YA jugados (condicionamiento) y solo simule lo
pendiente. Un grupo con todos los partidos fijos da tabla DETERMINISTA (no toca el
modelo); un partido pendiente sí invoca el muestreo.
"""

import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel
from src.predict.live import _simulate_group_conditional


def test_resultados_fijos_dan_tabla_determinista():
    # A gana sus 3 partidos → primero con 9 puntos; el modelo NO se usa (todo fijo)
    rows = [("A", "B", True, 3, 0), ("A", "C", True, 2, 0), ("A", "D", True, 1, 0),
            ("B", "C", True, 1, 1), ("B", "D", True, 2, 0), ("C", "D", True, 1, 0)]
    tabla = _simulate_group_conditional(model=None, rows=rows, rng=np.random.default_rng(0))
    assert tabla[0].team == "A"
    assert tabla[0].points == 9
    assert tabla[0].gf == 6 and tabla[0].ga == 0


def test_partido_pendiente_se_simula():
    """Si un partido está pendiente (None), debe muestrearse con el modelo (no romper)."""
    df = pd.DataFrame({"home_team": ["A", "B"], "away_team": ["B", "A"],
                       "home_score": [2, 0], "away_score": [0, 2]})
    model = PoissonModel(max_goals=6).fit(df)
    rows = [("A", "B", True, 2, 0),       # jugado: A 2-0 B
            ("A", "C", True, None, None),  # pendiente: se simula
            ("B", "C", True, 1, 1),
            ("A", "D", True, 3, 0), ("B", "D", True, 1, 0), ("C", "D", True, 0, 0)]
    tabla = _simulate_group_conditional(model, rows, np.random.default_rng(1))
    assert len(tabla) == 4                 # 4 equipos en el grupo
    # A ganó sus partidos jugados → debe estar arriba
    assert tabla[0].team == "A"
