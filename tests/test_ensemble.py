"""
test_ensemble.py — Pruebas del ensemble de modelos.

Verificamos los CONCEPTOS con modelos dummy de probabilidad fija (sin red, sin
entrenar de verdad): los extremos del peso (w=1 → model_a, w=0 → model_b), que el
resultado siempre sume 1, que fit() entrene a ambos sub-modelos, y la diferencia de
comportamiento entre el promedio aritmético y el geométrico.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleModel


class _Dummy:
    """Modelo de mentira: devuelve una probabilidad fija; recuerda si lo entrenaron."""

    def __init__(self, probs):
        self.probs = probs
        self.fitted = False

    def fit(self, df):
        self.fitted = True
        return self

    def predict_proba(self, home, away):
        return self.probs


def test_peso_extremo_recupera_cada_modelo():
    a = _Dummy((0.6, 0.2, 0.2))
    b = _Dummy((0.2, 0.3, 0.5))
    # w=1 → solo A.
    ens_a = EnsembleModel(a, b, weight=1.0, method="geometric")
    assert ens_a.predict_proba("X", "Y") == pytest.approx(a.probs, abs=1e-6)
    # w=0 → solo B.
    ens_b = EnsembleModel(a, b, weight=0.0, method="geometric")
    assert ens_b.predict_proba("X", "Y") == pytest.approx(b.probs, abs=1e-6)


def test_probabilidades_suman_uno():
    a = _Dummy((0.6, 0.2, 0.2))
    b = _Dummy((0.1, 0.4, 0.5))
    for method in ("geometric", "arithmetic"):
        ens = EnsembleModel(a, b, weight=0.5, method=method)
        assert sum(ens.predict_proba("X", "Y")) == pytest.approx(1.0)


def test_fit_entrena_ambos_submodelos():
    a, b = _Dummy((0.5, 0.3, 0.2)), _Dummy((0.4, 0.3, 0.3))
    ens = EnsembleModel(a, b)
    df = pd.DataFrame({"home_team": ["X"], "away_team": ["Y"]})
    ens.fit(df)
    assert a.fitted and b.fitted


def test_aritmetico_es_el_promedio_lineal():
    a = _Dummy((0.6, 0.2, 0.2))
    b = _Dummy((0.2, 0.4, 0.4))
    ens = EnsembleModel(a, b, weight=0.5, method="arithmetic")
    esperado = np.array([0.4, 0.3, 0.3])  # promedio simple, ya suma 1
    assert ens.predict_proba("X", "Y") == pytest.approx(tuple(esperado), abs=1e-6)


def test_geometrico_penaliza_el_cero_mas_que_el_aritmetico():
    """
    Si un modelo da ~0 a una clase, el geométrico la arrastra hacia 0 (producto),
    mientras el aritmético la rescata (suma). Esa es la diferencia clave que hace al
    geométrico mejor para log-loss cuando los modelos coinciden, y más arriesgado
    cuando discrepan.
    """
    a = _Dummy((0.98, 0.01, 0.01))   # casi seguro local
    b = _Dummy((0.01, 0.01, 0.98))   # casi seguro visita
    geo = EnsembleModel(a, b, weight=0.5, method="geometric").predict_proba("X", "Y")
    ari = EnsembleModel(a, b, weight=0.5, method="arithmetic").predict_proba("X", "Y")
    # En el geométrico, el empate (donde ambos coinciden bajo) NO es despreciable
    # frente a local/visita, porque local y visita se castigan por el casi-cero del otro.
    assert geo[1] > ari[1]


def test_rechaza_parametros_invalidos():
    a, b = _Dummy((0.5, 0.3, 0.2)), _Dummy((0.4, 0.3, 0.3))
    with pytest.raises(ValueError):
        EnsembleModel(a, b, weight=1.5)
    with pytest.raises(ValueError):
        EnsembleModel(a, b, method="harmonic")
