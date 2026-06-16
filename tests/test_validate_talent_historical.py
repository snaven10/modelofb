"""
test_validate_talent_historical.py — Pruebas de las funciones puras de la validación
histórica del talento (sin red ni archivos pesados).
"""

import numpy as np
import pytest

from src.models.elo import EloModel
from src.predict.validate_talent_historical import _predict_from_ratings, _sv_lookup


def test_predict_from_ratings_suma_uno_y_favorece_al_mejor():
    elo = EloModel()  # scale 400, home_adv 100, draw heurística
    # rating local muy superior → debe ser claro favorito
    ph, pd_, pa = _predict_from_ratings(elo, rh=1900, ra=1500, neutral=True)
    assert ph + pd_ + pa == pytest.approx(1.0)
    assert ph > pa                       # el mejor (local) gana más probable
    # invertir ratings invierte el favorito
    ph2, _, pa2 = _predict_from_ratings(elo, rh=1500, ra=1900, neutral=True)
    assert pa2 > ph2


def test_predict_neutral_quita_localia():
    elo = EloModel()
    # mismos ratings: en neutral debe ser parejo (p_home ≈ p_away)
    ph_n, _, pa_n = _predict_from_ratings(elo, 1700, 1700, neutral=True)
    assert ph_n == pytest.approx(pa_n, abs=1e-9)
    # con localía el local saca ventaja
    ph_l, _, pa_l = _predict_from_ratings(elo, 1700, 1700, neutral=False)
    assert ph_l > pa_l


def test_sv_lookup_alias_y_fallback_de_year():
    SV = {("Cote d'Ivoire", 2018): 500e6, ("Spain", 2017): 700e6}
    # alias martj42 -> dataset
    assert _sv_lookup(SV, "Ivory Coast", 2018) == 500e6
    # fallback al año previo si falta el exacto
    assert _sv_lookup(SV, "Spain", 2018) == 700e6
    # sin dato -> None
    assert _sv_lookup(SV, "Haiti", 2018) is None
