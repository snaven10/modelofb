"""
test_calibrate.py — Pruebas de la calibración de hiperparámetros.

Verificamos los CONCEPTOS que hacen honesta a la calibración:
  1. El split de 3 vías (train/val/test) respeta el orden temporal y NO se solapa.
  2. El grid search prueba TODAS las combos (ρ × ξ) y elige la de MENOR log-loss en
     validation — ni una más, ni una menos, y siempre el mínimo real.
  3. model_logloss ignora los partidos sin resultado (fixtures futuros) y da un número
     finito.

Para el grid usamos un DixonColesModel real sobre un mini-dataset; probamos propiedades
robustas (cardinalidad, que el best sea el mínimo), no valores exactos frágiles.

Correlo con:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibrate import grid_search, model_logloss, temporal_split_3


def _mini_df():
    """12 partidos, 4 equipos, fechas crecientes — suficiente para que DC entrene."""
    equipos = ["A", "B", "C", "D"]
    filas = []
    # tres "temporadas" de 4 partidos cada una, todos contra todos rotando.
    for season, year in enumerate(("2021", "2022", "2023")):
        for i in range(4):
            h = equipos[i]
            a = equipos[(i + 1 + season) % 4]
            filas.append((f"{year}-0{i + 1}-01", h, a, (i % 3), ((i + 1) % 3)))
    df = pd.DataFrame(filas, columns=["date", "home_team", "away_team",
                                      "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["H", "A"], default="D")
    return df


def test_split_3_respeta_orden_temporal():
    df = _mini_df()
    train, val, test = temporal_split_3(df, "2022-01-01", "2023-01-01")
    # Cada tramo cae en su ventana, sin solaparse.
    assert train["date"].max() < pd.Timestamp("2022-01-01")
    assert val["date"].min() >= pd.Timestamp("2022-01-01")
    assert val["date"].max() < pd.Timestamp("2023-01-01")
    assert test["date"].min() >= pd.Timestamp("2023-01-01")
    # Ningún partido se pierde ni se duplica.
    assert len(train) + len(val) + len(test) == len(df)


def test_grid_prueba_todas_las_combos_y_elige_el_minimo():
    df = _mini_df()
    train, val, _ = temporal_split_3(df, "2022-01-01", "2023-01-01")
    rhos = [0.0, -0.1]
    xis = [0.0, 0.3, 0.8]
    best, tabla = grid_search(train, val, rhos, xis)

    # Probó EXACTAMENTE el producto cartesiano: 2 × 3 = 6 combos.
    assert len(tabla) == len(rhos) * len(xis)
    combos = {(r["rho"], r["xi"]) for r in tabla}
    assert combos == {(r, x) for r in rhos for x in xis}

    # El best es el de menor val_logloss de TODA la tabla (no uno cualquiera).
    minimo = min(r["val_logloss"] for r in tabla)
    assert best["val_logloss"] == pytest.approx(minimo)
    assert (best["rho"], best["xi"]) in combos


def test_grid_resultados_finitos():
    df = _mini_df()
    train, val, _ = temporal_split_3(df, "2022-01-01", "2023-01-01")
    _, tabla = grid_search(train, val, [0.0], [0.0, 0.5])
    for r in tabla:
        assert np.isfinite(r["val_logloss"])
        assert set(r) == {"rho", "xi", "val_logloss"}


def test_model_logloss_ignora_partidos_sin_resultado():
    """Un fixture futuro (result NaN) no debe contar ni romper el cálculo."""
    df = _mini_df()
    model = __import__("src.models.dixon_coles", fromlist=["DixonColesModel"]) \
        .DixonColesModel(rho=0.0, xi=0.0).fit(df)

    con_futuro = df.copy()
    futuro = con_futuro.iloc[[0]].copy()
    futuro["result"] = pd.NA
    con_futuro = pd.concat([con_futuro, futuro], ignore_index=True)

    ll_base = model_logloss(model, df)
    ll_con_futuro = model_logloss(model, con_futuro)
    # El partido sin resultado se descarta → mismo log-loss que sin él.
    assert ll_con_futuro == pytest.approx(ll_base)
    assert np.isfinite(ll_base)
