"""
test_football_data_loader.py — Pruebas del loader de football-data.co.uk.

Igual que con StatsBomb: NO tocamos la red. Pre-sembramos un CSV crudo falso en
la carpeta de caché (con el formato REAL de football-data) y dejamos que el loader
lo lea desde disco. Así probamos NUESTRA lógica: parseo de fechas dd/mm/yy,
de-vigging de las cuotas, resolución del trío de odds y el esquema de salida.

Correlo con:  pytest -q
"""

import pandas as pd
import pytest

from src.data_loaders.football_data_loader import FootballDataLoader

# CSV crudo tal cual lo sirve football-data: cabecera con MUCHAS columnas, pero
# acá ponemos solo las que el loader mira. Fechas en dd/mm/yy (formato real).
# AvgCH/AvgCD/AvgCA = promedio del mercado al cierre (la opción preferida).
RAW_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgCH,AvgCD,AvgCA\n"
    "E0,11/08/23,Arsenal,Forest,2,1,H,1.50,4.20,6.50\n"
    "E0,12/08/23,Chelsea,Liverpool,1,1,D,2.40,3.40,3.00\n"
    "E0,13/08/23,Burnley,Man City,0,3,A,8.00,5.00,1.40\n"
    ",,,,,,,,,\n"  # fila basura al final, como en los CSV reales: debe descartarse
)


def _seed_cache(cache_dir, league="E0", season="2324") -> None:
    """Escribe el CSV crudo en la caché para que el loader lo lea sin red."""
    (cache_dir / f"{league}_{season}.csv").write_text(RAW_CSV, encoding="latin-1")


def test_esquema_y_resultados(tmp_path):
    _seed_cache(tmp_path)
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="2324")

    assert list(df.columns) == [
        "league", "date", "home_team", "away_team",
        "home_score", "away_score", "result",
        "odds_h", "odds_d", "odds_a",
        "p_market_h", "p_market_d", "p_market_a",
    ]
    # 3 partidos válidos; la fila basura se descartó.
    assert len(df) == 3
    assert set(df["result"]) == {"H", "D", "A"}


def test_fechas_dayfirst(tmp_path):
    """11/08/23 debe ser 11 de AGOSTO, no 8 de noviembre."""
    _seed_cache(tmp_path)
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="2324")

    primera = df.iloc[0]["date"]
    assert primera == pd.Timestamp("2023-08-11")


def test_devigging_suma_uno(tmp_path):
    """Las probabilidades de mercado, ya sin margen, deben sumar 1 en cada fila."""
    _seed_cache(tmp_path)
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="2324")

    sumas = df[["p_market_h", "p_market_d", "p_market_a"]].sum(axis=1)
    for s in sumas:
        assert s == pytest.approx(1.0, abs=1e-3)


def test_devigging_valor_concreto(tmp_path):
    """
    Verificación numérica del de-vigging en el primer partido (cuotas 1.50/4.20/6.50):
        crudas = 1/1.5=0.6667, 1/4.2=0.2381, 1/6.5=0.1538  -> overround=1.0586
        p_local = 0.6667 / 1.0586 = 0.6298
    """
    _seed_cache(tmp_path)
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="2324")

    fila = df[df["home_team"] == "Arsenal"].iloc[0]
    assert fila["p_market_h"] == pytest.approx(0.6298, abs=1e-3)


def test_fallback_odds_b365(tmp_path):
    """Si no hay AvgC*, debe caer a B365* (último recurso) sin romperse."""
    raw = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "E0,11/08/23,Arsenal,Forest,2,1,H,1.50,4.20,6.50\n"
    )
    (tmp_path / "E0_9999.csv").write_text(raw, encoding="latin-1")
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="9999")

    assert df.iloc[0]["odds_h"] == 1.50
    assert df.iloc[0]["p_market_h"] == pytest.approx(0.6298, abs=1e-3)


def test_sin_odds_no_rompe(tmp_path):
    """Un CSV sin ninguna columna de odds debe cargar con NaN, no explotar."""
    raw = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,11/08/23,Arsenal,Forest,2,1,H\n"
    )
    (tmp_path / "E0_0001.csv").write_text(raw, encoding="latin-1")
    loader = FootballDataLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load_season(league="E0", season="0001")

    assert df.iloc[0]["odds_h"] != df.iloc[0]["odds_h"]  # NaN != NaN
    assert len(df) == 1
