"""
test_international_loader.py — Pruebas del loader de internacionales (martj42).

Igual que los otros: NO tocamos la red. Sembramos un CSV crudo falso (con el
formato REAL del dataset) en la caché y dejamos que el loader lo lea del disco.
Probamos NUESTRA lógica: derivación de result H/D/A, flag competitivo vs amistoso,
booleano neutral, y el filtrado de partidos futuros (sin marcador).

Correlo con:  pytest -q
"""

import pandas as pd

from src.data_loaders.international_loader import InternationalResultsLoader

# Formato real del CSV de martj42. La última fila es un fixture FUTURO (sin
# marcador) — como el Mundial 2026 en el dataset real: debe filtrarse por defecto.
RAW_CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,False\n"
    "1930-07-30,Uruguay,Argentina,4,2,FIFA World Cup,Montevideo,Uruguay,False\n"
    "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,True\n"
    "2026-06-15,Brazil,Spain,,,FIFA World Cup,New York,United States,True\n"
)


def _seed_cache(cache_dir) -> None:
    (cache_dir / "results.csv").write_text(RAW_CSV, encoding="utf-8")


def test_esquema_y_filtra_futuros(tmp_path):
    _seed_cache(tmp_path)
    loader = InternationalResultsLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load()

    assert list(df.columns) == [
        "date", "home_team", "away_team",
        "home_score", "away_score", "result",
        "tournament", "is_competitive", "neutral",
    ]
    # 3 jugados; el fixture futuro Brazil-Spain (sin marcador) se descartó.
    assert len(df) == 3
    assert "Brazil" not in set(df["home_team"])


def test_include_unplayed_trae_el_futuro(tmp_path):
    """Con include_unplayed=True debe aparecer el fixture sin jugar, con result <NA>."""
    _seed_cache(tmp_path)
    loader = InternationalResultsLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load(include_unplayed=True)

    assert len(df) == 4
    futuro = df[df["home_team"] == "Brazil"].iloc[0]
    assert pd.isna(futuro["home_score"])
    assert pd.isna(futuro["result"])


def test_result_hda(tmp_path):
    _seed_cache(tmp_path)
    loader = InternationalResultsLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load()

    assert df[df["home_team"] == "Scotland"].iloc[0]["result"] == "D"  # 0-0
    assert df[df["home_team"] == "Uruguay"].iloc[0]["result"] == "H"   # 4-2
    # Argentina 3-3 France fue empate.
    assert df[df["home_team"] == "Argentina"].iloc[0]["result"] == "D"


def test_competitivo_vs_amistoso(tmp_path):
    _seed_cache(tmp_path)
    loader = InternationalResultsLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load()

    assert df[df["tournament"] == "Friendly"].iloc[0]["is_competitive"] == False  # noqa: E712
    assert df[df["tournament"] == "FIFA World Cup"].iloc[0]["is_competitive"] == True  # noqa: E712


def test_neutral_es_booleano(tmp_path):
    _seed_cache(tmp_path)
    loader = InternationalResultsLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.load()

    # El Argentina-France de Qatar 2022 fue en cancha neutral.
    assert bool(df[df["home_team"] == "Argentina"].iloc[0]["neutral"]) is True
    # El Scotland-England de 1872 NO.
    assert bool(df[df["home_team"] == "Scotland"].iloc[0]["neutral"]) is False
