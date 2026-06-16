"""
test_statsbomb_loader.py — Pruebas del loader de StatsBomb.

PRINCIPIO: un test unitario NO debe tocar la red. Si dependiera de bajar datos
reales de StatsBomb, fallaría cuando se cae internet, tardaría siglos y no estaría
probando TU código sino el de ellos. Por eso MOCKEAMOS statsbombpy: le inyectamos
datos falsos y controlados, y verificamos que la LÓGICA del loader (armar la tabla,
codificar H/D/A, cachear) sea correcta.

Correlo con:  pytest -q
"""

import pandas as pd
import pytest

from src.data_loaders import statsbomb_loader
from src.data_loaders.statsbomb_loader import StatsBombLoader


# ──────────────────────────────────────────────────────────────────────────────
# Datos falsos: un "mundial" mínimo de 2 partidos para probar los 3 resultados.
# ──────────────────────────────────────────────────────────────────────────────
FAKE_COMPETITIONS = pd.DataFrame(
    [
        {"competition_id": 43, "season_id": 999, "season_name": "2022"},
    ]
)

FAKE_MATCHES = pd.DataFrame(
    [
        {
            "match_id": 1,
            "match_date": "2022-11-20",
            "home_team": "Qatar",
            "away_team": "Ecuador",
            "home_score": 0,
            "away_score": 2,  # gana visita -> 'A'
        },
        {
            "match_id": 2,
            "match_date": "2022-11-21",
            "home_team": "England",
            "away_team": "Iran",
            "home_score": 6,
            "away_score": 2,  # gana local -> 'H'
        },
    ]
)

# Eventos falsos por partido: solo lo que el loader mira (type, team, xg de tiros).
FAKE_EVENTS = {
    1: pd.DataFrame(
        [
            {"type": "Shot", "team": "Qatar", "shot_statsbomb_xg": 0.10},
            {"type": "Shot", "team": "Ecuador", "shot_statsbomb_xg": 0.50},
            {"type": "Pass", "team": "Qatar", "shot_statsbomb_xg": None},  # no es tiro: se ignora
        ]
    ),
    2: pd.DataFrame(
        [
            {"type": "Shot", "team": "England", "shot_statsbomb_xg": 1.20},
            {"type": "Shot", "team": "England", "shot_statsbomb_xg": 0.80},
            {"type": "Shot", "team": "Iran", "shot_statsbomb_xg": 0.30},
        ]
    ),
}


@pytest.fixture
def mocked_sb(monkeypatch):
    """
    Reemplaza las funciones de statsbombpy por versiones que devuelven nuestros
    datos falsos. monkeypatch deshace los parches solos al terminar el test.
    """
    monkeypatch.setattr(statsbomb_loader.sb, "competitions", lambda: FAKE_COMPETITIONS)
    monkeypatch.setattr(
        statsbomb_loader.sb,
        "matches",
        lambda competition_id, season_id: FAKE_MATCHES,
    )
    monkeypatch.setattr(
        statsbomb_loader.sb,
        "events",
        lambda match_id: FAKE_EVENTS[match_id],
    )


def test_resolve_season_id_ok(mocked_sb):
    loader = StatsBombLoader(use_cache=False)
    assert loader.resolve_season_id(43, "2022") == 999


def test_resolve_season_id_inexistente(mocked_sb):
    """Pedir una temporada que no existe debe FALLAR ruidosamente, no inventar."""
    loader = StatsBombLoader(use_cache=False)
    with pytest.raises(ValueError):
        loader.resolve_season_id(43, "1990")


def test_build_dataset_estructura_y_resultados(mocked_sb, tmp_path):
    # tmp_path: carpeta temporal de pytest, así no ensuciamos data/raw real.
    loader = StatsBombLoader(cache_dir=tmp_path, use_cache=True)
    df = loader.build_match_dataset(competition_id=43, season_name="2022")

    # Estructura esperada
    assert list(df.columns) == [
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "home_xg", "away_xg", "result",
    ]
    assert len(df) == 2

    # Codificación del resultado 1X2
    qatar = df[df["home_team"] == "Qatar"].iloc[0]
    england = df[df["home_team"] == "England"].iloc[0]
    assert qatar["result"] == "A"   # 0-2, gana visita
    assert england["result"] == "H" # 6-2, gana local

    # Agregación de xG: England tiró dos veces (1.20 + 0.80 = 2.00)
    assert england["home_xg"] == pytest.approx(2.00)
    assert qatar["away_xg"] == pytest.approx(0.50)  # Ecuador, un solo tiro


def test_cache_se_escribe_y_se_relee(mocked_sb, tmp_path):
    """Primera llamada construye y guarda; el CSV debe quedar en disco."""
    loader = StatsBombLoader(cache_dir=tmp_path, use_cache=True)
    loader.build_match_dataset(competition_id=43, season_name="2022")

    cache_file = tmp_path / "matches_comp43_2022.csv"
    assert cache_file.exists(), "El loader debió escribir el CSV de caché"

    # Segunda llamada: debe leer del CSV (mismo resultado).
    df2 = loader.build_match_dataset(competition_id=43, season_name="2022")
    assert len(df2) == 2
