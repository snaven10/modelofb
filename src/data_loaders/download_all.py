"""
download_all.py — Descarga masiva de football-data.co.uk.

¿POR QUÉ ESTE SCRIPT?
    El FootballDataLoader baja UNA (liga, temporada) a la vez. Para entrenar un
    modelo con volumen de verdad necesitamos MUCHOS partidos: varias ligas grandes
    a lo largo de varias temporadas. Este script orquesta esa descarga, junta todo
    en un solo dataset limpio y lo guarda en data/processed.

    Es idempotente: el loader cachea cada CSV crudo en disco, así que correrlo de
    nuevo NO vuelve a pegarle al servidor (salvo temporadas nuevas que falten).

QUÉ BAJA POR DEFECTO:
    - Las top-5 ligas europeas (donde hay volumen Y odds de cierre de calidad).
    - Las últimas 7 temporadas. OJO: las odds de CIERRE promedio (AvgC*) solo
      existen desde ~2019/20; por eso no bajamos más viejo (sería ruido sin
      benchmark de mercado para calibrar).

ROBUSTEZ:
    Si una (liga, temporada) no existe todavía (404) la salta y lo reporta, en vez
    de explotar y botar todo lo demás.

Uso:  python -m src.data_loaders.download_all
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.data_loaders.football_data_loader import FootballDataLoader

logger = logging.getLogger(__name__)

# Las top-5: código de football-data -> nombre legible (solo para el reporte).
LEAGUES = {
    "E0": "Premier League (ING)",
    "SP1": "LaLiga (ESP)",
    "I1": "Serie A (ITA)",
    "D1": "Bundesliga (ALE)",
    "F1": "Ligue 1 (FRA)",
}

# Códigos de 4 dígitos: "1920" = temporada 2019/20. Desde aquí hay odds de cierre.
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "matches_top5.parquet"


def download_all(
    leagues: dict[str, str] = LEAGUES,
    seasons: list[str] = SEASONS,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Baja todas las combinaciones (liga, temporada), las junta y las guarda."""
    loader = FootballDataLoader()
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for league, nombre in leagues.items():
        for season in seasons:
            try:
                df = loader.load_season(league=league, season=season)
                frames.append(df)
                logger.info("OK  %-22s %s -> %4d partidos", nombre, season, len(df))
            except requests.HTTPError as exc:
                # 404 = esa temporada aún no está publicada. La saltamos sin drama.
                skipped.append(f"{league}/{season} ({exc.response.status_code})")
                logger.warning("SKIP %s %s: %s", league, season, exc)

    if not frames:
        raise RuntimeError("No se bajó NADA. ¿Red caída o todo dio 404?")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False)

    # ── Reporte final, para saber EXACTAMENTE qué quedó en el dataset ──
    logger.info("=" * 60)
    logger.info("Dataset combinado: %d partidos", len(combined))
    logger.info("Guardado en: %s", output_file)
    if skipped:
        logger.info("Saltadas (no publicadas aún): %s", ", ".join(skipped))
    con_odds = combined["p_market_h"].notna().sum()
    logger.info("Partidos con odds de mercado: %d (%.0f%%)",
                con_odds, 100 * con_odds / len(combined))
    logger.info("Rango de fechas: %s -> %s",
                combined["date"].min().date(), combined["date"].max().date())

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = download_all()
    print("\n--- Primeras filas ---")
    print(df.head().to_string(index=False))
    print("\n--- Partidos por liga ---")
    print(df["league"].value_counts().to_string())
