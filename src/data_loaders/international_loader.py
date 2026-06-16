"""
international_loader.py — Cargador de resultados internacionales de selecciones.

¿QUÉ ES Y POR QUÉ LO NECESITAMOS?
    El dataset de clubes (football-data.co.uk) tiene VOLUMEN y odds, pero NO tiene
    selecciones: Liverpool no es Inglaterra. Para predecir el Mundial necesitamos
    el nivel REAL de cada PAÍS, y eso solo se aprende de partidos de selecciones.

    Fuente: martj42/international_results (GitHub, público, sin API key).
    ~49,000 partidos de selecciones desde 1872: amistosos, eliminatorias, Copas
    continentales y Mundiales. Argentina sola aparece >1,000 veces. ESE volumen es
    lo que hace que el Elo de selecciones converja a un rating creíble.

DOS COLUMNAS QUE LOS CLUBES NO TIENEN Y AQUÍ SON ORO:
    - `neutral`: ¿se jugó en cancha neutral? En el Mundial casi todo es neutral
      (salvo el anfitrión), así que la "ventaja de localía" NO se aplica igual que
      en una liga. El modelo TIENE que distinguirlo o sobreestima al "local".
    - `tournament`: tipo de partido. Un amistoso vale MENOS que una eliminatoria
      (los equipos prueban jugadores, no van a muerte). El Elo serio pondera por
      importancia del partido; acá dejamos la columna lista para eso.

OJO — PARTIDOS FUTUROS:
    El CSV incluye fixtures ya programados SIN jugar (marcador vacío), incluido el
    Mundial 2026. Para ENTRENAR solo sirven los ya jugados, así que por defecto
    filtramos los que no tienen marcador. Con include_unplayed=True los traés todos
    (útil para ver el calendario del Mundial sin gastar la API key).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw/international")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Mismo esquema base que los demás loaders (clubes, StatsBomb): que todos los
# datasets hablen el mismo idioma. Aquí no hay odds, pero sí columnas propias de
# selecciones (tournament, neutral) que conservamos.
BASE_COLUMNS = {
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_score": "home_score",
    "away_score": "away_score",
    "tournament": "tournament",
    "neutral": "neutral",
}

# Qué cuenta como partido "competitivo" (a muerte) vs amistoso. Heurística simple:
# todo lo que NO sea un amistoso. El Elo puede usar esto para ponderar.
FRIENDLY_LABEL = "Friendly"


class InternationalResultsLoader:
    """Descarga y limpia los resultados internacionales de selecciones, con caché."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Descarga cruda (con caché) — mismo patrón que los otros loaders
    # ──────────────────────────────────────────────────────────────────────
    def _raw_path(self) -> Path:
        return self.cache_dir / "results.csv"

    def download_raw(self) -> pd.DataFrame:
        """Baja el CSV una vez y lo cachea; en corridas siguientes lo lee del disco."""
        cache_file = self._raw_path()

        if self.use_cache and cache_file.exists():
            logger.info("CSV cacheado: %s", cache_file)
            return pd.read_csv(cache_file)

        logger.info("Descargando %s ...", RESULTS_URL)
        resp = requests.get(RESULTS_URL, timeout=60)
        resp.raise_for_status()
        cache_file.write_bytes(resp.content)
        return pd.read_csv(cache_file)

    # ──────────────────────────────────────────────────────────────────────
    # Limpieza
    # ──────────────────────────────────────────────────────────────────────
    def _clean(self, raw: pd.DataFrame, include_unplayed: bool) -> pd.DataFrame:
        df = raw.rename(columns=BASE_COLUMNS).copy()

        # Fechas: este dataset usa ISO (yyyy-mm-dd), formato sano y sin ambigüedad.
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Partidos sin jugar = marcador vacío. Para entrenar los botamos; el Elo no
        # puede aprender de un partido que no ha pasado (sería adivinar el futuro).
        if not include_unplayed:
            df = df.dropna(subset=["home_score", "away_score"])

        df = df.dropna(subset=["home_team", "away_team", "date"]).reset_index(drop=True)

        # Marcadores numéricos. Los dejamos en float por ahora: los futuros quedan
        # como NaN, que en las comparaciones de abajo se evalúan como False (lo que
        # np.select necesita). Casteamos a Int64 nullable AL FINAL.
        home = pd.to_numeric(df["home_score"], errors="coerce")
        away = pd.to_numeric(df["away_score"], errors="coerce")

        # result H/D/A sobre los floats (NaN -> comparación False, no rompe np.select).
        df["result"] = np.select([home > away, home < away], ["H", "A"], default="D")
        # Donde no hay marcador (fixture futuro) el result no aplica: <NA>.
        df.loc[home.isna(), "result"] = pd.NA

        # Ahora sí: marcadores a Int64 nullable (futuros quedan <NA>).
        df["home_score"] = home.astype("Int64")
        df["away_score"] = away.astype("Int64")

        # neutral a booleano limpio.
        df["neutral"] = df["neutral"].astype("boolean")

        # Bandera de importancia: amistoso vs competitivo (para ponderar en el Elo).
        df["is_competitive"] = df["tournament"] != FRIENDLY_LABEL

        ordered = [
            "date", "home_team", "away_team",
            "home_score", "away_score", "result",
            "tournament", "is_competitive", "neutral",
        ]
        return df[ordered].sort_values("date").reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────
    def load(self, include_unplayed: bool = False) -> pd.DataFrame:
        """
        Devuelve todos los internacionales ya limpios y ordenados por fecha.

        include_unplayed=False (default): solo partidos jugados → para ENTRENAR.
        include_unplayed=True: incluye fixtures futuros (p.ej. el calendario del
        Mundial 2026) → para ver el calendario sin tocar la API key.
        """
        raw = self.download_raw()
        return self._clean(raw, include_unplayed=include_unplayed)


# ──────────────────────────────────────────────────────────────────────────────
# Prueba a mano:  python -m src.data_loaders.international_loader
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    loader = InternationalResultsLoader()
    df = loader.load()
    print(df.head(10).to_string(index=False))
    print(f"\nPartidos jugados: {len(df)}")
    print(f"Rango: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Competitivos: {df['is_competitive'].sum()}  |  Amistosos: {(~df['is_competitive']).sum()}")
    print(f"En cancha neutral: {df['neutral'].sum()}")
