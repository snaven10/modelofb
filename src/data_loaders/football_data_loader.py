"""
football_data_loader.py — Cargador de football-data.co.uk.

¿QUÉ ES football-data.co.uk?
    Un archivo público de CSVs con resultados históricos de las grandes ligas
    europeas, temporada por temporada, e incluyendo algo ORO para nosotros: las
    **odds (cuotas) de las casas de apuestas**, incluidas las de CIERRE (las que
    estaban justo antes de empezar el partido). Sin API key, sin límites.

¿PARA QUÉ LO USAMOS?
    1. ENTRENAR: miles de partidos reales (mucho más volumen que un mundial).
    2. CALIBRAR / BACKTEST: las odds de cierre son el "consenso del mercado", el
       mejor predictor público que existe. Si nuestro modelo no le gana (o al
       menos se le acerca) al mercado, sabemos que aún está flojo.

EL CONCEPTO CLAVE — DE LAS CUOTAS A PROBABILIDADES (de-vigging):
    Una cuota decimal de 2.00 NO es 50%. La probabilidad IMPLÍCITA cruda es 1/cuota,
    PERO las casas meten un margen: la suma de las tres probabilidades crudas da
    MÁS de 1 (el "overround"). Para recuperar la probabilidad que el mercado
    realmente estima, normalizamos para que las tres sumen 1:

        p_cruda_i = 1 / cuota_i
        overround = suma de las p_cruda
        p_mercado_i = p_cruda_i / overround     <- esto SÍ suma 1

    Esa p_mercado es nuestro benchmark de calibración. (Hay métodos más finos como
    Shin o power, pero el normalizado básico es el estándar para empezar.)

URL de las CSV:
    https://www.football-data.co.uk/mmz4281/<TEMPORADA>/<LIGA>.csv
    - TEMPORADA: código de 4 dígitos, ej. "2324" = temporada 2023/24.
    - LIGA: ej. E0 = Premier, SP1 = LaLiga, I1 = Serie A, D1 = Bundesliga, F1 = Ligue 1.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw/football_data")
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Las columnas que nos importan del CSV crudo, y cómo las renombramos a nuestro
# esquema limpio (el MISMO que usa el loader de StatsBomb: así todos los datasets
# hablan el mismo idioma y los modelos no tienen que adivinar nombres).
#   FTHG/FTAG = Full Time Home/Away Goals.  FTR = Full Time Result (H/D/A).
BASE_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_score",
    "FTAG": "away_score",
    "FTR": "result",
}

# Tríos de columnas de odds, en orden de PREFERENCIA. Buscamos el primero que el
# CSV tenga disponible (la disponibilidad cambia entre temporadas viejas y nuevas):
#   1. AvgC*  = promedio del MERCADO al CIERRE  (lo mejor: consenso, no una sola casa)
#   2. B365C* = Bet365 al cierre                 (si no hay promedio de cierre)
#   3. B365*  = Bet365 (no necesariamente cierre)(último recurso)
# El sufijo H/D/A = Home / Draw / Away.
ODDS_PREFERENCES = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
]


class FootballDataLoader:
    """Descarga y limpia CSVs de football-data.co.uk, con caché en disco."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Descarga cruda (con caché)
    # ──────────────────────────────────────────────────────────────────────
    def _raw_csv_path(self, league: str, season: str) -> Path:
        return self.cache_dir / f"{league}_{season}.csv"

    def download_raw(self, league: str, season: str) -> pd.DataFrame:
        """
        Devuelve el CSV CRUDO de (liga, temporada). Si está cacheado lo lee del
        disco; si no, lo baja UNA vez y lo guarda.

        Mismo patrón de caché que StatsBomb: bajamos una vez, reusamos siempre.
        Acá importa de verdad porque cada corrida volvería a pegarle al servidor.
        """
        cache_file = self._raw_csv_path(league, season)

        if self.use_cache and cache_file.exists():
            logger.info("CSV cacheado: %s", cache_file)
            # encoding latin-1: football-data trae nombres con acentos/caracteres
            # raros y NO es UTF-8. Si usás el default, te explota en algunos archivos.
            return pd.read_csv(cache_file, encoding="latin-1")

        url = f"{BASE_URL}/{season}/{league}.csv"
        logger.info("Descargando %s ...", url)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()  # falla ruidoso si el servidor devolvió 404/500

        # Guardamos el texto crudo tal cual para la caché.
        cache_file.write_bytes(resp.content)
        # Lo parseamos desde el contenido en memoria (mismo encoding latin-1).
        return pd.read_csv(StringIO(resp.content.decode("latin-1")))

    # ──────────────────────────────────────────────────────────────────────
    # Resolución de columnas de odds
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pick_odds_columns(columns: list[str]) -> tuple[str, str, str] | None:
        """
        Elige el mejor trío de columnas de odds disponible según ODDS_PREFERENCES.
        Devuelve None si el CSV no trae odds (algunas temporadas muy viejas).
        """
        for trio in ODDS_PREFERENCES:
            if all(col in columns for col in trio):
                return trio
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Limpieza + probabilidades de mercado
    # ──────────────────────────────────────────────────────────────────────
    def _clean(self, raw: pd.DataFrame, league: str) -> pd.DataFrame:
        """Convierte un CSV crudo en nuestra tabla limpia con probabilidades de mercado."""
        # 1. Quedarnos solo con las columnas base que existan (defensa por si falta alguna).
        present_base = {src: dst for src, dst in BASE_COLUMNS.items() if src in raw.columns}
        df = raw[list(present_base.keys())].rename(columns=present_base).copy()

        # 2. Fechas: football-data usa dd/mm/yy y dd/mm/yyyy MEZCLADOS entre temporadas.
        #    - dayfirst=True es OBLIGATORIO: sin esto, pandas lee 03/04 como "abril 3"
        #      en vez de "3 de abril" y te corrompe TODAS las fechas EN SILENCIO.
        #    - format="mixed" le dice explícitamente "los formatos varían, parsealos
        #      fila por fila": evita el warning de inferencia y deja la intención clara.
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")

        # 3. Botar filas basura (CSVs traen renglones vacíos al final).
        df = df.dropna(subset=["home_team", "away_team", "date"]).reset_index(drop=True)

        # 4. Etiqueta de liga, para cuando concatenemos varias.
        df["league"] = league

        # 5. Odds + de-vigging
        odds_cols = self._pick_odds_columns(list(raw.columns))
        if odds_cols is None:
            # Sin odds: dejamos las columnas en NaN para no romper el esquema común.
            logger.warning("La liga %s no traía columnas de odds reconocidas.", league)
            df[["odds_h", "odds_d", "odds_a"]] = float("nan")
            df[["p_market_h", "p_market_d", "p_market_a"]] = float("nan")
        else:
            ch, cd, ca = odds_cols
            # Alineamos las odds a las filas que sobrevivieron al dropna.
            df["odds_h"] = pd.to_numeric(raw.loc[df.index, ch], errors="coerce").values
            df["odds_d"] = pd.to_numeric(raw.loc[df.index, cd], errors="coerce").values
            df["odds_a"] = pd.to_numeric(raw.loc[df.index, ca], errors="coerce").values
            self._add_market_probabilities(df)

        # 6. Tipos limpios para los goles.
        for col in ("home_score", "away_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # 7. Esquema de salida ordenado y consistente.
        ordered = [
            "league", "date", "home_team", "away_team",
            "home_score", "away_score", "result",
            "odds_h", "odds_d", "odds_a",
            "p_market_h", "p_market_d", "p_market_a",
        ]
        return df[[c for c in ordered if c in df.columns]].sort_values("date").reset_index(drop=True)

    @staticmethod
    def _add_market_probabilities(df: pd.DataFrame) -> None:
        """
        Añade p_market_h/d/a: probabilidades del mercado SIN el margen de la casa.
        Modifica df in-place. Ver la explicación del de-vigging arriba del módulo.
        """
        # Probabilidad cruda = 1 / cuota.
        raw_h = 1.0 / df["odds_h"]
        raw_d = 1.0 / df["odds_d"]
        raw_a = 1.0 / df["odds_a"]

        # El overround es la suma de las crudas (debería ser > 1: ahí está el margen).
        overround = raw_h + raw_d + raw_a

        # Normalizamos: ahora las tres suman exactamente 1 -> probabilidades reales.
        df["p_market_h"] = (raw_h / overround).round(4)
        df["p_market_d"] = (raw_d / overround).round(4)
        df["p_market_a"] = (raw_a / overround).round(4)

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────
    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Una sola (liga, temporada), ya limpia."""
        raw = self.download_raw(league, season)
        return self._clean(raw, league)

    def load_seasons(self, league: str, seasons: list[str]) -> pd.DataFrame:
        """
        Varias temporadas de una liga, concatenadas y ordenadas por fecha.

        Ordenar por fecha es la base del SPLIT TEMPORAL que haremos al entrenar:
        entrenás con las temporadas viejas y validás con las nuevas. Jamás al revés
        (sería "ver el futuro" = trampa que infla las métricas y miente).
        """
        frames = [self.load_season(league, s) for s in seasons]
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Prueba a mano:  python -m src.data_loaders.football_data_loader
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    loader = FootballDataLoader()
    # Premier League (E0), temporada 2023/24.
    df = loader.load_season(league="E0", season="2324")
    print(df.head(10).to_string(index=False))
    print(f"\nPartidos: {len(df)}")
    # Chequeo rápido: las probabilidades de mercado deben sumar ~1 por fila.
    suma = (df[["p_market_h", "p_market_d", "p_market_a"]].sum(axis=1)).head()
    print("\nSuma de p_market por fila (debe ser ~1.0):")
    print(suma.to_string(index=False))
