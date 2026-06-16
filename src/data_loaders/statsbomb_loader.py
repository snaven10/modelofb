"""
statsbomb_loader.py — Cargador de StatsBomb Open Data.

¿POR QUÉ EMPEZAMOS POR AQUÍ Y NO POR API-Football?
    StatsBomb regala datos históricos de ALTÍSIMA calidad SIN API key, y lo más
    importante para este proyecto: ya traen el **xG (expected goals) calculado**.
    El xG es la métrica que mide la CALIDAD de las oportunidades de gol, no solo
    si entraron o no. Un equipo puede ganar 1-0 con un golazo de chiripa teniendo
    un xG de 0.3, mientras el rival merecía 2.5 goles. Los goles reales son
    ruidosos; el xG es la señal. Por eso uno de nuestros modelos (Poisson con xG)
    va a predecir MEJOR que el que usa goles crudos. Pero eso es Fase 3.

¿QUÉ HACE ESTE MÓDULO?
    Convierte el desorden de eventos de StatsBomb en una tabla LIMPIA, una fila
    por partido, lista para entrenar:
        match_id | date | home_team | away_team | home_score | away_score
                 | home_xg | away_xg | result

CONCEPTO CLAVE — CACHÉ EN DISCO:
    StatsBomb sirve los datos desde GitHub. Bajar los eventos de un mundial entero
    son cientos de llamadas lentas. La PRIMERA vez bajamos y guardamos en disco
    (data/raw/statsbomb/). Las siguientes veces leemos del disco en milisegundos.
    Este MISMO patrón es OBLIGATORIO con API-Football, donde cada llamada cuenta
    contra tu límite diario. Aprendé el patrón aquí, donde equivocarse no cuesta.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from statsbombpy import sb

# statsbombpy es ruidoso (avisa en cada llamada que estás usando open data).
# Le subimos el nivel de log para que no nos llene la terminal de basura.
logging.getLogger("statsbombpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Carpeta donde cacheamos los datasets ya armados. Está en .gitignore: son datos
# reproducibles, no tienen por qué vivir en el repo.
DEFAULT_CACHE_DIR = Path("data/raw/statsbomb")


class StatsBombLoader:
    """
    Acceso a StatsBomb Open Data con caché en disco.

    Las competiciones que nos interesan (el usuario las pidió explícitamente):
        - Mundial masculino 2022  -> competition_id = 43
        - Mundial femenino 2023   -> competition_id = 72

    OJO: NO hardcodeamos los season_id. ¿Por qué? Porque son números internos de
    StatsBomb que cambian entre competiciones y es facilísimo equivocarse y bajar
    el mundial que no era. En vez de eso los RESOLVEMOS dinámicamente por el nombre
    de la temporada (ver resolve_season_id). Verificar > suponer.
    """

    # IDs de competición documentados como constantes legibles. Un número suelto
    # en el código no le dice nada a nadie; una constante con nombre, sí.
    WORLD_CUP_MEN = 43
    WOMENS_WORLD_CUP = 72

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True) -> None:
        """
        Args:
            cache_dir: dónde guardar/leer los datasets cacheados.
            use_cache: si False, ignora el disco y baja todo de nuevo (útil para
                       forzar una actualización o depurar el loader).
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        # parents=True crea la cadena de carpetas; exist_ok=True no falla si ya existe.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Descubrimiento: qué hay disponible
    # ──────────────────────────────────────────────────────────────────────
    def list_competitions(self) -> pd.DataFrame:
        """
        Devuelve TODAS las competiciones del open data.

        Usalo para explorar: corré esto en un notebook y mirá qué competition_id
        y season_id existen antes de pedir partidos. Es tu mapa del territorio.
        """
        return sb.competitions()

    def resolve_season_id(self, competition_id: int, season_name: str) -> int:
        """
        Traduce un nombre de temporada legible (ej. "2022") al season_id interno
        que StatsBomb necesita.

        POR QUÉ existe: pedir partidos requiere (competition_id, season_id). El
        season_id es un número opaco. Resolverlo por nombre evita el clásico bug
        de "bajé datos pero del año equivocado y no me di cuenta".

        Lanza ValueError si no encuentra la temporada: fallar RUIDOSAMENTE es
        mejor que devolver datos silenciosamente incorrectos.
        """
        comps = self.list_competitions()
        match = comps[
            (comps["competition_id"] == competition_id)
            & (comps["season_name"] == season_name)
        ]
        if match.empty:
            disponibles = (
                comps.loc[comps["competition_id"] == competition_id, "season_name"]
                .tolist()
            )
            raise ValueError(
                f"No encontré temporada '{season_name}' para competition_id={competition_id}. "
                f"Temporadas disponibles: {disponibles}"
            )
        return int(match.iloc[0]["season_id"])

    # ──────────────────────────────────────────────────────────────────────
    # Carga cruda
    # ──────────────────────────────────────────────────────────────────────
    def load_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Lista de partidos de una temporada (sin xG todavía, eso viene de eventos)."""
        return sb.matches(competition_id=competition_id, season_id=season_id)

    def _aggregate_match_xg(self, match_id: int) -> tuple[dict[str, float], list[str]]:
        """
        Suma el xG de cada equipo en UN partido, a partir de sus eventos.

        CONCEPTO: el xG de un partido no viene servido; hay que CONSTRUIRLO. Cada
        evento de tipo "Shot" trae su xG individual (shot_statsbomb_xg). El xG del
        equipo es la suma de los xG de todos sus tiros. Eso es el "merecimiento"
        ofensivo del equipo en ese partido.

        Devuelve:
            ({nombre_equipo: xg_total, ...}, [lista de equipos en orden de aparición])
        """
        events = sb.events(match_id=match_id)

        # Si por lo que sea no hay columna de xG en este partido, devolvemos 0.0.
        # Defensa contra datasets incompletos: nunca asumás que la columna existe.
        if "shot_statsbomb_xg" not in events.columns:
            return {}, []

        shots = events[events["type"] == "Shot"]
        # groupby por equipo y sumamos el xG de sus tiros.
        xg_by_team = shots.groupby("team")["shot_statsbomb_xg"].sum().to_dict()
        teams = events["team"].dropna().unique().tolist()
        return xg_by_team, teams

    # ──────────────────────────────────────────────────────────────────────
    # Dataset final, listo para modelar
    # ──────────────────────────────────────────────────────────────────────
    def build_match_dataset(
        self,
        competition_id: int,
        season_name: str,
    ) -> pd.DataFrame:
        """
        Arma el dataset LIMPIO a nivel de partido, con caché en disco.

        Flujo:
            1. ¿Existe el CSV cacheado? -> leelo y listo (rápido).
            2. Si no: resolvé season_id, bajá partidos, agregá xG por partido,
               armá la tabla, guardala en disco para la próxima.

        Columnas de salida:
            match_id, date, home_team, away_team, home_score, away_score,
            home_xg, away_xg, result   (result: 'H' local gana / 'D' empate / 'A' visita gana)

        El 'result' como H/D/A es el estándar en quinielas y en football-data.co.uk;
        lo usamos desde ya para que todos los datasets hablen el mismo idioma.
        """
        cache_file = self.cache_dir / f"matches_comp{competition_id}_{season_name}.csv"

        # --- Paso 1: intento leer de caché ---
        if self.use_cache and cache_file.exists():
            logger.info("Leyendo dataset cacheado: %s", cache_file)
            return pd.read_csv(cache_file, parse_dates=["date"])

        # --- Paso 2: construir desde cero ---
        logger.info("Sin caché. Construyendo dataset desde StatsBomb (esto tarda)...")
        season_id = self.resolve_season_id(competition_id, season_name)
        matches = self.load_matches(competition_id, season_id)

        filas = []
        for _, m in matches.iterrows():
            home = m["home_team"]
            away = m["away_team"]
            xg_by_team, _ = self._aggregate_match_xg(m["match_id"])

            home_score = m["home_score"]
            away_score = m["away_score"]

            # Codificamos el resultado 1X2 (estándar de apuestas).
            if home_score > away_score:
                result = "H"
            elif home_score < away_score:
                result = "A"
            else:
                result = "D"

            filas.append(
                {
                    "match_id": m["match_id"],
                    "date": m["match_date"],
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    # .get() con default 0.0: si un equipo no registró tiros, su xG es 0.
                    "home_xg": round(float(xg_by_team.get(home, 0.0)), 3),
                    "away_xg": round(float(xg_by_team.get(away, 0.0)), 3),
                    "result": result,
                }
            )

        df = pd.DataFrame(filas)
        df["date"] = pd.to_datetime(df["date"])
        # Ordenamos por fecha: es la base del split temporal que haremos al entrenar
        # (entrenar con lo viejo, validar con lo nuevo). Nunca mezclar el futuro.
        df = df.sort_values("date").reset_index(drop=True)

        # --- Guardar en caché para la próxima ---
        df.to_csv(cache_file, index=False)
        logger.info("Dataset guardado en caché: %s (%d partidos)", cache_file, len(df))

        return df


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución directa para probar a mano:  python -m src.data_loaders.statsbomb_loader
# No es la forma definitiva de usarlo (eso será desde scripts/notebooks), pero
# sirve para verificar que el loader respira.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    loader = StatsBombLoader()

    print("Construyendo dataset del Mundial masculino 2022...")
    df = loader.build_match_dataset(
        competition_id=StatsBombLoader.WORLD_CUP_MEN,
        season_name="2022",
    )
    print(df.head(10).to_string(index=False))
    print(f"\nTotal de partidos: {len(df)}")
