"""
world_cup.py — Predicción del Mundial 2026 con el Elo de selecciones.

LA JUGADA (y por qué NO usamos API-Football):
    El plan free de API-Football NO da acceso a la temporada 2026 (solo 2022–2024).
    Pero el dataset martj42/international_results YA trae los 72 fixtures del Mundial
    2026 (FIFA World Cup, 48 selecciones) — incluida la columna `neutral` que marca a
    los anfitriones (México/USA/Canadá juegan de local; el resto en cancha neutral).
    Es gratis, ya está en disco, y los nombres de equipo calzan POR CONSTRUCCIÓN con
    el entrenamiento del Elo (mismo dataset) → cero problema de mapeo de nombres.

CÓMO FUNCIONA:
    1. Cargamos TODOS los internacionales (con fixtures futuros incluidos).
    2. Entrenamos el Elo SOLO con los partidos JUGADOS (los que tienen marcador). El
       orden cronológico lo respeta el fit() del Elo.
    3. Para cada fixture del Mundial, predecimos 1X2 con la `neutral` real de esa fila
       (anfitrión → localía; resto → neutral, sin ventaja inventada).

LIMITACIÓN HONESTA: esto predice CADA partido por separado (fase de grupos tal como
está calendarizada). Estimar "quién sale campeón" requiere simular el bracket completo
(Monte Carlo) — es el paso siguiente, construido sobre esta misma base.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_loaders.international_loader import InternationalResultsLoader
from src.models.elo import EloModel

logger = logging.getLogger(__name__)

WORLD_CUP_2026_FROM = "2026-06-01"  # el Mundial 2026 arranca el 11 de junio.


def _split_played_vs_fixtures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parte el dataset en (jugados, fixtures-del-Mundial-2026).
      - jugados: tienen marcador → sirven para ENTRENAR.
      - fixtures: FIFA World Cup desde junio 2026, sin marcador → a PREDECIR.
    """
    played = df.dropna(subset=["home_score", "away_score"])
    fixtures = df[
        (df["date"] >= pd.Timestamp(WORLD_CUP_2026_FROM))
        & (df["tournament"].str.contains("World Cup", case=False, na=False))
        & (df["home_score"].isna())
    ].copy()
    return played, fixtures


def predict_world_cup(model: EloModel | None = None) -> pd.DataFrame:
    """
    Entrena el Elo con los partidos jugados y predice los fixtures del Mundial 2026.

    Devuelve un DataFrame con: date, home_team, away_team, neutral y las tres
    probabilidades p_home / p_draw / p_away, más el favorito (etiqueta H/D/A) y la
    probabilidad de que NO sea empate hacia el favorito (para ordenar por claridad).
    """
    df = InternationalResultsLoader().load(include_unplayed=True)
    played, fixtures = _split_played_vs_fixtures(df)
    if fixtures.empty:
        raise ValueError("No hay fixtures del Mundial 2026 en el dataset (¿se actualizó?).")

    model = model or EloModel()
    model.fit(played)

    filas = []
    for f in fixtures.itertuples(index=False):
        ph, pdraw, pa = model.predict_proba(f.home_team, f.away_team, neutral=bool(f.neutral))
        filas.append({
            "date": f.date,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "neutral": bool(f.neutral),
            "p_home": ph,
            "p_draw": pdraw,
            "p_away": pa,
        })

    out = pd.DataFrame(filas).sort_values("date").reset_index(drop=True)
    # Favorito 1X2 y su probabilidad (para lecturas rápidas).
    probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
    etiquetas = ["H", "D", "A"]
    out["favorito"] = [etiquetas[i] for i in probs.argmax(axis=1)]
    out["p_favorito"] = probs.max(axis=1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.world_cup
# Predice los 72 fixtures del Mundial 2026 y muestra los primeros + el ranking Elo
# de los 48 participantes (el favorito real al título según fuerza acumulada).
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    pd.set_option("display.width", 120)

    pred = predict_world_cup()
    print(f"=== Mundial 2026 · {len(pred)} partidos predichos (Elo de selecciones) ===\n")

    show = pred.head(12).copy()
    for c in ("p_home", "p_draw", "p_away", "p_favorito"):
        show[c] = (show[c] * 100).round(1)
    print(show[["date", "home_team", "away_team", "neutral",
                "p_home", "p_draw", "p_away"]].to_string(index=False))

    # Ranking Elo de los 48 participantes: el termómetro de quién llega lejos.
    elo = EloModel().fit(
        InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"]))
    participantes = sorted(set(pred["home_team"]) | set(pred["away_team"]))
    rank = (elo.ranking()
            .query("team in @participantes")
            .head(12)
            .reset_index(drop=True))
    print(f"\n=== Top-12 favoritos al título por Elo (de los 48 del Mundial) ===")
    print(rank.to_string(index=False))
