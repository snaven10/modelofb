"""
poisson.py — Modelo Poisson de goles para fútbol.

El salto respecto al Elo: el Elo dice "quién gana"; Poisson modela CUÁNTOS GOLES
mete cada equipo, y de ahí sale TODO (1X2, marcadores exactos, over/under, gana
por 2+...). Implementa exactamente lo que discutimos antes de codear:

  1. λ (goles esperados) vía ATAQUE × DEFENSA, relativos a la media:
         λ_local  = media_local  · ataque_local  · defensa_visita
         λ_visita = media_visita · ataque_visita · defensa_local
     La localía sale GRATIS: en los datos el local marca más que el visitante, así
     que media_local > media_visita ya la captura. Sin bonus inventado (≠ Elo).

  2. Distribución de Poisson para los goles de cada equipo:
         P(k goles) = λ^k · e^(−λ) / k!

  3. MATRIZ DE MARCADORES (asumiendo independencia entre los dos equipos):
         P(marcador i–j) = P(local marca i) · P(visita marca j)
     Sumando celdas: P(gana local) = triángulo inferior, P(empate) = diagonal,
     P(gana visita) = triángulo superior. El empate emerge SOLO (≠ Elo).

MÉTODO DE ESTIMACIÓN: ratios de tasas (el Poisson "didáctico", sin optimización
iterativa). Transparente y suficiente como primer modelo de goles. La mejora
Dixon-Coles (corrección de marcadores bajos + decay temporal) queda para después.

OJO — UNA LIGA A LA VEZ: las fuerzas ataque/defensa son RELATIVAS a la media de su
competición. Mezclar ligas distintas (Premier con LaLiga) distorsiona, porque un
equipo solo enfrenta a los de su liga. Entrená un PoissonModel por liga.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class PoissonModel:
    """Modelo de goles ataque/defensa con distribución de Poisson."""

    def __init__(self, max_goals: int = 10) -> None:
        """
        max_goals: hasta cuántos goles por equipo consideramos en la matriz de
        marcadores. 10 cubre >99.99% de la probabilidad real; más es desperdicio.
        """
        self.max_goals = max_goals

        # Se llenan en fit():
        self.home_mean: float = 0.0   # media de goles del LOCAL por partido
        self.away_mean: float = 0.0   # media de goles del VISITANTE por partido
        # fuerzas por equipo (1.0 = exactamente la media de la liga):
        self.attack_home: dict[str, float] = {}
        self.defense_home: dict[str, float] = {}
        self.attack_away: dict[str, float] = {}
        self.defense_away: dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Entrenamiento
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        """
        Estima las fuerzas de cada equipo a partir de partidos jugados.

        Espera el esquema común: home_team, away_team, home_score, away_score.
        Separamos ataque/defensa de LOCAL y de VISITA porque un equipo no rinde
        igual en casa que fuera, y esa diferencia es justo lo que da la localía.
        """
        data = df.dropna(subset=["home_score", "away_score"]).copy()
        data["home_score"] = data["home_score"].astype(int)
        data["away_score"] = data["away_score"].astype(int)

        # Medias globales: el "tono" goleador de la liga, y la ventaja de localía.
        self.home_mean = data["home_score"].mean()
        self.away_mean = data["away_score"].mean()

        # Estadísticas por equipo, separando local y visita.
        # Marcados/recibidos PROMEDIO, divididos por la media correspondiente = fuerza.
        as_home = data.groupby("home_team").agg(
            gf=("home_score", "mean"), ga=("away_score", "mean")
        )
        as_away = data.groupby("away_team").agg(
            gf=("away_score", "mean"), ga=("home_score", "mean")
        )

        # ataque = cuánto marca vs la media de marcar en ese rol.
        # defensa = cuánto le marcan vs la media de encajar en ese rol.
        #   defensa < 1  -> encaja MENOS que el promedio = buena defensa.
        self.attack_home = (as_home["gf"] / self.home_mean).to_dict()
        self.defense_home = (as_home["ga"] / self.away_mean).to_dict()
        self.attack_away = (as_away["gf"] / self.away_mean).to_dict()
        self.defense_away = (as_away["ga"] / self.home_mean).to_dict()

        logger.info("Poisson entrenado · %d partidos · %d equipos · media local=%.2f visita=%.2f",
                    len(data), len(self.attack_home), self.home_mean, self.away_mean)
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Goles esperados (λ)
    # ──────────────────────────────────────────────────────────────────────
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """
        λ_local y λ_visita para un cruce. Equipo desconocido = fuerza neutral (1.0),
        que equivale a "un equipo promedio de la liga".
        """
        lam_home = (self.home_mean
                    * self.attack_home.get(home, 1.0)
                    * self.defense_away.get(away, 1.0))
        lam_away = (self.away_mean
                    * self.attack_away.get(away, 1.0)
                    * self.defense_home.get(home, 1.0))
        return lam_home, lam_away

    # ──────────────────────────────────────────────────────────────────────
    # Matriz de marcadores
    # ──────────────────────────────────────────────────────────────────────
    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """
        Matriz (max_goals+1 × max_goals+1) con P(marcador i–j).
        Fila i = goles del local, columna j = goles de la visita.
        """
        lam_home, lam_away = self.expected_goals(home, away)
        ks = np.arange(self.max_goals + 1)

        # Vector de P(k goles) para cada equipo según su λ (la pmf de Poisson).
        p_home = poisson.pmf(ks, lam_home)
        p_away = poisson.pmf(ks, lam_away)

        # Producto externo = matriz conjunta (asume independencia entre los equipos).
        return np.outer(p_home, p_away)

    # ──────────────────────────────────────────────────────────────────────
    # Probabilidades derivadas
    # ──────────────────────────────────────────────────────────────────────
    def predict_proba(self, home: str, away: str) -> tuple[float, float, float]:
        """(p_local, p_empate, p_visita) sumando las regiones de la matriz."""
        m = self.score_matrix(home, away)
        p_home = np.tril(m, -1).sum()   # i > j  (triángulo inferior)
        p_draw = np.trace(m)            # i = j  (diagonal): el empate emerge solo
        p_away = np.triu(m, 1).sum()    # i < j  (triángulo superior)

        total = p_home + p_draw + p_away  # ~1, normalizamos por el truncado en max_goals
        return p_home / total, p_draw / total, p_away / total

    def most_likely_score(self, home: str, away: str) -> tuple[int, int, float]:
        """El marcador exacto más probable y su probabilidad."""
        m = self.score_matrix(home, away)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        return int(i), int(j), float(m[i, j])

    def prob_over(self, home: str, away: str, line: float = 2.5) -> float:
        """P(total de goles > line). Default 2.5 = el over/under clásico."""
        m = self.score_matrix(home, away)
        idx = np.add.outer(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        return float(m[idx > line].sum())


# ──────────────────────────────────────────────────────────────────────────────
# Demo a mano:  python -m src.models.poisson
# Entrena con la Premier League real y predice un par de cruces.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_parquet("data/processed/matches_top5.parquet")
    premier = df[df["league"] == "E0"]  # una liga a la vez
    model = PoissonModel().fit(premier)

    for home, away in [("Man City", "Sheffield United"), ("Arsenal", "Liverpool")]:
        lh, la = model.expected_goals(home, away)
        ph, pd_, pa = model.predict_proba(home, away)
        i, j, p = model.most_likely_score(home, away)
        over = model.prob_over(home, away, 2.5)
        print(f"\n=== {home} vs {away} ===")
        print(f"Goles esperados: {home} {lh:.2f} - {la:.2f} {away}")
        print(f"1X2: local {ph:.1%} | empate {pd_:.1%} | visita {pa:.1%}")
        print(f"Marcador más probable: {i}-{j} ({p:.1%})  |  Over 2.5: {over:.1%}")
