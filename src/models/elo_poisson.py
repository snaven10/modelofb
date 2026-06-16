"""
elo_poisson.py — Modelo de goles ajustado por fuerza del rival, vía Elo.

POR QUÉ EXISTE (el bug que corrige):
    El PoissonModel cuenta goles en ABSOLUTO (promedios de marcados/recibidos). Eso
    sobreestima a equipos que golean y no encajan contra rivales DÉBILES de su
    confederación (Irán, Egipto, Nueva Zelanda): el modelo confunde "rival flojo" con
    "yo soy fuerte". En el Monte Carlo, Irán salía 3º al título — irreal.

    El Elo NO cae en eso: la sorpresa de cada partido se pondera por el rating del
    rival, así que ganarle a un flojo casi no sube tu rating. Por eso el ranking Elo
    (España, Argentina, Francia, Brasil...) es el creíble.

LA IDEA:
    Derivar los goles esperados (λ) de la DIFERENCIA DE ELO, no de promedios crudos.
        supremacía = (rating_local_con_localía − rating_visita) · b
        λ_local  = total_base/2 + supremacía/2
        λ_visita = total_base/2 − supremacía/2
    `b` (goles por punto de Elo) y `total_base` (goles totales por partido) se calibran
    DESDE LOS DATOS, no a ojo.

INTERFAZ: expone expected_goals(a, b) con `a` como local (igual que PoissonModel), así
el simulador lo usa sin cambios — el promediado de orientaciones para cancha neutral
sigue funcionando. También predict_proba(a, b) para usarlo como modelo 1X2.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.elo import EloModel

logger = logging.getLogger(__name__)


class EloGoalModel:
    """Goles esperados derivados de la fuerza Elo (ajustada por rival)."""

    def __init__(self, max_goals: int = 10, elo: EloModel | None = None) -> None:
        self.max_goals = max_goals
        self.elo = elo or EloModel()
        self.total_base: float = 2.6      # goles totales por partido (se recalibra en fit)
        self.b: float = 0.001             # goles por punto de Elo (se recalibra en fit)

    def fit(self, df: pd.DataFrame) -> "EloGoalModel":
        """
        Entrena el Elo y calibra (total_base, b) DESDE LOS DATOS:
          - total_base = media de goles totales por partido.
          - b = pendiente de regresar (dif. de goles) sobre (dif. de rating + localía).
            Usamos ratings finales como proxy de fuerza (suficiente para fijar la ESCALA;
            no necesitamos el rating exacto pre-partido para calibrar una pendiente).
        """
        self.elo.fit(df)
        data = df.dropna(subset=["home_score", "away_score"]).copy()
        data["home_score"] = data["home_score"].astype(int)
        data["away_score"] = data["away_score"].astype(int)

        self.total_base = float((data["home_score"] + data["away_score"]).mean())

        # Regresión sin intercepto: goal_diff ≈ b · rating_diff_con_localía.
        ratings = self.elo.ratings
        ha = self.elo.home_advantage
        rd, gd = [], []
        for r in data.itertuples(index=False):
            if r.home_team in ratings and r.away_team in ratings:
                neutral = bool(getattr(r, "neutral", False))
                adv = 0.0 if neutral else ha
                rd.append(ratings[r.home_team] + adv - ratings[r.away_team])
                gd.append(r.home_score - r.away_score)
        rd_arr, gd_arr = np.array(rd), np.array(gd)
        # pendiente por mínimos cuadrados sin intercepto: b = Σ(x·y)/Σ(x²)
        denom = float((rd_arr ** 2).sum())
        self.b = float((rd_arr * gd_arr).sum() / denom) if denom > 0 else 0.001

        logger.info("EloGoalModel · total_base=%.2f goles · b=%.5f goles/punto-Elo",
                    self.total_base, self.b)
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """λ_local y λ_visita. `home` lleva la ventaja de localía del Elo."""
        rh = self.elo.get_rating(home) + self.elo.home_advantage
        ra = self.elo.get_rating(away)
        supremacy = (rh - ra) * self.b
        lam_home = max(self.total_base / 2.0 + supremacy / 2.0, 0.05)
        lam_away = max(self.total_base / 2.0 - supremacy / 2.0, 0.05)
        return lam_home, lam_away

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam_home, lam_away = self.expected_goals(home, away)
        ks = np.arange(self.max_goals + 1)
        return np.outer(poisson.pmf(ks, lam_home), poisson.pmf(ks, lam_away))

    def predict_proba(self, home: str, away: str) -> tuple[float, float, float]:
        m = self.score_matrix(home, away)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.models.elo_poisson
# Compara λ del EloGoalModel vs el Poisson crudo para los casos sesgados.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    from src.data_loaders.international_loader import InternationalResultsLoader
    from src.models.poisson import PoissonModel

    df = InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"])
    moderno = df[df["date"] >= "2010-01-01"]

    egm = EloGoalModel().fit(moderno)
    pm = PoissonModel(max_goals=10).fit(moderno)

    print(f"total_base={egm.total_base:.2f}  b={egm.b:.5f}\n")
    print("Partido (neutral)          | λ Elo-goles      | λ Poisson crudo")
    casos = [("Spain", "Cape Verde"), ("Iran", "Spain"), ("Iran", "Brazil"),
             ("New Zealand", "France"), ("Argentina", "Jordan"), ("Brazil", "Iran")]
    for a, b in casos:
        # neutral = promedio de las dos orientaciones (como hace el simulador)
        e1 = egm.expected_goals(a, b); e2 = egm.expected_goals(b, a)
        le = ((e1[0] + e2[1]) / 2, (e1[1] + e2[0]) / 2)
        p1 = pm.expected_goals(a, b); p2 = pm.expected_goals(b, a)
        lp = ((p1[0] + p2[1]) / 2, (p1[1] + p2[0]) / 2)
        print(f"{a:12}-{b:12} | {le[0]:.2f} - {le[1]:.2f}      | {lp[0]:.2f} - {lp[1]:.2f}")
