"""
dixon_coles.py — Modelo Dixon-Coles (1997): el Poisson, parchado.

Extiende PoissonModel arreglando sus dos defectos conocidos:

  1. DEPENDENCIA EN MARCADORES BAJOS (parámetro ρ / rho):
     El Poisson asume goles independientes; en realidad 0-0 y 1-1 ocurren MÁS de lo
     que predice. La corrección τ ajusta SOLO las 4 celdas bajas de la matriz:
         τ(0,0)=1−λμρ   τ(0,1)=1+λρ   τ(1,0)=1+μρ   τ(1,1)=1−ρ
     Con ρ<0 (~−0.1) infla los empates bajos y desinfla 1-0 / 0-1.

  2. DECAY TEMPORAL (parámetro ξ / xi):
     Un partido viejo no debería pesar igual que uno reciente (los equipos cambian).
     Peso de cada partido = exp(−ξ·Δt_años). ξ=0 → todo pesa igual (= Poisson normal);
     ξ>0 → el pasado se desvanece.

Hereda de PoissonModel: reusa ataque/defensa, expected_goals, predict_proba, etc.
Solo cambia el fit (ahora ponderado) y la score_matrix (ahora con τ).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.poisson import PoissonModel

logger = logging.getLogger(__name__)


class DixonColesModel(PoissonModel):
    """Poisson + corrección de marcadores bajos (ρ) + decay temporal (ξ)."""

    def __init__(self, max_goals: int = 10, rho: float = -0.10, xi: float = 0.0) -> None:
        """
        rho  dependencia de marcadores bajos (negativo ~−0.1). 0 = sin corrección.
        xi   decay temporal por AÑO. 0 = sin decay. Half-life = ln(2)/xi años.
        """
        super().__init__(max_goals)
        self.rho = rho
        self.xi = xi

    # ──────────────────────────────────────────────────────────────────────
    # fit ponderado por recencia
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """
        Igual que el Poisson, pero las medias de ataque/defensa son PROMEDIOS
        PONDERADOS: los partidos recientes pesan más (peso = exp(−ξ·Δt_años)).
        Con ξ=0 todos los pesos valen 1 y recuperás el Poisson normal.
        """
        data = df.dropna(subset=["home_score", "away_score"]).copy()
        data["home_score"] = data["home_score"].astype(int)
        data["away_score"] = data["away_score"].astype(int)

        # Peso temporal de cada partido respecto al MÁS reciente del train.
        if self.xi > 0 and "date" in data.columns:
            edad_anios = (data["date"].max() - data["date"]).dt.days / 365.25
            data["w"] = np.exp(-self.xi * edad_anios)
        else:
            data["w"] = 1.0

        # Medias globales ponderadas (tono goleador + localía).
        self.home_mean = self._wmean(data["home_score"], data["w"])
        self.away_mean = self._wmean(data["away_score"], data["w"])

        # Fuerzas por equipo: promedio ponderado de goles a favor / en contra, por rol.
        self.attack_home = (self._wgroup(data, "home_team", "home_score") / self.home_mean).to_dict()
        self.defense_home = (self._wgroup(data, "home_team", "away_score") / self.away_mean).to_dict()
        self.attack_away = (self._wgroup(data, "away_team", "away_score") / self.away_mean).to_dict()
        self.defense_away = (self._wgroup(data, "away_team", "home_score") / self.home_mean).to_dict()

        logger.info("Dixon-Coles entrenado · %d partidos · ρ=%.3f ξ=%.3f · media local=%.2f visita=%.2f",
                    len(data), self.rho, self.xi, self.home_mean, self.away_mean)
        return self

    @staticmethod
    def _wmean(values: pd.Series, weights: pd.Series) -> float:
        """Promedio ponderado simple."""
        return float((values * weights).sum() / weights.sum())

    @staticmethod
    def _wgroup(data: pd.DataFrame, group_col: str, val_col: str) -> pd.Series:
        """Promedio ponderado de `val_col` por grupo `group_col` (peso en data['w'])."""
        num = (data[val_col] * data["w"]).groupby(data[group_col]).sum()
        den = data["w"].groupby(data[group_col]).sum()
        return num / den

    # ──────────────────────────────────────────────────────────────────────
    # Matriz de marcadores con la corrección τ
    # ──────────────────────────────────────────────────────────────────────
    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Matriz de Poisson con la corrección Dixon-Coles aplicada a las 4 celdas bajas."""
        lam_home, lam_away = self.expected_goals(home, away)
        ks = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(ks, lam_home), poisson.pmf(ks, lam_away))

        # τ solo toca (0,0),(0,1),(1,0),(1,1). Fila = local (x), columna = visita (y).
        r = self.rho
        m[0, 0] *= 1.0 - lam_home * lam_away * r
        m[0, 1] *= 1.0 + lam_home * r
        m[1, 0] *= 1.0 + lam_away * r
        m[1, 1] *= 1.0 - r

        # Tras τ la matriz ya no suma 1 exacto; renormalizamos para que most_likely_score
        # y prob_over sean coherentes (predict_proba normaliza igual, pero así todo cierra).
        return m / m.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.models.dixon_coles
# Compara contra el mercado: Poisson base vs Dixon-Coles (ρ) vs DC con decay (ρ+ξ).
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.evaluation.backtest import backtest_vs_market, temporal_split

    df = pd.read_parquet("data/processed/matches_top5.parquet")
    premier = df[df["league"] == "E0"]
    train, test = temporal_split(premier, "2024-08-01")

    candidatos = {
        "Poisson base":          PoissonModel(),
        "Dixon-Coles (ρ=-0.10)": DixonColesModel(rho=-0.10, xi=0.0),
        "DC + decay (ξ=0.30)":   DixonColesModel(rho=-0.10, xi=0.30),
    }

    print(f"\n=== Comparativa vs Mercado · Premier · {len(test)} partidos no vistos ===")
    print(f"{'modelo':<24}{'log-loss':>10}{'gap vs mercado':>16}")
    market_ll = None
    for nombre, modelo in candidatos.items():
        res = backtest_vs_market(modelo, train.copy(), test.copy())
        market_ll = res["market"]["log_loss"]
        print(f"{nombre:<24}{res['model']['log_loss']:>10.4f}{res['logloss_gap']:>+16.4f}")
    print(f"{'MERCADO':<24}{market_ll:>10.4f}{0.0:>+16.4f}")
    print("(menor log-loss = mejor; gap negativo = le ganás al mercado)")
