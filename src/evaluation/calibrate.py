"""
calibrate.py — Calibración de hiperparámetros con validación temporal.

Acá elegimos los mejores (ρ, ξ) del Dixon-Coles SIN hacer trampa. La regla de oro:

    train / validation / test, en ese orden temporal.
      - train:      el modelo aprende ataque/defensa.
      - validation: probamos cada (ρ, ξ) y elegimos el de menor log-loss.
      - test:       SAGRADO. No se toca durante la calibración. Se usa una sola vez,
                    al final, con los parámetros ya fijados → único veredicto honesto.

Si eligieras los parámetros mirando el test, ese log-loss dejaría de ser honesto
(lo optimizaste al examen final) y el modelo rendiría peor con datos nuevos de verdad.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluation.metrics import log_loss
from src.models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)


def temporal_split_3(
    df: pd.DataFrame, cut1: str, cut2: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parte el dataset en train (<cut1), validation ([cut1,cut2)) y test (>=cut2)."""
    c1, c2 = pd.Timestamp(cut1), pd.Timestamp(cut2)
    train = df[df["date"] < c1]
    val = df[(df["date"] >= c1) & (df["date"] < c2)]
    test = df[df["date"] >= c2]
    return train, val, test


def model_logloss(model, df: pd.DataFrame) -> float:
    """Log-loss del modelo sobre df (necesita columnas home_team/away_team/result)."""
    df = df.dropna(subset=["result"])
    probs = np.array([model.predict_proba(r.home_team, r.away_team)
                      for r in df.itertuples(index=False)], dtype=float)
    return log_loss(df["result"].to_numpy(), probs)


def grid_search(
    train: pd.DataFrame,
    val: pd.DataFrame,
    rhos: list[float],
    xis: list[float],
) -> tuple[dict, list[dict]]:
    """
    Prueba cada combinación (ρ, ξ): entrena Dixon-Coles en train, mide log-loss en
    VALIDATION. Devuelve (mejor_combo, tabla_completa). El mejor = menor log-loss.
    """
    resultados: list[dict] = []
    for rho in rhos:
        for xi in xis:
            model = DixonColesModel(rho=rho, xi=xi).fit(train)
            ll = model_logloss(model, val)
            resultados.append({"rho": rho, "xi": xi, "val_logloss": ll})

    best = min(resultados, key=lambda r: r["val_logloss"])
    logger.info("Grid search: %d combos · mejor ρ=%.2f ξ=%.2f (val log-loss=%.4f)",
                len(resultados), best["rho"], best["xi"], best["val_logloss"])
    return best, resultados


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.evaluation.calibrate
# Calibra (ρ, ξ) en validation y dispara UN veredicto final en test vs mercado.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    from src.evaluation.backtest import backtest_vs_market

    df = pd.read_parquet("data/processed/matches_top5.parquet")
    premier = df[df["league"] == "E0"]

    # 3 vías: train hasta 2023/24, validation 2023/24, test 2024/25 + 2025/26.
    train, val, test = temporal_split_3(premier, "2023-08-01", "2024-08-01")
    print(f"Split temporal · train={len(train)}  val={len(val)}  test={len(test)}")

    rhos = [0.0, -0.05, -0.10, -0.15]
    xis = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    best, tabla = grid_search(train, val, rhos, xis)

    print(f"\nMejor en validation: ρ={best['rho']}  ξ={best['xi']}  "
          f"(val log-loss={best['val_logloss']:.4f})")

    # Mapa de log-loss en validation por ξ (con el mejor ρ), para ver la forma.
    print(f"\nlog-loss en validation por ξ (ρ={best['rho']}):")
    for r in tabla:
        if r["rho"] == best["rho"]:
            marca = "  <- mejor" if r["xi"] == best["xi"] else ""
            print(f"   ξ={r['xi']:<4} -> {r['val_logloss']:.4f}{marca}")

    # VEREDICTO FINAL: re-entreno con train+val usando los mejores params, evalúo TEST.
    train_val = pd.concat([train, val])
    final = DixonColesModel(rho=best["rho"], xi=best["xi"])
    res = backtest_vs_market(final, train_val, test)
    print(f"\n=== VEREDICTO en test ({res['n_test']} partidos, intocados hasta ahora) ===")
    print(f"Dixon-Coles calibrado: log-loss {res['model']['log_loss']:.4f}")
    print(f"Mercado:               log-loss {res['market']['log_loss']:.4f}")
    print(f"Gap final: {res['logloss_gap']:+.4f}  (era +0.0788 con el Poisson base)")
