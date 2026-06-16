"""
backtest.py — Backtesting temporal contra el mercado.

El juez final de un modelo. Hace lo que discutimos:

  1. SPLIT TEMPORAL: entrena con el pasado, evalúa con el futuro (partidos que el
     modelo NUNCA vio). Evaluar sobre lo que entrenaste = hacerte trampa.

  2. COMPARA CONTRA EL MERCADO: las odds de cierre (ya de-vigged en p_market_*) son
     el mejor predictor público que existe. Si tu modelo no se le acerca en log-loss,
     está flojo. El mercado es la vara de la realidad.

Funciona con cualquier modelo que tenga .fit(df) y .predict_proba(home, away) ->
(p_H, p_D, p_A). Nuestro PoissonModel cumple esa interfaz.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASSES, evaluate

logger = logging.getLogger(__name__)


def temporal_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parte el dataset por fecha: train = antes del cutoff, test = desde el cutoff.
    `cutoff` es una fecha 'YYYY-MM-DD'. El orden temporal es sagrado.
    """
    fecha = pd.Timestamp(cutoff)
    train = df[df["date"] < fecha]
    test = df[df["date"] >= fecha]
    return train, test


def _market_probs(test: pd.DataFrame) -> np.ndarray:
    """Las probabilidades del mercado (ya de-vigged) como array (N, 3) en orden H/D/A."""
    return test[["p_market_h", "p_market_d", "p_market_a"]].to_numpy(dtype=float)


def _model_probs(model, test: pd.DataFrame) -> np.ndarray:
    """Predice cada partido del test con el modelo, en el mismo orden H/D/A."""
    filas = [model.predict_proba(r.home_team, r.away_team)
             for r in test.itertuples(index=False)]
    return np.array(filas, dtype=float)


def backtest_vs_market(model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Entrena el modelo en train, lo evalúa en test, y compara CONTRA el mercado sobre
    los MISMOS partidos. Devuelve métricas lado a lado más la diferencia de log-loss.

    Solo evaluamos partidos del test que tengan odds de mercado (comparación justa).
    """
    model.fit(train)

    # Comparación justa: solo partidos con odds y con resultado conocido.
    test = test.dropna(subset=["p_market_h", "p_market_d", "p_market_a", "result"]).copy()
    if test.empty:
        raise ValueError("El test no tiene partidos con odds de mercado y resultado.")

    y_true = test["result"].to_numpy()
    model_p = _model_probs(model, test)
    market_p = _market_probs(test)

    model_metrics = evaluate(y_true, model_p)
    market_metrics = evaluate(y_true, market_p)

    # diff > 0 => el mercado es mejor (menor log-loss); < 0 => le ganamos al mercado.
    gap = model_metrics["log_loss"] - market_metrics["log_loss"]

    logger.info("Backtest · train=%d test=%d · modelo log-loss=%.4f vs mercado=%.4f (gap=%+.4f)",
                len(train), len(test), model_metrics["log_loss"],
                market_metrics["log_loss"], gap)

    return {
        "classes": CLASSES,
        "model": model_metrics,
        "market": market_metrics,
        "logloss_gap": gap,           # negativo = el modelo le gana al mercado
        "n_test": model_metrics["n"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.evaluation.backtest
# Poisson sobre la Premier: entrena 2019→jul-2024, evalúa 2024/25 + 2025/26.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.evaluation.metrics import calibration_table
    from src.models.poisson import PoissonModel

    df = pd.read_parquet("data/processed/matches_top5.parquet")
    premier = df[df["league"] == "E0"]
    train, test = temporal_split(premier, "2024-08-01")

    res = backtest_vs_market(PoissonModel(), train, test)

    print(f"\n=== Backtest Poisson vs Mercado · Premier · {res['n_test']} partidos no vistos ===")
    print(f"{'métrica':<12}{'MODELO':>12}{'MERCADO':>12}")
    for k in ("log_loss", "brier", "accuracy"):
        print(f"{k:<12}{res['model'][k]:>12.4f}{res['market'][k]:>12.4f}")
    veredicto = "el modelo LE GANA al mercado" if res["logloss_gap"] < 0 else "el mercado gana"
    print(f"\nGap de log-loss: {res['logloss_gap']:+.4f}  →  {veredicto}")
    print("(menor log-loss = mejor; el mercado es dificilísimo de batir)")
