"""
verify.py — Verificación retrospectiva: ¿le acertó el modelo al Mundial real?

LA IDEA (la prueba de fuego honesta):
    Ya hay partidos JUGADOS del Mundial 2026. Podemos comprobar si el modelo los
    habría acertado... pero SIN trampa. Si el Elo se entrena con TODO el dataset (que
    ya incluye esos resultados), "predecirlos" sería hacer trampa: el modelo ya los vio.

    Por eso entrenamos el Elo SOLO con partidos ANTERIORES al inicio del Mundial
    (< 2026-06-11). Ningún partido a verificar está en el entrenamiento → cero leakage.
    Luego predecimos cada partido ya jugado y comparamos con el resultado REAL.

QUÉ REPORTAMOS (dos medidas, porque el acierto solo no basta):
    - ACIERTO 1X2: ¿el resultado real (H/D/A) fue el de mayor probabilidad? Ojo: el
      Elo rara vez hace al empate el más probable, así que los empates suelen "fallar"
      aunque la probabilidad asignada fuera razonable.
    - LOG-LOSS: castiga la confianza equivocada; es la medida justa para probabilidades.
      Lo comparamos contra el azar (ln 3 ≈ 1.0986) para saber si el modelo aporta señal.

REFINAMIENTO POSIBLE (no hecho): walk-forward — actualizar el Elo con cada jornada del
Mundial antes de predecir la siguiente. Acá usamos un corte fijo pre-torneo, más simple
y igual de honesto (sin leakage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_loaders.international_loader import InternationalResultsLoader
from src.evaluation.metrics import calibration_table, evaluate, log_loss
from src.models.elo import EloModel

logger = logging.getLogger(__name__)

WORLD_CUP_START = "2026-06-11"


def verify_world_cup(cutoff: str = WORLD_CUP_START) -> tuple[pd.DataFrame, dict]:
    """
    Entrena el Elo con partidos < cutoff y predice los partidos del Mundial ya jugados.
    Devuelve (tabla partido a partido, métricas agregadas).
    """
    df = InternationalResultsLoader().load(include_unplayed=True)

    train = df[(df["date"] < pd.Timestamp(cutoff))].dropna(subset=["home_score", "away_score"])
    played = df[
        (df["date"] >= pd.Timestamp(cutoff))
        & (df["tournament"].str.contains("World Cup", case=False, na=False))
        & (df["home_score"].notna())
    ].copy()
    if played.empty:
        raise ValueError("No hay partidos del Mundial ya jugados para verificar.")

    model = EloModel().fit(train)

    filas, probs_real, aciertos = [], [], 0
    for r in played.itertuples(index=False):
        ph, pdraw, pa = model.predict_proba(r.home_team, r.away_team, neutral=bool(r.neutral))
        # Resultado real H/D/A
        if r.home_score > r.away_score:
            real = "H"
        elif r.home_score < r.away_score:
            real = "A"
        else:
            real = "D"
        # Predicción 1X2 (la más probable)
        etiquetas = ["H", "D", "A"]
        pred = etiquetas[int(np.argmax([ph, pdraw, pa]))]
        ok = pred == real
        aciertos += ok
        probs_real.append({"H": ph, "D": pdraw, "A": pa}[real])
        filas.append({
            "date": r.date.date(),
            "partido": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
            "real": real,
            "pred": pred,
            "p_real": round({"H": ph, "D": pdraw, "A": pa}[real], 3),
            "ok": "✓" if ok else "✗",
        })

    tabla = pd.DataFrame(filas)
    n = len(played)
    metrics = {
        "n": n,
        "aciertos": aciertos,
        "accuracy": aciertos / n,
        "log_loss": float(-np.mean(np.log(np.clip(probs_real, 1e-15, 1)))),
        "log_loss_azar": float(np.log(3)),
    }
    return tabla, metrics


def walk_forward_verify(start: str = "2018-01-01",
                        competitive_only: bool = False) -> tuple[dict, np.ndarray]:
    """
    Verificación WALK-FORWARD sobre muchos partidos (la prueba estadísticamente seria).

    Recorre los partidos en orden cronológico: ANTES de cada uno predice con el Elo
    entrenado solo con el pasado, registra, y DESPUÉS lo actualiza. Así cada predicción
    usa exclusivamente información previa — cero leakage, sobre miles de partidos.

    El Elo se "calienta" con todo lo anterior a `start` (eso fija también la curva de
    empate, calibrada solo con el pasado). Durante el test solo se actualizan ratings.

    Devuelve (métricas, tabla_de_calibración).
    """
    df = (InternationalResultsLoader().load()
          .dropna(subset=["home_score", "away_score"])
          .sort_values("date"))
    warmup = df[df["date"] < pd.Timestamp(start)]
    test = df[df["date"] >= pd.Timestamp(start)]
    if competitive_only:
        test = test[test["is_competitive"]]
    if warmup.empty or test.empty:
        raise ValueError("Rango insuficiente para walk-forward.")

    elo = EloModel().fit(warmup)          # ratings + curva de empate, solo con el pasado

    y, probs = [], []
    for r in test.itertuples(index=False):
        neutral = bool(getattr(r, "neutral", False))
        ph, pdraw, pa = elo.predict_proba(r.home_team, r.away_team, neutral=neutral)
        real = "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D")
        y.append(real)
        probs.append([ph, pdraw, pa])
        # walk-forward: actualizar DESPUÉS de predecir (la predicción no vio este partido)
        elo.update_one(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                       neutral=neutral, is_competitive=bool(getattr(r, "is_competitive", True)))

    y_arr, probs_arr = np.array(y), np.array(probs)
    metrics = evaluate(y_arr, probs_arr)
    metrics["log_loss_azar"] = float(np.log(3))
    # empates vs decisivos
    empate_mask = y_arr == "D"
    metrics["pct_empates"] = float(empate_mask.mean())
    return metrics, calibration_table(y_arr, probs_arr, n_bins=10)


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.verify
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    pd.set_option("display.width", 120)

    tabla, m = verify_world_cup()
    print(f"=== Verificación: {m['n']} partidos jugados del Mundial 2026 ===")
    print("(Elo entrenado solo con partidos PRE-Mundial — sin leakage)\n")
    print(tabla.to_string(index=False))
    print(f"\nAciertos 1X2: {m['aciertos']}/{m['n']} ({m['accuracy']*100:.1f}%)")
    print(f"Log-loss del modelo: {m['log_loss']:.4f}  vs  azar: {m['log_loss_azar']:.4f}")
    veredicto = ("le gana al azar (aporta señal)" if m["log_loss"] < m["log_loss_azar"]
                 else "NO le gana al azar")
    print(f"→ El modelo {veredicto}.")
