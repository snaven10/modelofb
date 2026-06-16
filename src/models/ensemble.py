"""
ensemble.py — Combina varios modelos 1X2 en una sola predicción.

¿POR QUÉ UN ENSEMBLE?
    Dos modelos que aciertan distinto, combinados, aciertan más. El Elo mide fuerza
    relativa acumulada (quién le gana a quién); el Poisson/Dixon-Coles mide ataque y
    defensa vía goles. Capturan señales diferentes, así que sus errores no están
    perfectamente correlacionados: al promediarlos, parte del error se cancela.

DOS FORMAS DE COMBINAR (elegís con `method`):
    - "arithmetic":  p = w·p_A + (1−w)·p_B            (promedio lineal, intuitivo)
    - "geometric":   p ∝ p_A^w · p_B^(1−w)  (renormalizado)
      El promedio geométrico (log-pooling) suele ganarle al aritmético en LOG-LOSS,
      porque la métrica que optimizamos premia justo esta forma de promediar y castiga
      con dureza que un modelo asigne ~0 a algo que ocurre.

EL PESO w (hiperparámetro, calibrable):
    w=1 → solo el primer modelo; w=0 → solo el segundo; en medio, la mezcla. Se elige
    con el MISMO grid search en validation que usamos para Dixon-Coles (sin tocar test).

INTERFAZ: cualquier modelo con .fit(df) y .predict_proba(home, away) -> (pH, pD, pA).
Tanto EloModel como PoissonModel/DixonColesModel la cumplen.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-12  # piso para no hacer log(0) en el promedio geométrico.


class EnsembleModel:
    """Combina exactamente dos modelos 1X2 con un peso w y un método de mezcla."""

    def __init__(self, model_a, model_b, weight: float = 0.5,
                 method: str = "geometric") -> None:
        """
        model_a, model_b  los dos modelos a combinar (ya construidos, sin entrenar).
        weight            w ∈ [0, 1]: cuánto pesa model_a. (1−w) pesa model_b.
        method            "geometric" (log-pooling, default) o "arithmetic".
        """
        if method not in ("geometric", "arithmetic"):
            raise ValueError(f"method debe ser 'geometric' o 'arithmetic', no {method!r}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight debe estar en [0, 1], no {weight}")
        self.model_a = model_a
        self.model_b = model_b
        self.weight = weight
        self.method = method

    def fit(self, df: pd.DataFrame) -> "EnsembleModel":
        """Entrena AMBOS sub-modelos sobre el mismo dataset (mismos equipos)."""
        self.model_a.fit(df)
        self.model_b.fit(df)
        return self

    def predict_proba(self, home: str, away: str) -> tuple[float, float, float]:
        """Combina las dos predicciones 1X2 según method y weight. Suma 1."""
        pa = np.asarray(self.model_a.predict_proba(home, away), dtype=float)
        pb = np.asarray(self.model_b.predict_proba(home, away), dtype=float)
        w = self.weight

        if self.method == "arithmetic":
            combinado = w * pa + (1.0 - w) * pb
        else:  # geometric / log-pooling
            # p ∝ pa^w · pb^(1−w). Lo hacemos en espacio log para estabilidad.
            log_mix = w * np.log(pa + _EPS) + (1.0 - w) * np.log(pb + _EPS)
            combinado = np.exp(log_mix)

        combinado = combinado / combinado.sum()  # renormaliza a probabilidad
        return float(combinado[0]), float(combinado[1]), float(combinado[2])


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.models.ensemble
# Calibra w en validation y dispara el veredicto en test: Elo solo vs DC solo vs
# Ensemble, todos contra el mercado (Premier, partidos no vistos).
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    from src.evaluation.backtest import backtest_vs_market
    from src.evaluation.calibrate import model_logloss, temporal_split_3
    from src.models.dixon_coles import DixonColesModel
    from src.models.elo import EloModel

    df = pd.read_parquet("data/processed/matches_top5.parquet")
    premier = df[df["league"] == "E0"]
    train, val, test = temporal_split_3(premier, "2023-08-01", "2024-08-01")
    print(f"Split temporal · train={len(train)}  val={len(val)}  test={len(test)}")

    # Entrenamos cada modelo UNA vez en train para calibrar w en validation.
    elo_v = EloModel().fit(train)
    dc_v = DixonColesModel(rho=0.0, xi=0.5).fit(train)  # mejor (ρ,ξ) ya calibrado

    print("\nCalibrando w (geométrico) en validation:")
    best_w, best_ll = 0.0, float("inf")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ens = EnsembleModel(elo_v, dc_v, weight=w, method="geometric")
        ll = model_logloss(ens, val)
        marca = ""
        if ll < best_ll:
            best_ll, best_w = ll, w
        print(f"   w={w:<4} (w·Elo + (1-w)·DC) -> val log-loss={ll:.4f}")
    print(f"\nMejor w={best_w} (val log-loss={best_ll:.4f})")

    # VEREDICTO en test intocado: cada modelo solo vs el ensemble, contra el mercado.
    train_val = pd.concat([train, val])
    print(f"\n=== VEREDICTO en test ({len(test)} partidos no vistos) ===")
    for nombre, modelo in [
        ("Elo solo", EloModel()),
        ("Dixon-Coles solo", DixonColesModel(rho=0.0, xi=0.5)),
        (f"Ensemble (w={best_w})",
         EnsembleModel(EloModel(), DixonColesModel(rho=0.0, xi=0.5),
                       weight=best_w, method="geometric")),
    ]:
        res = backtest_vs_market(modelo, train_val, test)
        print(f"{nombre:<22} log-loss {res['model']['log_loss']:.4f}  "
              f"gap vs mercado {res['logloss_gap']:+.4f}")
    print(f"{'Mercado':<22} log-loss {res['market']['log_loss']:.4f}")
