"""
metrics.py — Métricas para evaluar predicciones probabilísticas 1X2.

Acá viven los JUECES de nuestros modelos. Todas trabajan sobre el mismo formato:

    y_true : array de etiquetas reales, cada una "H" / "D" / "A".
    probs  : array (N, 3) con las probabilidades [P(H), P(D), P(A)] por partido,
             SIEMPRE en ese orden de columnas (CLASSES).

Las métricas clave (log-loss y Brier) son "menor = mejor": miden ERROR, no acierto.
"""

from __future__ import annotations

import numpy as np

# Orden canónico de clases. TODO el proyecto usa este orden para las columnas de probs.
CLASSES = ("H", "D", "A")
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def _one_hot(y_true: np.ndarray) -> np.ndarray:
    """Convierte etiquetas H/D/A en filas one-hot, p.ej. 'D' -> [0, 1, 0]."""
    idx = np.array([_CLASS_IDX[y] for y in y_true])
    oh = np.zeros((len(y_true), len(CLASSES)))
    oh[np.arange(len(y_true)), idx] = 1.0
    return oh


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Log-loss (cross-entropy). LA métrica reina. Penaliza fuerte la confianza
    equivocada: estar seguro de algo falso te destroza. Menor = mejor.

    El `eps` (clip) evita log(0) = −infinito si el modelo asignó probabilidad 0
    exacta al resultado que sí ocurrió. Sin esto, un solo partido te rompe la métrica.
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    idx = np.array([_CLASS_IDX[y] for y in y_true])
    # Probabilidad que el modelo le dio al resultado REAL de cada partido.
    p_real = probs[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(p_real)))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier multiclase: error cuadrático medio entre el vector de probabilidades y la
    realidad one-hot. Más suave que el log-loss. Menor = mejor. Rango [0, 2].
    """
    probs = np.asarray(probs, dtype=float)
    oh = _one_hot(np.asarray(y_true))
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Fracción de partidos donde la clase MÁS probable fue la real. Engañosa para
    probabilidades (no distingue 51% de 99%), la reportamos solo de contexto.
    """
    probs = np.asarray(probs, dtype=float)
    pred_idx = np.argmax(probs, axis=1)
    true_idx = np.array([_CLASS_IDX[y] for y in y_true])
    return float(np.mean(pred_idx == true_idx))


def evaluate(y_true, probs) -> dict[str, float]:
    """Las tres métricas de un saque, para reportar un modelo de un vistazo."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    return {
        "log_loss": log_loss(y_true, probs),
        "brier": brier_score(y_true, probs),
        "accuracy": accuracy(y_true, probs),
        "n": int(len(y_true)),
    }


def calibration_table(y_true, probs, n_bins: int = 10) -> np.ndarray:
    """
    Curva de fiabilidad (one-vs-rest agregada). Aplana TODAS las predicciones por
    clase en pares (prob_predicha, ocurrió_o_no) y los agrupa en bins de
    probabilidad. En cada bin compara la prob promedio predicha vs la frecuencia
    real observada: si están cerca, el modelo está calibrado en ese rango.

    Devuelve un array (n_bins, 3): [prob_media_predicha, frecuencia_real, conteo].
    Los bins vacíos quedan en NaN. Un modelo calibrado tiene col0 ≈ col1.
    """
    probs = np.asarray(probs, dtype=float).ravel()         # todas las probs por clase
    hits = _one_hot(np.asarray(y_true)).ravel()            # 1 si esa clase ocurrió
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin de cada predicción; -1 para quedar dentro de [0, n_bins-1].
    bin_idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)

    out = np.full((n_bins, 3), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count > 0:
            out[b] = [probs[mask].mean(), hits[mask].mean(), count]
    return out
