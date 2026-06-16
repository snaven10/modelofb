"""
test_metrics.py — Pruebas de las métricas de evaluación.

Verificamos propiedades CONCEPTUALES con valores calculables a mano: que el log-loss
castigue más la confianza equivocada, que una predicción perfecta dé error 0, y que
log-loss/Brier coincidan con su fórmula en casos conocidos.

Correlo con:  pytest -q
"""

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_table,
    evaluate,
    log_loss,
)


def test_log_loss_prediccion_perfecta_es_casi_cero():
    """Si das ~100% a lo que pasó, el log-loss tiende a 0."""
    y = np.array(["H", "A"])
    probs = np.array([[0.999999, 0.0000005, 0.0000005],
                      [0.0000005, 0.0000005, 0.999999]])
    assert log_loss(y, probs) == pytest.approx(0.0, abs=1e-4)


def test_log_loss_castiga_la_confianza_equivocada():
    """Errar con 99% de confianza debe doler MUCHO más que errar dudando (40%)."""
    y = np.array(["H"])
    seguro_y_mal = np.array([[0.005, 0.005, 0.99]])   # apostó fuerte al visitante; ganó local
    dudoso = np.array([[0.40, 0.20, 0.40]])
    assert log_loss(y, seguro_y_mal) > log_loss(y, dudoso)


def test_log_loss_valor_concreto():
    """Una sola predicción de 0.5 al resultado real -> log-loss = −ln(0.5)."""
    y = np.array(["D"])
    probs = np.array([[0.25, 0.50, 0.25]])
    assert log_loss(y, probs) == pytest.approx(-math.log(0.5), abs=1e-9)


def test_log_loss_no_explota_con_cero():
    """Probabilidad 0 al resultado real NO debe dar infinito (gracias al clip)."""
    y = np.array(["H"])
    probs = np.array([[0.0, 0.5, 0.5]])
    assert np.isfinite(log_loss(y, probs))


def test_brier_prediccion_perfecta_es_cero():
    y = np.array(["H"])
    probs = np.array([[1.0, 0.0, 0.0]])
    assert brier_score(y, probs) == pytest.approx(0.0)


def test_brier_valor_concreto():
    """[0.6,0.25,0.15] con ganador local: (0.6−1)²+0.25²+0.15² = 0.245."""
    y = np.array(["H"])
    probs = np.array([[0.6, 0.25, 0.15]])
    assert brier_score(y, probs) == pytest.approx(0.245, abs=1e-9)


def test_accuracy_cuenta_argmax():
    y = np.array(["H", "D", "A"])
    probs = np.array([[0.7, 0.2, 0.1],   # predice H, real H  -> acierto
                      [0.2, 0.3, 0.5],   # predice A, real D  -> fallo
                      [0.1, 0.2, 0.7]])  # predice A, real A  -> acierto
    assert accuracy(y, probs) == pytest.approx(2 / 3)


def test_evaluate_devuelve_las_tres():
    y = np.array(["H", "A"])
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    res = evaluate(y, probs)
    assert set(res) == {"log_loss", "brier", "accuracy", "n"}
    assert res["n"] == 2


def test_calibracion_modelo_perfecto():
    """
    Modelo perfectamente calibrado: en CADA bin poblado, la prob promedio predicha
    debe coincidir con la frecuencia real observada (col0 ≈ col1).

    Setup: das 0.7 a H y 0.15 a D y A en 10 partidos; gana H en 7 y A en 3. Así el
    0.7 ocurre el 70% y el 0.15 (de A) el 30%... todo cuadra por construcción.
    (No hardcodeamos el índice del bin: linspace mete error de float en los bordes.)
    """
    y = np.array(["H"] * 7 + ["A"] * 3)
    probs = np.tile([0.7, 0.15, 0.15], (10, 1))
    tabla = calibration_table(y, probs, n_bins=10)

    poblados = tabla[~np.isnan(tabla[:, 0])]
    assert len(poblados) >= 1
    for prob_predicha, freq_real, _ in poblados:
        assert prob_predicha == pytest.approx(freq_real, abs=1e-9)
