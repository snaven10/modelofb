"""
talent.py — Ajuste del modelo por TALENTO de plantel (valor de mercado).

POR QUÉ:
    La validación reveló que el Elo (basado en resultados) subestima a equipos con
    planteles caros — porque no "ve" el talento individual. El valor de mercado total
    por selección (Transfermarkt) es la señal que falta. Mezclar Elo + valor acerca el
    modelo al consenso del mercado: correlación con el mercado 0.864 → 0.942 al 50/50.

CÓMO:
    Blend en z-space (ambas señales estandarizadas), de vuelta a la escala Elo:
        rating_ajustado = ((1-w)·z(Elo) + w·z(log valor)) · std(Elo) + media(Elo)
    w=0 → Elo puro; w=1 → puro talento; w≈0.5 → máxima correlación con el mercado.

HALLAZGO CRÍTICO (validación robusta — corrige el w):
    La validación contra RESULTADOS reales (13.851 partidos 2012-2026, ver
    validate_talent_historical.py) reveló que el peso óptimo es w≈0.1 con una mejora
    MÍNIMA (log-loss 0.8990→0.8975, ~0.17%). El w=0.5 (que MAXIMIZA el parecido al
    mercado) EMPEORA la predicción de resultados (0.8990→0.9094). Lección: "parecerse al
    mercado" ≠ "predecir mejor" — el mercado sobrevalora el talento. El Elo ya captura
    casi toda la señal (los resultados ya reflejan el talento indirectamente); meter
    talento con peso alto sobre-pondera el plantel y arruina la predicción. Por eso el
    DEFAULT es w=0.1 (óptimo predictivo), no 0.5. Usá w=0.5 SOLO si querés el ranking
    "alineado al mercado" (cosmético), no para predecir.
"""

from __future__ import annotations

import logging

import numpy as np

from src.data_loaders.squad_values import squad_values
from src.models.elo import EloModel

logger = logging.getLogger(__name__)


def adjust_elo_by_talent(elo: EloModel, w: float = 0.5) -> EloModel:
    """
    Reescribe (in-place) los ratings de las selecciones del Mundial mezclando su Elo
    con el log de su valor de plantel. Devuelve el mismo EloModel, ajustado.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w debe estar en [0, 1], no {w}")
    sv = squad_values()
    teams = list(sv.keys())
    elo_r = np.array([elo.get_rating(t) for t in teams], dtype=float)
    val = np.log(np.array([sv[t] for t in teams], dtype=float))

    mu, sd = elo_r.mean(), elo_r.std()
    ez = (elo_r - mu) / sd
    vz = (val - val.mean()) / val.std()
    comb = (1.0 - w) * ez + w * vz
    new_r = comb * sd + mu          # de vuelta a la escala Elo

    for t, r in zip(teams, new_r):
        elo.ratings[t] = float(r)
    logger.info("Elo ajustado por talento (w=%.2f) para %d selecciones", w, len(teams))
    return elo


def champion_with_talent(w: float = 0.1, n: int = 10000, seed: int = 42):
    """
    Monte Carlo del Mundial con el modelo de goles ajustado por talento de plantel.
    Default w=0.1 (óptimo PREDICTIVO validado en 13.851 partidos). Pasá w=0.5 para el
    ranking "alineado al mercado" (sube Francia/Inglaterra) — cosmético, no predictivo.
    """
    from src.predict.bracket import monte_carlo
    from src.predict.simulate import train_elo_goal_model

    egm = train_elo_goal_model()
    adjust_elo_by_talent(egm.elo, w)        # ajusta el Elo interno del modelo de goles
    return monte_carlo(n=n, seed=seed, model=egm)


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.talent
# Compara la prob de campeón SIN y CON talento, contra el mercado.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    import pandas as pd

    from src.data_loaders.odds_api_loader import OddsApiLoader
    from src.predict.bracket import monte_carlo

    N = 10000
    sin = monte_carlo(n=N).set_index("team")["Champion"]
    con = champion_with_talent(w=0.5, n=N).set_index("team")["Champion"]
    mkt = OddsApiLoader().world_cup_winner_market()
    alias = {"USA": "United States", "Czechia": "Czech Republic", "IR Iran": "Iran"}
    mkt = {alias.get(k, k): v for k, v in mkt.items()}

    teams = list(mkt.keys())
    df = pd.DataFrame({
        "team": teams,
        "sin_talento": [sin.get(t, 0.0) for t in teams],
        "con_talento": [con.get(t, 0.0) for t in teams],
        "mercado": [mkt[t] for t in teams],
    }).sort_values("mercado", ascending=False).reset_index(drop=True)

    show = df.head(10).copy()
    for c in ("sin_talento", "con_talento", "mercado"):
        show[c] = (show[c] * 100).round(1)
    print("=== Prob. de campeón: SIN vs CON talento vs MERCADO (%) ===\n")
    print(show.to_string(index=False))

    c_sin = df["sin_talento"].corr(df["mercado"], method="spearman")
    c_con = df["con_talento"].corr(df["mercado"], method="spearman")
    print(f"\nCorrelación con el mercado:  sin talento {c_sin:.3f}  →  con talento {c_con:.3f}")
