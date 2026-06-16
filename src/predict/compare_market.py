"""
compare_market.py — Contrasta la predicción del modelo contra el mercado real.

La validación más directa: nuestra probabilidad de campeón (Monte Carlo) vs la
probabilidad IMPLÍCITA del mercado (odds outright de The Odds API, sin overround).
Donde discrepan, el modelo aprende algo de sí mismo: el Elo solo ve resultados, el
mercado ve también plantel/talento/lesiones.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_loaders.odds_api_loader import OddsApiLoader

logger = logging.getLogger(__name__)

# Alias por si The Odds API nombra distinto que martj42 (se completa si aparecen).
_ALIAS = {
    "USA": "United States", "South Korea": "South Korea", "Turkey": "Turkey",
    "IR Iran": "Iran", "Czechia": "Czech Republic",
}


def _norm(name: str) -> str:
    return _ALIAS.get(name, name)


def compare_champion(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Tabla comparativa modelo vs mercado por selección, ordenada por prob de mercado.
    Incluye la diferencia (modelo − mercado): positivo = creemos MÁS que el mercado.
    """
    from src.predict.bracket import monte_carlo

    modelo = monte_carlo(n=n, seed=seed).set_index("team")["Champion"]
    mercado = OddsApiLoader().world_cup_winner_market()
    mercado = {_norm(t): p for t, p in mercado.items()}

    filas = []
    for team, p_mkt in mercado.items():
        p_mod = float(modelo.get(team, 0.0))
        filas.append({"team": team, "p_modelo": p_mod, "p_mercado": p_mkt,
                      "dif": p_mod - p_mkt})
    tabla = pd.DataFrame(filas).sort_values("p_mercado", ascending=False).reset_index(drop=True)
    return tabla


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.compare_market
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    pd.set_option("display.width", 120)

    t = compare_champion(n=10000)
    show = t.head(12).copy()
    for c in ("p_modelo", "p_mercado", "dif"):
        show[c] = (show[c] * 100).round(1)
    print("=== Prob. de campeón: MODELO vs MERCADO (%) ===\n")
    print(show.to_string(index=False))

    # Correlación de rangos: ¿ordenamos parecido al mercado?
    corr = t["p_modelo"].corr(t["p_mercado"], method="spearman")
    print(f"\nCorrelación de Spearman (orden) modelo vs mercado: {corr:.3f}")
    sobre = t.nlargest(3, "dif")[["team", "dif"]]
    sub = t.nsmallest(3, "dif")[["team", "dif"]]
    print("\nDonde creemos MÁS que el mercado:",
          ", ".join(f"{r.team} {r.dif*100:+.1f}%" for r in sobre.itertuples()))
    print("Donde creemos MENOS que el mercado:",
          ", ".join(f"{r.team} {r.dif*100:+.1f}%" for r in sub.itertuples()))
