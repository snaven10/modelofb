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

from src.data_loaders.international_loader import InternationalResultsLoader
from src.data_loaders.odds_api_loader import OddsApiLoader
from src.models.elo import EloModel

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


def compare_matches(model: EloModel | None = None) -> pd.DataFrame:
    """
    Compara, partido a partido, nuestra predicción 1X2 contra las odds del mercado de
    los próximos partidos del Mundial (los que The Odds API tiene con cuota abierta).

    Validación PROSPECTIVA: aún no se jugaron, así que no sabemos quién acertó; medimos
    si COINCIDIMOS con el mercado (mismo favorito, probabilidades parecidas). El Elo se
    entrena con todo lo jugado hasta hoy (incluidas las jornadas del Mundial ya jugadas).
    """
    odds = OddsApiLoader().world_cup_match_market()
    if odds.empty:
        raise ValueError("The Odds API no devolvió partidos del Mundial con odds 1X2.")

    df = InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"])
    model = model or EloModel().fit(df)

    etiquetas = ["H", "D", "A"]
    filas = []
    for r in odds.itertuples(index=False):
        h, a = _norm(r.home_team), _norm(r.away_team)
        ph, pd_, pa = model.predict_proba(h, a, neutral=True)   # Mundial: cancha neutral
        fav_mod = etiquetas[[ph, pd_, pa].index(max(ph, pd_, pa))]
        m = [r.p_market_h, r.p_market_d, r.p_market_a]
        fav_mkt = etiquetas[m.index(max(m))]
        filas.append({
            "partido": f"{r.home_team} vs {r.away_team}",
            "fav_modelo": fav_mod, "fav_mercado": fav_mkt,
            "coincide": fav_mod == fav_mkt,
            "p_local_mod": round(ph, 3), "p_local_mkt": round(r.p_market_h, 3),
        })
    return pd.DataFrame(filas)


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
