"""
validate_talent_historical.py — ¿El talento mejora la predicción de RESULTADOS? (de raíz)

La validación que faltaba. El talento agregado mejoraba el parecido al mercado y el
log-loss en 16 partidos del Mundial — muestra chica. Acá lo validamos sobre MILES de
partidos históricos, usando el valor de plantel VIGENTE en cada época (player_valuations
trae valores con fecha), reconstruido por (país, año).

MÉTODO (walk-forward honesto, sin leakage):
    Elo online entrenado con el pasado. Para cada partido, la PREDICCIÓN usa un rating
    ajustado por talento (blend del Elo con el log del valor de plantel de ese año),
    pero el Elo se ACTUALIZA con los ratings verdaderos (resultados). El valor de plantel
    de un año es información disponible antes del partido → no es leakage.

Comparamos el log-loss con w=0 (Elo puro) vs varios pesos de talento. Si baja con w>0
sobre miles de partidos, el talento mejora la predicción de verdad — no solo el mercado.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loaders.international_loader import InternationalResultsLoader
from src.models.elo import EloModel

logger = logging.getLogger(__name__)

_PS = Path("data/raw/player_scores")
# alias martj42 → nombre en el dataset Transfermarkt
_ALIAS_TO_TM = {
    "Ivory Coast": "Cote d'Ivoire", "South Korea": "Korea, South",
    "United States": "USA", "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina", "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
}


def squad_value_history(top_n: int = 23) -> tuple[dict, float, float]:
    """
    Reconstruye el valor de plantel por (país, año): top-N del último valor de cada
    jugador de esa nacionalidad en ese año. Devuelve (dict, media_log, std_log).
    """
    val = pd.read_csv(_PS / "player_valuations.csv", parse_dates=["date"])
    pl = pd.read_csv(_PS / "players.csv", usecols=["player_id", "country_of_citizenship"])
    v = val.merge(pl, on="player_id").dropna(subset=["country_of_citizenship"])
    v["year"] = v["date"].dt.year
    last = (v.sort_values("date")
            .groupby(["country_of_citizenship", "year", "player_id"])["market_value_in_eur"]
            .last().reset_index())
    sv = last.groupby(["country_of_citizenship", "year"]).apply(
        lambda g: g.nlargest(top_n, "market_value_in_eur")["market_value_in_eur"].sum(),
        include_groups=False)
    SV = {(c, y): float(x) for (c, y), x in sv.items()}
    logs = np.log([x for x in SV.values() if x > 0])
    return SV, float(logs.mean()), float(logs.std())


def _sv_lookup(SV: dict, team: str, year: int) -> float | None:
    """Valor de plantel del equipo en ese año (o el año previo si falta)."""
    name = _ALIAS_TO_TM.get(team, team)
    return SV.get((name, year)) or SV.get((name, year - 1))


def _predict_from_ratings(elo: EloModel, rh: float, ra: float, neutral: bool):
    """1X2 a partir de ratings EXPLÍCITOS (no toca los ratings internos del Elo)."""
    diff = rh + (0.0 if neutral else elo.home_advantage) - ra
    e = 1.0 / (1.0 + 10 ** (-diff / elo.scale))
    p_draw = elo._draw_prob(e)
    p_home = max(e - 0.5 * p_draw, 0.0)
    p_away = max((1.0 - e) - 0.5 * p_draw, 0.0)
    tot = p_home + p_draw + p_away
    return p_home / tot, p_draw / tot, p_away / tot


def walk_forward_talent(w: float, start: str = "2012-01-01",
                        SV=None, lmu=None, lsd=None, elo_spread: float = 180.0) -> tuple[float, int]:
    """
    Walk-forward sobre partidos >= start. La predicción usa rating ajustado por talento
    (peso w); el Elo se actualiza con resultados reales. Devuelve (log_loss, n_partidos).
    """
    if SV is None:
        SV, lmu, lsd = squad_value_history()
    df = (InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"])
          .sort_values("date"))
    warm, test = df[df["date"] < pd.Timestamp(start)], df[df["date"] >= pd.Timestamp(start)]
    elo = EloModel().fit(warm)

    def adj(team, base, year):
        if w == 0:
            return base
        s = _sv_lookup(SV, team, year)
        if not s or s <= 0:
            return base                      # sin dato de valor → Elo puro
        tz = (np.log(s) - lmu) / lsd
        return (1.0 - w) * base + w * (1500.0 + tz * elo_spread)

    ll = []
    for r in test.itertuples(index=False):
        yr = r.date.year
        neutral = bool(getattr(r, "neutral", False))
        rh = adj(r.home_team, elo.get_rating(r.home_team), yr)
        ra = adj(r.away_team, elo.get_rating(r.away_team), yr)
        ph, pd_, pa = _predict_from_ratings(elo, rh, ra, neutral)
        real = "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D")
        ll.append(-np.log(max({"H": ph, "D": pd_, "A": pa}[real], 1e-15)))
        elo.update_one(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                       neutral=neutral, is_competitive=bool(getattr(r, "is_competitive", True)))
    return float(np.mean(ll)), len(ll)


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.validate_talent_historical
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    SV, lmu, lsd = squad_value_history()
    print(f"Squad values reconstruidos: {len(SV)} país-año\n")
    print("Walk-forward 2012→2026 (miles de partidos) · log-loss por peso de talento:")
    base = None
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        ll, n = walk_forward_talent(w, SV=SV, lmu=lmu, lsd=lsd)
        if base is None:
            base = ll
        flecha = "" if w == 0 else ("↓ mejora" if ll < base else "↑ empeora")
        print(f"  w={w:<4} → log-loss {ll:.4f}  (n={n}) {flecha}")
    print(f"\n  (azar = {np.log(3):.4f})")
