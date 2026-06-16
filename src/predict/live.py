"""
live.py — Simulación del Mundial CONDICIONADA a los resultados ya jugados.

El Monte Carlo de bracket.py re-simula TODOS los partidos desde cero, incluso los que
ya se jugaron — desperdicia información. Acá fijamos los resultados REALES conocidos
(los que el dataset ya trae con marcador) y simulamos SOLO lo pendiente: las jornadas
de grupos que faltan + todas las eliminatorias. Eso da la probabilidad de campeón
CONDICIONADA al estado real del torneo — más precisa a medida que avanza.

OJO conceptual (lo que más sims NO hacen): correr más simulaciones reduce el ERROR DE
MUESTREO (la prob converge a lo que el modelo cree), pero NO acerca la predicción al
resultado real ni reduce la aleatoriedad del fútbol. Ver convergence() abajo.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.predict.bracket import (
    GROUPS_BY_LETTER,
    KO_FEEDS,
    R32,
    ROUND_OF,
    _resolve_spec,
    assign_thirds,
    play_knockout,
)
from src.predict.simulate import (
    TeamRecord,
    _rank_group,
    best_eight_thirds,
    load_world_cup_fixtures,
    sample_score,
    train_elo_goal_model,
)

logger = logging.getLogger(__name__)


def _conditional_group_rows(fixtures: pd.DataFrame
                            ) -> dict[str, list[tuple]]:
    """
    Por letra de grupo, los partidos como (home, away, neutral, hs, as): hs/as enteros
    si YA se jugó (resultado fijo), o None si está pendiente (a simular).
    """
    out: dict[str, list[tuple]] = {}
    for letra, equipos in GROUPS_BY_LETTER.items():
        eqset = set(equipos)
        rows = []
        for r in fixtures.itertuples(index=False):
            if r.home_team in eqset and r.away_team in eqset:
                jugado = not pd.isna(r.home_score)
                rows.append((r.home_team, r.away_team, bool(r.neutral),
                             int(r.home_score) if jugado else None,
                             int(r.away_score) if jugado else None))
        out[letra] = rows
    return out


def _simulate_group_conditional(model, rows, rng):
    """Tabla de un grupo: usa el resultado real si el partido ya se jugó, si no lo simula."""
    teams = sorted({t for h, a, *_ in rows for t in (h, a)})
    records = {t: TeamRecord(t) for t in teams}
    for home, away, neutral, hs, as_ in rows:
        if hs is not None:
            ga, gb = hs, as_                      # resultado REAL conocido (condicionado)
        else:
            ga, gb = sample_score(model, home, away, neutral, rng)   # pendiente → simular
        records[home].add(ga, gb)
        records[away].add(gb, ga)
    return _rank_group(records, rng)


def monte_carlo_conditional(n: int = 10000, seed: int = 42, model=None) -> pd.DataFrame:
    """
    Monte Carlo CONDICIONADO: fija los partidos ya jugados y simula el resto. Mismas
    columnas que bracket.monte_carlo (prob por ronda + campeón), pero condicionado al
    estado real del torneo.
    """
    rng = np.random.default_rng(seed)
    model = model or train_elo_goal_model()
    fixtures = load_world_cup_fixtures()
    grows = _conditional_group_rows(fixtures)
    all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    counters = {r: Counter() for r in ["R32", "R16", "QF", "SF", "Final", "Champion"]}

    for _ in range(n):
        standings = {L: _simulate_group_conditional(model, rows, rng) for L, rows in grows.items()}
        thirds = assign_thirds(best_eight_thirds(standings, rng), rng)
        for L, tabla in standings.items():
            counters["R32"][tabla[0].team] += 1
            counters["R32"][tabla[1].team] += 1
        for _slot, L in thirds.items():
            counters["R32"][standings[L][2].team] += 1

        winners = {}
        for m, (hs, as_) in R32.items():
            winners[m] = play_knockout(model, _resolve_spec(hs, standings, thirds),
                                       _resolve_spec(as_, standings, thirds), rng)
        for m in sorted(KO_FEEDS):
            fh, fa = KO_FEEDS[m]
            winners[m] = play_knockout(model, winners[fh], winners[fa], rng)
        for m in ROUND_OF["R32"]:
            counters["R16"][winners[m]] += 1
        for m in ROUND_OF["R16"]:
            counters["QF"][winners[m]] += 1
        for m in ROUND_OF["QF"]:
            counters["SF"][winners[m]] += 1
        for m in ROUND_OF["SF"]:
            counters["Final"][winners[m]] += 1
        counters["Champion"][winners[104]] += 1

    tab = pd.DataFrame({"team": all_teams})
    for r in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
        tab[r] = [counters[r][t] / n for t in all_teams]
    return tab.sort_values("Champion", ascending=False).reset_index(drop=True)


def convergence(team: str, ns: list[int], reps: int = 5, seed0: int = 0) -> pd.DataFrame:
    """
    ¿Cuántas simulaciones hacen falta? Para varios N, corre `reps` Monte Carlos con
    seeds distintos y mide la dispersión de la prob de campeón de `team`. Cuando el
    desvío entre corridas se vuelve chico, ese N basta (más no aporta precisión útil).
    """
    model = train_elo_goal_model()       # entrenar una vez, reutilizar
    filas = []
    for N in ns:
        ps = []
        for k in range(reps):
            t = monte_carlo_conditional(n=N, seed=seed0 + k, model=model)
            ps.append(float(t.set_index("team")["Champion"].get(team, 0.0)))
        filas.append({"N": N, "media": np.mean(ps), "desvio": np.std(ps),
                      "min": np.min(ps), "max": np.max(ps)})
    return pd.DataFrame(filas)


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.live
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    pd.set_option("display.width", 120)

    from src.predict.bracket import monte_carlo

    N = 10000
    pre = monte_carlo(n=N).set_index("team")["Champion"]
    cond = monte_carlo_conditional(n=N).set_index("team")["Champion"]
    teams = cond.sort_values(ascending=False).head(10).index
    df = pd.DataFrame({"team": teams,
                       "pre_torneo": [pre.get(t, 0) * 100 for t in teams],
                       "condicionado": [cond.get(t, 0) * 100 for t in teams]}).round(1)
    print(f"=== Prob. de campeón: pre-torneo vs CONDICIONADO a lo jugado ({N} sims) ===\n")
    print(df.to_string(index=False))
